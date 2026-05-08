"""Differential cross-provider evaluation.

`diff_eval` runs the same `EvalCase` set against multiple runners (a dict of
``name -> Runner``) in parallel and surfaces where they disagree.

The unit of parallelism is the runner: each runner walks its full case list
sequentially (via ``run_eval``-style iteration) and the runner-level coroutines
are gathered via ``asyncio.gather(..., return_exceptions=True)``. This makes
wall-time roughly ``slowest_runner * len(cases)`` and isolates failures so one
flaky provider does not kill the whole matrix.

Outlier detection re-uses ``compare_sessions``, which already strips tool-call
ids so identical behaviour across runners is recognised even when the model
emits different ids.
"""

from __future__ import annotations

import asyncio
import html
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from harness.agents.definition import SubAgent
from harness.agents.orchestrator import Orchestrator, Runner
from harness.hooks.runner import HookRunner
from harness.memory.record import SessionRecord
from harness.memory.session import Session
from harness.memory.store import InMemoryStore
from harness.prompts.messages import Message
from harness.replay.harness import EvalCase, SessionDiff, compare_sessions
from harness.tools.dispatcher import Dispatcher

if TYPE_CHECKING:
    from harness.telemetry.recorder import Telemetry

__all__ = ["DiffMatrix", "DiffOutlier", "diff_eval"]


_TEMPLATE_PATH = Path(__file__).parent / "templates" / "diff_report.html"


# ---------------------------------------------------------------------------
# Public dataclasses


@dataclass(frozen=True)
class DiffOutlier:
    """A single (case, dissenting-runner) disagreement.

    ``consensus_runners`` is the largest cluster of runners whose pairwise
    ``compare_sessions`` all match. ``dissenting_runner`` is one runner whose
    output disagrees with that consensus. ``diff`` is the diff between the
    consensus's first runner and the dissenter, useful for surfacing the
    actual divergence text.

    When no clear consensus exists (e.g. all three runners disagree, or
    runners split 2-2), ``consensus_runners`` is empty and one
    ``DiffOutlier`` is emitted per dissenting runner — each comparing
    against an arbitrary peer so callers can still surface a diff.
    """

    case: EvalCase
    dissenting_runner: str
    consensus_runners: list[str]
    diff: SessionDiff


@dataclass(frozen=True)
class DiffMatrix:
    """A case x runner grid of session records (or exceptions).

    ``records`` is keyed by ``(case_name, runner_name)``. A value of type
    ``Exception`` means that runner raised while processing that case;
    such (case, runner) pairs are excluded from ``unanimous()`` and treated
    as ``"errored"`` in ``report_html``.
    """

    cases: list[EvalCase]
    runner_names: list[str]
    records: dict[tuple[str, str], SessionRecord | Exception] = field(default_factory=dict)

    # ---- queries -----------------------------------------------------

    def unanimous(self) -> list[EvalCase]:
        """Return cases where every runner produced an equivalent record.

        A case is unanimous iff (a) no runner raised for that case and (b)
        every pair of runners satisfies ``compare_sessions(...).matches``.
        Cases with fewer than 2 runners are never unanimous (no pair to
        compare).
        """
        if len(self.runner_names) < 2:
            return []
        out: list[EvalCase] = []
        for case in self.cases:
            records = self._records_for(case)
            if records is None:
                continue
            if _all_pairs_match(list(records.values())):
                out.append(case)
        return out

    def outliers(self, threshold: float = 1.0) -> list[DiffOutlier]:
        """Return cases where at least one runner disagrees with the consensus.

        ``threshold`` is reserved for future similarity-based clustering;
        the current implementation treats compare_sessions as a strict
        equivalence relation (matches/doesn't), so any non-zero threshold
        yields the same set: every case that is not unanimous.

        Policy:
        * If a clear majority cluster exists (largest group strictly larger
          than every other group, including singletons), the dissenters are
          recorded with ``consensus_runners`` set to that cluster.
        * If no clear majority exists (all distinct, or tied largest groups),
          every runner is reported as a ``DiffOutlier`` with empty
          ``consensus_runners`` — the diff is taken against the first peer
          so callers still see a concrete divergence.
        * Cases where any runner raised are reported with the erroring
          runner(s) listed as dissenters, and the consensus drawn from the
          remaining successful runners (if they cluster).
        """
        del threshold  # reserved
        if len(self.runner_names) < 2:
            return []
        out: list[DiffOutlier] = []
        for case in self.cases:
            out.extend(self._outliers_for(case))
        return out

    def report_html(self, path: str | Path) -> None:
        """Render a static HTML matrix report to ``path``.

        Each row is a case; each column is a runner. Cells are colour-coded:
        green = consensus, red = dissenting, gray = errored. Cell text is the
        runner's last assistant message (truncated). All user-controlled
        strings are HTML-escaped before substitution.
        """
        Path(path).write_text(self._render_html(), encoding="utf-8")

    # ---- internals ---------------------------------------------------

    def _records_for(self, case: EvalCase) -> dict[str, SessionRecord] | None:
        """Return successful records for a case, or None if any runner errored."""
        out: dict[str, SessionRecord] = {}
        for runner_name in self.runner_names:
            value = self.records.get((case.name, runner_name))
            if not isinstance(value, SessionRecord):
                return None
            out[runner_name] = value
        return out

    def _outliers_for(self, case: EvalCase) -> list[DiffOutlier]:
        # Partition runners into "ok" (have a SessionRecord) and "errored".
        ok: dict[str, SessionRecord] = {}
        errored: list[str] = []
        for runner_name in self.runner_names:
            value = self.records.get((case.name, runner_name))
            if isinstance(value, SessionRecord):
                ok[runner_name] = value
            else:
                errored.append(runner_name)

        # Cluster the ok runners by record equivalence.
        clusters = _cluster_runners(ok)

        # Identify the consensus cluster (strictly largest), if any.
        consensus = _largest_cluster(clusters)
        consensus_names = list(consensus) if consensus is not None else []

        out: list[DiffOutlier] = []

        # Errored runners are always dissenters — diff against the first
        # consensus runner if available, otherwise the first ok runner,
        # otherwise an empty diff.
        if errored:
            peer = _first_peer(consensus_names, ok)
            for runner_name in errored:
                if peer is None:
                    diff = _empty_diff(case.name)
                else:
                    diff = _diff_for_error(case.name, ok[peer])
                out.append(
                    DiffOutlier(
                        case=case,
                        dissenting_runner=runner_name,
                        consensus_runners=consensus_names,
                        diff=diff,
                    )
                )

        # When a consensus exists, every ok runner outside it is a dissenter.
        if consensus is not None:
            consensus_record = ok[consensus[0]]
            for runner_name, record in ok.items():
                if runner_name in consensus:
                    continue
                diff = compare_sessions(consensus_record, record, name=case.name)
                out.append(
                    DiffOutlier(
                        case=case,
                        dissenting_runner=runner_name,
                        consensus_runners=consensus_names,
                        diff=diff,
                    )
                )
            return out

        # No consensus: emit a DiffOutlier per ok runner (against the first
        # peer that disagrees with it) only if there is real disagreement.
        if len(ok) >= 2 and not _all_pairs_match(list(ok.values())):
            ok_names = list(ok)
            for i, runner_name in enumerate(ok_names):
                # Pick the first other runner whose record differs.
                peer_name = _first_disagreeing_peer(ok_names, i, ok)
                if peer_name is None:
                    # This runner agrees with everyone — not a dissenter
                    # in the no-consensus branch.
                    continue
                diff = compare_sessions(ok[peer_name], ok[runner_name], name=case.name)
                out.append(
                    DiffOutlier(
                        case=case,
                        dissenting_runner=runner_name,
                        consensus_runners=consensus_names,
                        diff=diff,
                    )
                )
        return out

    def _render_html(self) -> str:
        template = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))

        runner_headers = "".join(f"<th>{html.escape(name)}</th>" for name in self.runner_names)

        rows: list[str] = []
        for case in self.cases:
            cells: list[str] = []
            classification = self._classify_case(case)
            for runner_name in self.runner_names:
                cls, summary = classification[runner_name]
                cells.append(f'<td class="{cls}"><pre>{html.escape(summary)}</pre></td>')
            rows.append(f'<tr><td class="case">{html.escape(case.name)}</td>{"".join(cells)}</tr>')

        unanimous_count = len(self.unanimous())
        outlier_cases = {o.case.name for o in self.outliers()}
        summary = (
            f"{len(self.cases)} cases x {len(self.runner_names)} runners; "
            f"{unanimous_count} unanimous, {len(outlier_cases)} with outliers."
        )

        return template.substitute(
            title=html.escape(", ".join(self.runner_names) if self.runner_names else "(empty)"),
            summary=html.escape(summary),
            runner_headers=runner_headers,
            rows="\n    ".join(rows) if rows else "",
        )

    def _classify_case(self, case: EvalCase) -> dict[str, tuple[str, str]]:
        """Return ``runner -> (css_class, summary_text)`` for a case row."""
        ok: dict[str, SessionRecord] = {}
        errored: dict[str, Exception] = {}
        for runner_name in self.runner_names:
            value = self.records.get((case.name, runner_name))
            if isinstance(value, SessionRecord):
                ok[runner_name] = value
            elif isinstance(value, Exception):
                errored[runner_name] = value

        clusters = _cluster_runners(ok)
        consensus = _largest_cluster(clusters)
        consensus_set = set(consensus) if consensus is not None else set()

        out: dict[str, tuple[str, str]] = {}
        for runner_name in self.runner_names:
            if runner_name in errored:
                exc = errored[runner_name]
                out[runner_name] = ("error", f"{type(exc).__name__}: {exc}")
                continue
            record = ok[runner_name]
            summary = _last_assistant_text(record)
            if not consensus:
                if len(ok) >= 2 and _all_pairs_match(list(ok.values())):
                    out[runner_name] = ("consensus", summary)
                else:
                    out[runner_name] = ("dissent", summary)
                continue
            cls = "consensus" if runner_name in consensus_set else "dissent"
            out[runner_name] = (cls, summary)
        return out


# ---------------------------------------------------------------------------
# entry point


async def diff_eval(
    cases: list[EvalCase],
    *,
    agent: SubAgent,
    runners: dict[str, Runner],
    dispatcher: Dispatcher | None = None,
    hooks: HookRunner | None = None,
    telemetry: Telemetry | None = None,
) -> DiffMatrix:
    """Run ``cases`` against each runner in parallel and return a ``DiffMatrix``.

    Concurrency: each runner walks its case list sequentially; the runners run
    concurrently via ``asyncio.gather(..., return_exceptions=True)``. Wall time
    is therefore approximately ``slowest_runner * len(cases)``, not
    ``len(cases) * len(runners) * slowest_runner``.

    Failure isolation: an exception raised mid-cases for runner R is captured
    per-case so the matrix stays rectangular. Other runners are unaffected.

    ``dispatcher`` and ``hooks`` are optional — defaults are constructed when
    omitted. Each runner is wrapped in its own ``Orchestrator``, sharing the
    dispatcher/hooks/telemetry passed in.
    """
    runner_names = list(runners)
    cases = list(cases)
    shared_dispatcher = dispatcher if dispatcher is not None else Dispatcher()
    shared_hooks = hooks if hooks is not None else HookRunner()

    async def per_runner(name: str, runner: Runner) -> list[SessionRecord | Exception]:
        del name  # currently unused; kept for future logging hooks
        orchestrator = Orchestrator(
            shared_dispatcher,
            shared_hooks,
            runner,
            telemetry=telemetry,
        )
        out: list[SessionRecord | Exception] = []
        for case in cases:
            try:
                record = await _run_case(orchestrator, agent, case)
            except Exception as exc:  # noqa: BLE001 - recorded into matrix
                out.append(exc)
            else:
                out.append(record)
        return out

    results = await asyncio.gather(
        *(per_runner(n, runners[n]) for n in runner_names),
        return_exceptions=True,
    )

    records: dict[tuple[str, str], SessionRecord | Exception] = {}
    for runner_name, runner_result in zip(runner_names, results, strict=True):
        if isinstance(runner_result, BaseException):
            # Unlikely (per_runner catches per-case), but defensive.
            for case in cases:
                records[(case.name, runner_name)] = _to_exception(runner_result)
            continue
        for case, record_or_exc in zip(cases, runner_result, strict=True):
            records[(case.name, runner_name)] = record_or_exc

    return DiffMatrix(cases=cases, runner_names=runner_names, records=records)


# ---------------------------------------------------------------------------
# helpers


async def _run_case(
    orchestrator: Orchestrator,
    agent: SubAgent,
    case: EvalCase,
) -> SessionRecord:
    store = InMemoryStore()
    session = Session(orchestrator, agent, store)
    for prompt in case.prompts:
        await session.send(prompt)
    record = await store.load(session.session_id)
    if record is None:
        raise RuntimeError(
            f"session {session.session_id!r} vanished after diff_eval call — "
            "store backing dropped state"
        )
    return record


def _to_exception(value: BaseException) -> Exception:
    if isinstance(value, Exception):
        return value
    return RuntimeError(f"{type(value).__name__}: {value}")


def _records_match(a: SessionRecord, b: SessionRecord) -> bool:
    return compare_sessions(a, b).matches


def _cluster_runners(ok: dict[str, SessionRecord]) -> list[list[str]]:
    """Group runner names by record equivalence under ``compare_sessions``."""
    clusters: list[list[str]] = []
    for runner_name, record in ok.items():
        placed = False
        for cluster in clusters:
            sample_record = ok[cluster[0]]
            if _records_match(sample_record, record):
                cluster.append(runner_name)
                placed = True
                break
        if not placed:
            clusters.append([runner_name])
    return clusters


def _largest_cluster(clusters: list[list[str]]) -> list[str] | None:
    """Return the strictly-largest cluster, or None when tied or empty.

    A consensus requires an unambiguous majority. With clusters [[A, B], [C]]
    the consensus is [A, B]. With [[A], [B], [C]] there is no consensus.
    With [[A, B], [C, D]] there is no consensus (tied largest).
    """
    if not clusters:
        return None
    sizes = sorted((len(c) for c in clusters), reverse=True)
    if len(sizes) == 1:
        return clusters[0] if sizes[0] >= 2 else None
    if sizes[0] == sizes[1]:
        return None
    for cluster in clusters:
        if len(cluster) == sizes[0]:
            return cluster
    return None


def _all_pairs_match(records: list[SessionRecord]) -> bool:
    if len(records) < 2:
        return True
    base = records[0]
    return all(_records_match(base, r) for r in records[1:])


def _first_peer(consensus: list[str], ok: dict[str, SessionRecord]) -> str | None:
    if consensus:
        return consensus[0]
    if ok:
        return next(iter(ok))
    return None


def _first_disagreeing_peer(names: list[str], idx: int, ok: dict[str, SessionRecord]) -> str | None:
    target = ok[names[idx]]
    for j, other in enumerate(names):
        if j == idx:
            continue
        if not _records_match(target, ok[other]):
            return other
    return None


def _last_assistant_text(record: SessionRecord) -> str:
    """Return the last assistant message's concatenated text (or a placeholder)."""
    for msg in reversed(record.messages):
        if msg.role != "assistant":
            continue
        chunks = [b.text for b in msg.content if b.type == "text" and b.text]
        if chunks:
            return _truncate("".join(chunks))
        return _truncate(_describe_blocks(msg))
    return "(no assistant turn)"


def _describe_blocks(msg: Message) -> str:
    parts: list[str] = []
    for block in msg.content:
        if block.type == "tool_use" and block.tool_use is not None:
            parts.append(f"<tool_use {block.tool_use.name}>")
        elif block.type == "tool_result" and block.tool_result is not None:
            parts.append("<tool_result>")
        else:
            parts.append(f"<{block.type}>")
    return "".join(parts) if parts else "(empty)"


def _truncate(s: str, limit: int = 240) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _empty_diff(name: str) -> SessionDiff:
    return SessionDiff(name=name, matches=False, turns=[])


def _diff_for_error(name: str, peer: SessionRecord) -> SessionDiff:
    """A SessionDiff against an empty record, used when the dissenter errored."""
    placeholder = SessionRecord(
        session_id="<errored>",
        agent=peer.agent,
        messages=[],
    )
    return compare_sessions(peer, placeholder, name=name)
