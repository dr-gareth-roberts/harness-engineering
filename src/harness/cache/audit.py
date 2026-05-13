"""`audit` — walk a `FingerprintStore` and surface drift.

A `DriftEvent` is "the hash for breakpoint N changed between two
adjacent records in the audit window." Adjacent means sorted by
timestamp — the store doesn't promise order, so `audit` collects then
sorts.

`audit` returns at most one `DriftEvent` per *transition*: if a hash
flips A -> B -> A inside the window we surface two events. This matches
test 6 ("DriftEvent with diff when a hash changes") and test 5 ("no
drift events when all hashes are equal"), and it's the minimum that
lets a noisy timestamp leak still be visible if the operator runs
audit just after the latest call.

`stable_prefixes` is the complement: every breakpoint index for which
the audit window contains records and they all share a single hash.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from harness.cache.store import FingerprintRecord, FingerprintStore

_DEFAULT_DIFF_LINES = 20

# Heuristic hint patterns. Order matters — most specific wins. The first
# matcher whose regex hits the unified-diff body annotates the event.
_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
        "looks like a timestamp leak — move out of the cached prefix or freeze it",
    ),
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "looks like a uuid leak — pull the identifier out of the cached prefix",
    ),
    (
        re.compile(r"\b\d{10,}\b"),
        "looks like a unix timestamp or counter leak — exclude from the cached prefix",
    ),
)


@dataclass(frozen=True)
class DriftEvent:
    """One observed prefix-hash transition.

    `before_*` and `after_*` describe the bracketing records; `diff`
    is the unified diff of their full prompts (truncated to the first
    20 changed lines). `hint` is a one-line guess at the cause when a
    well-known pattern (timestamp, uuid, …) appears in the diff.
    """

    breakpoint_index: int
    before_ts: datetime
    after_ts: datetime
    before_hash: str
    after_hash: str
    diff: str
    hint: str | None = None


@dataclass(frozen=True)
class DriftReport:
    """Outcome of `audit`. `stable_prefixes` and `drift_events` are
    independent: a single audit can show some breakpoints stable and
    others drifting."""

    stable_prefixes: list[int] = field(default_factory=list)
    drift_events: list[DriftEvent] = field(default_factory=list)
    audited_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _make_diff(before: str | None, after: str | None, max_lines: int = _DEFAULT_DIFF_LINES) -> str:
    """Render a unified diff of two JSON-serialized prompts.

    Returns a placeholder string when either side is `None` — the watcher
    can't always capture both sides (`full_capture="never"` strips them
    entirely). The placeholder is still a valid diff body so the CLI
    doesn't have to special-case it.
    """
    if before is None and after is None:
        return "<no full prompts captured; run with full_capture='on_drift' or 'always'>\n"
    before_text = before if before is not None else "<unavailable>"
    after_text = after if after is not None else "<unavailable>"

    before_lines = before_text.splitlines(keepends=True)
    after_lines = after_text.splitlines(keepends=True)
    if before_lines and not before_lines[-1].endswith("\n"):
        before_lines[-1] += "\n"
    if after_lines and not after_lines[-1].endswith("\n"):
        after_lines[-1] += "\n"

    diff_iter = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile="before",
        tofile="after",
        lineterm="",
    )
    diff_lines = list(diff_iter)
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines]
        diff_lines.append("... (diff truncated)")
    return "\n".join(diff_lines) + "\n"


def _hint_for(diff_body: str) -> str | None:
    for pattern, message in _HINTS:
        if pattern.search(diff_body):
            return message
    return None


async def audit(store: FingerprintStore, window_hours: float = 24.0) -> DriftReport:
    """Walk records ordered by timestamp; surface drift transitions.

    `window_hours` defines the lookback. Records older than `now -
    window_hours` are ignored — keeps a long-lived store from forcing
    an O(history) scan on every audit.
    """
    cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
    by_breakpoint: dict[int, list[FingerprintRecord]] = defaultdict(list)
    async for record in store.iter_recent(since=cutoff):
        by_breakpoint[record.breakpoint_index].append(record)

    stable: list[int] = []
    events: list[DriftEvent] = []

    for breakpoint_index, records in sorted(by_breakpoint.items()):
        records.sort(key=lambda r: r.timestamp)
        if not records:
            continue

        # `stable_prefixes` = every audited record carries the same hash.
        unique_hashes = {r.hash for r in records}
        if len(unique_hashes) == 1:
            stable.append(breakpoint_index)
            continue

        # Walk pairwise; record one DriftEvent per transition. Backfill
        # the "before" full_prompt from the most recent prior record
        # that carried one (mirrors the watcher's `on_drift` policy:
        # the previous prompt may be on an earlier record). Track the
        # timestamp of that source record so `before_ts` matches the
        # body we actually report — otherwise `before_ts` would point
        # at an intermediate identical record while the diff body came
        # from an earlier one.
        prev_full_prompt: str | None = None
        prev_full_prompt_ts: datetime | None = None
        for prior, current in zip(records, records[1:], strict=False):
            if prior.full_prompt is not None:
                prev_full_prompt = prior.full_prompt
                prev_full_prompt_ts = prior.timestamp
            if prior.hash == current.hash:
                continue
            diff = _make_diff(prev_full_prompt, current.full_prompt)
            before_ts = prev_full_prompt_ts if prev_full_prompt_ts is not None else prior.timestamp
            events.append(
                DriftEvent(
                    breakpoint_index=breakpoint_index,
                    before_ts=before_ts,
                    after_ts=current.timestamp,
                    before_hash=prior.hash,
                    after_hash=current.hash,
                    diff=diff,
                    hint=_hint_for(diff),
                )
            )

    return DriftReport(stable_prefixes=stable, drift_events=events)
