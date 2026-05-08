"""`infer_plan_from_records` — mine a `Plan` from successful `SessionRecord`s.

The companion to `harness.plan.derive_plan`: where `derive_plan` asks a *live*
planner to emit a plan, this helper *infers* a plan from recorded successful
trajectories — no model in the loop, just the runs you've already paid for.

# Algorithm

1. Filter inputs through a success predicate. The default heuristic is
   defined locally (see `_default_success`); callers may pass any predicate
   matching `Callable[[SessionRecord], bool]` to fit their notion of
   "successful".
2. From each surviving record, extract the assistant tool_use sequence —
   the names of every tool_use block in every assistant message, in order.
   (A single assistant message may contain multiple tool_use blocks; they
   all contribute to the per-record tuple.)
3. Group identical name-tuples and count. Tie-break: among sequences tied
   for most common, pick the one whose first occurrence (in the filtered
   record order) appeared *earliest*.
4. Build a `Plan` whose steps mirror that modal sequence — one
   `PlannedToolCall(tool_name=name)` per name. `arguments_match` is left
   `None` so the caller can post-process if they want stricter matching.

# Why these choices?

We considered several alternatives and rejected each:

* **Longest common prefix (LCP) of all sequences** — collapses to the empty
  prefix the moment any two records disagree on the first call. Brittle.
* **Bigram-derived expected sequence** — useful for prediction (the
  `harness.speculate` module uses it), but doesn't yield a single linear
  plan. Composing per-name transition probabilities into a deterministic
  step list is non-obvious and prone to producing nonsense traversals.
* **Intersection of all sequences** — strips the plan to only steps every
  record performs in identical positions. Loses too much signal: a record
  that fetches and a record that searches both look "non-empty" even when
  one tool was used. Set semantics also drop ordering.

Modal-sequence matches the prevailing-strategy intuition: "what did most
of my known-good runs actually do?". When tied, "which strategy did I
develop first?" is a stable, deterministic preference — and the test
fixtures pin it that way.

# Why default `mode="superset"`?

The default `Plan.mode` is `"strict"` (no extra calls allowed; plan must
fully match), which is the right default for a *live-planner-emitted*
plan: a planner that says "do X, then Y" is making a commitment.

An *inferred* plan is qualitatively different. We mined a *minimum*
expected set from the past; the model may legitimately produce extras
(retries, side checks, follow-ups) without that constituting deviation.
Strict mode would punish every such variation. `superset` reads the plan
as "every step here must happen, in order; extras are fine".

Callers who want stricter enforcement can pass `mode="strict"` explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from harness.plan.plan import Plan, PlanMode, PlannedToolCall

if TYPE_CHECKING:
    from harness.memory.record import SessionRecord


SuccessPredicate = Callable[["SessionRecord"], bool]


def _default_success(record: SessionRecord) -> bool:
    """Default success heuristic.

    Treats a record as successful when **all** of:

    * The trajectory has at least one message.
    * The final message's role is `"assistant"` (i.e. the assistant
      "owned" the conclusion — a trailing `user` message usually means
      the run was cut off mid-loop).
    * No tool_result block carries `is_error=True`.
    * Every assistant tool_use has a matching user tool_result by id —
      no orphan tool_uses (a tool_use without a corresponding result
      typically means the executor hung or crashed).

    This is a conservative heuristic: it filters out obvious failures
    while letting the caller substitute a domain-specific predicate
    (e.g. "score >= 0.8" pulled from `record.metadata`) for finer
    control.
    """
    if not record.messages:
        return False
    last = record.messages[-1]
    if last.role != "assistant":
        return False
    # No error tool_results.
    for msg in record.messages:
        for block in msg.content:
            if (
                block.type == "tool_result"
                and block.tool_result is not None
                and block.tool_result.is_error
            ):
                return False
    # No orphan tool_uses (every assistant tool_use has a matching user tool_result).
    tool_use_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    for msg in record.messages:
        for block in msg.content:
            if block.type == "tool_use" and block.tool_use is not None and block.tool_use.id:
                tool_use_ids.add(block.tool_use.id)
            if (
                block.type == "tool_result"
                and block.tool_result is not None
                and block.tool_result.id
            ):
                tool_result_ids.add(block.tool_result.id)
    return tool_use_ids <= tool_result_ids


def _tool_use_sequence(record: SessionRecord) -> tuple[str, ...]:
    """Extract the ordered tool_use name sequence from a record.

    Walks every assistant message in order, then every block in each
    message in order, collecting the `tool_use.name` of every tool_use
    block. A single message may contribute multiple names.
    """
    names: list[str] = []
    for msg in record.messages:
        if msg.role != "assistant":
            continue
        for block in msg.content:
            if block.type == "tool_use" and block.tool_use is not None:
                names.append(block.tool_use.name)
    return tuple(names)


def infer_plan_from_records(
    records: list[SessionRecord],
    *,
    success: SuccessPredicate | None = None,
    mode: PlanMode = "superset",
) -> Plan:
    """Mine a `Plan` from successful `SessionRecord`s.

    Args:
        records: The candidate trajectories to mine. Order matters only
            for the tiebreak — see below.
        success: A predicate filtering records to "successful" ones. If
            `None`, the built-in `_default_success` heuristic is used:
            assistant-terminated, no error tool_results, no orphan
            tool_uses.
        mode: The deviation mode for the returned plan. Defaults to
            `"superset"` — extras are allowed beyond the inferred
            minimum sequence. Pass `"strict"` for exact-match
            enforcement, or `"subset"` for "any subsequence of these
            calls is fine".

    Returns:
        A `Plan` whose steps are the modal tool_use name sequence among
        successful records, each as `PlannedToolCall(tool_name=name)`
        with no argument constraints. Returns `Plan(steps=[], mode=mode)`
        when no records survive filtering, or when no surviving record
        contains any tool_use.

    Tiebreak: when multiple sequences tie for most common, the one whose
    first occurrence appeared *earliest* in the filtered record list
    wins. This makes the result deterministic in caller-controlled
    input order.

    Argument matching: each step's `arguments_match` and
    `arguments_regex` are left `None`. The caller may post-process the
    returned plan if they want to tighten matching (e.g. by computing
    the modal arguments per step from the same modal-sequence records).
    Doing it inside this helper would over-fit the API to one tightening
    strategy; leaving it open keeps the inference primitive simple.
    """
    predicate: SuccessPredicate = _default_success if success is None else success

    # Walk in input order so first-occurrence indices reflect caller input.
    counts: dict[tuple[str, ...], int] = {}
    first_seen: dict[tuple[str, ...], int] = {}
    survived_index = 0
    for record in records:
        if not predicate(record):
            continue
        sequence = _tool_use_sequence(record)
        if not sequence:
            # An empty tool_use sequence is meaningless as a plan step
            # source; skip it for selection purposes.
            survived_index += 1
            continue
        counts[sequence] = counts.get(sequence, 0) + 1
        if sequence not in first_seen:
            first_seen[sequence] = survived_index
        survived_index += 1

    if not counts:
        return Plan(steps=[], mode=mode)

    # Pick max count, tie-break on *earliest* first-occurrence index
    # (smallest first_seen wins). The negation in the key flips the
    # max-comparison so smaller indices rank higher.
    modal_sequence = max(counts.items(), key=lambda item: (item[1], -first_seen[item[0]]))[0]

    steps = [PlannedToolCall(tool_name=name) for name in modal_sequence]
    return Plan(steps=steps, mode=mode)
