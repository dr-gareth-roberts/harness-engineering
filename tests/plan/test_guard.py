"""Tests for `PlanGuardedRunner` enforcement modes.

Covers tests 4, 5, 6, 7, and 8 from the design doc:

  4. Strict mode: executor calls expected tool -> passes; calls different
     tool -> `PlanViolation`.
  5. Strict mode: extra tool call beyond the plan -> `PlanViolation`.
  6. Superset mode: extra tool call -> allowed.
  7. Subset mode: missing a planned tool -> allowed; wrong-tool still fails.
  8. `PlanViolation.step_index`, `.expected`, and `.actual` are populated.

The fake runner used here is a minimal closure satisfying the `Runner`
protocol; we feed it pre-built `Message`s with tool_use blocks so the guard
can inspect them.
"""

from __future__ import annotations

import pytest

from harness.agents.definition import SubAgent
from harness.plan import Plan, PlanGuardedRunner, PlannedToolCall, PlanViolation
from harness.prompts.messages import ContentBlock, Message
from harness.tools.schema import ToolCall


def _agent() -> SubAgent:
    return SubAgent(name="exec", system_prompt="be helpful", model="test-model")


def _tool_use_message(*calls: ToolCall) -> Message:
    return Message(
        role="assistant",
        content=[ContentBlock(type="tool_use", tool_use=c) for c in calls],
    )


def _scripted_runner(replies: list[Message]):  # type: ignore[no-untyped-def]
    """A closure satisfying the Runner protocol; pops one reply per call."""
    iterator = iter(replies)

    async def run(agent: SubAgent, messages: list[Message]) -> Message:
        return next(iterator)

    return run


# --- Test 4 ---------------------------------------------------------------


async def test_strict_mode_passes_when_tool_matches() -> None:
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search", arguments_match={"q": "rust"}),
        ]
    )
    runner = _scripted_runner(
        [_tool_use_message(ToolCall(name="search", arguments={"q": "rust"}, id="c1"))]
    )
    guard = PlanGuardedRunner(runner, plan)

    result = await guard(_agent(), [])
    # Passthrough: same Message object content.
    assert result.role == "assistant"
    assert result.content[0].tool_use is not None
    assert result.content[0].tool_use.name == "search"
    assert guard.step_index == 1
    # Strict requires plan exhaustion at finalize; here it is.
    guard.finalize()


async def test_strict_mode_raises_on_wrong_tool() -> None:
    plan = Plan(
        steps=[PlannedToolCall(tool_name="search")],
    )
    runner = _scripted_runner(
        [_tool_use_message(ToolCall(name="delete", arguments={"id": 1}, id="c1"))]
    )
    guard = PlanGuardedRunner(runner, plan)

    with pytest.raises(PlanViolation) as excinfo:
        await guard(_agent(), [])
    err = excinfo.value
    assert err.step_index == 0
    assert err.expected is not None
    assert err.expected.tool_name == "search"
    assert err.actual is not None
    assert err.actual.name == "delete"


async def test_strict_mode_raises_on_argument_mismatch() -> None:
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search", arguments_match={"q": "rust"}),
        ]
    )
    runner = _scripted_runner(
        [_tool_use_message(ToolCall(name="search", arguments={"q": "go"}, id="c1"))]
    )
    guard = PlanGuardedRunner(runner, plan)

    with pytest.raises(PlanViolation) as excinfo:
        await guard(_agent(), [])
    assert excinfo.value.step_index == 0
    # The tool name matched, but the args didn't — so `expected` is the active
    # step and `actual` is the offending call (with the wrong args).
    assert excinfo.value.expected is not None
    assert excinfo.value.expected.tool_name == "search"
    assert excinfo.value.actual is not None
    assert excinfo.value.actual.arguments == {"q": "go"}


# --- Test 5 ---------------------------------------------------------------


async def test_strict_mode_raises_on_extra_call_beyond_plan() -> None:
    plan = Plan(steps=[PlannedToolCall(tool_name="search")])
    runner = _scripted_runner(
        [
            _tool_use_message(
                ToolCall(name="search", arguments={"q": "x"}, id="c1"),
                ToolCall(name="parse", arguments={}, id="c2"),
            )
        ]
    )
    guard = PlanGuardedRunner(runner, plan)

    with pytest.raises(PlanViolation) as excinfo:
        await guard(_agent(), [])
    err = excinfo.value
    # Step pointer landed past the end; extra-call signature.
    assert err.step_index == 1  # one past `len(plan.steps)`-1 = past end
    assert err.expected is None
    assert err.actual is not None
    assert err.actual.name == "parse"


# --- Test 6 ---------------------------------------------------------------


async def test_superset_mode_allows_extra_calls() -> None:
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="parse"),
        ],
        mode="superset",
    )
    runner = _scripted_runner(
        [
            _tool_use_message(
                ToolCall(name="search", arguments={"q": "x"}, id="c1"),
                # Extra call between steps; allowed in superset.
                ToolCall(name="lookup", arguments={"id": 1}, id="c2"),
                ToolCall(name="parse", arguments={}, id="c3"),
                # Extra call past the plan; allowed in superset.
                ToolCall(name="summarize", arguments={}, id="c4"),
            )
        ]
    )
    guard = PlanGuardedRunner(runner, plan)
    await guard(_agent(), [])
    # Plan was fully consumed.
    assert guard.step_index == 2
    guard.finalize()


async def test_superset_mode_still_requires_plan_exhaustion() -> None:
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="parse"),
        ],
        mode="superset",
    )
    runner = _scripted_runner([_tool_use_message(ToolCall(name="search", arguments={}, id="c1"))])
    guard = PlanGuardedRunner(runner, plan)
    await guard(_agent(), [])
    assert guard.step_index == 1
    # The second step never fired -> finalize must violate.
    with pytest.raises(PlanViolation) as excinfo:
        guard.finalize()
    assert excinfo.value.step_index == 1
    assert excinfo.value.expected is not None
    assert excinfo.value.expected.tool_name == "parse"
    assert excinfo.value.actual is None


# --- Test 7 ---------------------------------------------------------------


async def test_subset_mode_allows_missing_planned_steps() -> None:
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="parse"),
            PlannedToolCall(tool_name="summarize"),
        ],
        mode="subset",
    )
    # Executor skips `parse` — that's fine in subset mode.
    runner = _scripted_runner(
        [
            _tool_use_message(
                ToolCall(name="search", arguments={"q": "x"}, id="c1"),
                ToolCall(name="summarize", arguments={}, id="c2"),
            )
        ]
    )
    guard = PlanGuardedRunner(runner, plan)
    await guard(_agent(), [])
    # We advanced past `summarize` (index 2), so step_index = 3.
    assert guard.step_index == 3
    # Subset is happy with leftover plan steps; finalize is a no-op.
    guard.finalize()


async def test_subset_mode_still_fails_on_wrong_tool() -> None:
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="parse"),
        ],
        mode="subset",
    )
    runner = _scripted_runner([_tool_use_message(ToolCall(name="delete", arguments={}, id="c1"))])
    guard = PlanGuardedRunner(runner, plan)
    with pytest.raises(PlanViolation) as excinfo:
        await guard(_agent(), [])
    # Wrong tool from the start -> still violation.
    assert excinfo.value.actual is not None
    assert excinfo.value.actual.name == "delete"


async def test_subset_mode_finalize_does_not_complain_about_unmet_plan() -> None:
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="parse"),
        ],
        mode="subset",
    )
    runner = _scripted_runner([_tool_use_message(ToolCall(name="search", arguments={}, id="c1"))])
    guard = PlanGuardedRunner(runner, plan)
    await guard(_agent(), [])
    # Plan has parse left untouched; subset doesn't care.
    guard.finalize()


# --- Test 8 ---------------------------------------------------------------


async def test_plan_violation_carries_structured_fields() -> None:
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="parse"),
        ]
    )
    runner = _scripted_runner(
        [
            _tool_use_message(
                ToolCall(name="search", arguments={}, id="c1"),
                ToolCall(name="delete", arguments={"id": 99}, id="c2"),
            )
        ]
    )
    guard = PlanGuardedRunner(runner, plan)
    with pytest.raises(PlanViolation) as excinfo:
        await guard(_agent(), [])
    err = excinfo.value
    assert err.step_index == 1
    assert err.expected is not None
    assert err.expected.tool_name == "parse"
    assert err.actual is not None
    assert err.actual.name == "delete"
    assert err.actual.arguments == {"id": 99}
    # The exception's str() should be informative.
    assert "parse" in str(err)
    assert "delete" in str(err)


async def test_state_persists_across_multiple_runner_calls() -> None:
    """Plan can span multiple turns: the step pointer is per-instance."""
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="summarize"),
        ]
    )
    runner = _scripted_runner(
        [
            _tool_use_message(ToolCall(name="search", arguments={}, id="c1")),
            _tool_use_message(ToolCall(name="summarize", arguments={}, id="c2")),
        ]
    )
    guard = PlanGuardedRunner(runner, plan)
    await guard(_agent(), [])
    assert guard.step_index == 1
    await guard(_agent(), [])
    assert guard.step_index == 2
    guard.finalize()


async def test_text_only_messages_are_passthrough() -> None:
    """A turn that emits only text (no tool_use blocks) doesn't advance the plan."""
    plan = Plan(steps=[PlannedToolCall(tool_name="search")])
    text_message = Message(
        role="assistant",
        content=[ContentBlock(type="text", text="thinking...")],
    )
    runner = _scripted_runner(
        [
            text_message,
            _tool_use_message(ToolCall(name="search", arguments={}, id="c1")),
        ]
    )
    guard = PlanGuardedRunner(runner, plan)
    await guard(_agent(), [])
    assert guard.step_index == 0
    await guard(_agent(), [])
    assert guard.step_index == 1
    guard.finalize()


async def test_finalize_is_idempotent() -> None:
    plan = Plan(steps=[PlannedToolCall(tool_name="search")])
    runner = _scripted_runner([_tool_use_message(ToolCall(name="search", arguments={}, id="c1"))])
    guard = PlanGuardedRunner(runner, plan)
    await guard(_agent(), [])
    guard.finalize()
    # Second call must not raise.
    guard.finalize()


# --- M3.4 — subset matching is O(N), not O(N²) ----------------------------


async def test_subset_mode_advances_past_unmatched_steps() -> None:
    """Subset matching with a call that lines up with a step several positions
    ahead must skip past the intermediate (unmatched) steps and land the
    cursor on `match_index + 1`. Pins the cursor-advance contract that the
    O(N) tool-name index relies on.
    """
    # Build a plan where `target` sits five positions past `_step_index`.
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="a"),
            PlannedToolCall(tool_name="b"),
            PlannedToolCall(tool_name="c"),
            PlannedToolCall(tool_name="d"),
            PlannedToolCall(tool_name="e"),
            PlannedToolCall(tool_name="target"),
            PlannedToolCall(tool_name="trailing"),
        ],
        mode="subset",
    )
    runner = _scripted_runner([_tool_use_message(ToolCall(name="target", arguments={}, id="c1"))])
    guard = PlanGuardedRunner(runner, plan)
    await guard(_agent(), [])
    # Match index is 5 (zero-based), so cursor lands on 5+1=6.
    assert guard.step_index == 6
    guard.finalize()


async def test_subset_matching_is_linear_in_plan_size() -> None:
    """Behaviour pin for M3.4: subset matching must tick each step at most
    a bounded number of times across the whole session, not once per call.

    The pre-fix implementation ticked every remaining step on every observed
    call (≈ N² total ticks for an N-step plan run end-to-end). With the
    tool-name index plus precompiled DFAs, the total tick count is the
    number of matching candidates considered — at most N for unique tool
    names. We assert the wall-clock proxy: total ticks ≤ 3 N for a
    100-step plan with unique tool names.
    """
    n = 100
    plan = Plan(
        steps=[PlannedToolCall(tool_name=f"tool_{i}") for i in range(n)],
        mode="subset",
    )
    # Drive the guard with every step in order — one tool_use per turn so
    # the inner loop is exercised at every step boundary.
    replies = [
        _tool_use_message(ToolCall(name=f"tool_{i}", arguments={}, id=f"c{i}")) for i in range(n)
    ]
    runner = _scripted_runner(replies)
    guard = PlanGuardedRunner(runner, plan)

    # Count DFA ticks via monkey-patch on the precompiled DFA instances.
    tick_count = 0
    for dfa in guard._step_dfas:
        original_tick = dfa.tick

        def counted_tick(message, _orig=original_tick):  # type: ignore[no-untyped-def]
            nonlocal tick_count
            tick_count += 1
            return _orig(message)

        dfa.tick = counted_tick  # type: ignore[method-assign]

    for _ in range(n):
        await guard(_agent(), [])

    # Linear-or-better: with unique tool names every call hits exactly one
    # candidate, so total ticks should equal n. Leave headroom for any
    # future short-circuit edge to land below 3n; well clear of n² = 10000.
    assert tick_count <= 3 * n, (
        f"expected linear tick count (~{n}), got {tick_count} for n={n}; "
        "subset matching may have regressed to quadratic"
    )
    assert guard.step_index == n


async def test_precompiled_dfa_reset_isolates_consecutive_ticks() -> None:
    """Regression pin: caching DFAs across ticks requires `reset()` between
    ticks. Without it, an `Always(...)` DFA that fired a violation on one
    step would latch `_violated=True` and silently accept the next call.

    Scenario: subset plan with two `search` steps. First call has wrong
    args (DFA misses); second call has correct args. The misses must not
    poison the DFA for the second call.
    """
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search", arguments_match={"q": "rust"}),
            PlannedToolCall(tool_name="search", arguments_match={"q": "go"}),
        ],
        mode="subset",
    )
    runner = _scripted_runner(
        [
            _tool_use_message(ToolCall(name="search", arguments={"q": "go"}, id="c1")),
        ]
    )
    guard = PlanGuardedRunner(runner, plan)
    await guard(_agent(), [])
    # The first step's DFA missed (q="go" ≠ "rust"); the second step's DFA
    # matched. Cursor lands on 1+1=2.
    assert guard.step_index == 2
