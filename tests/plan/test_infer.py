"""Tests for `infer_plan_from_records`.

Covers the modal-sequence selection algorithm with first-occurrence
(earliest) tiebreak, the default success heuristic (assistant-terminated,
no error tool_results, no orphan tool_uses), the custom-predicate path,
and the empty / fully-filtered edge cases.

We construct `SessionRecord`s directly rather than going through a real
runner — the inference helper only cares about the message structure.
"""

from __future__ import annotations

from harness.agents import SubAgent
from harness.memory import SessionRecord
from harness.plan import Plan, PlannedToolCall
from harness.plan.infer import infer_plan_from_records
from harness.prompts import assistant_tool_use, text, user_tool_result
from harness.prompts.messages import ContentBlock, Message
from harness.tools import ToolCall, ToolResult


def _agent() -> SubAgent:
    return SubAgent(name="t", system_prompt="", model="test-model")


def _make_record(
    *,
    session_id: str,
    tool_names: list[str],
    finalize_with_assistant_text: bool = True,
) -> SessionRecord:
    """Build a successful trajectory: user prompt, then for each tool name
    an assistant tool_use + matching user tool_result, optionally closed
    with a trailing assistant text message so the heuristic accepts it.

    Tool-call ids are deterministic (`{session_id}-N`) so tool_use and
    tool_result blocks pair cleanly.
    """
    messages: list[Message] = [text("user", "do the work")]
    for i, name in enumerate(tool_names):
        call_id = f"{session_id}-{i}"
        messages.append(assistant_tool_use(ToolCall(name=name, arguments={}, id=call_id)))
        messages.append(user_tool_result(ToolResult(id=call_id, content="ok", is_error=False)))
    if finalize_with_assistant_text:
        messages.append(text("assistant", "done"))
    return SessionRecord(session_id=session_id, agent=_agent(), messages=messages)


# --- Test 1: empty input --------------------------------------------------


def test_empty_records_returns_empty_superset_plan() -> None:
    plan = infer_plan_from_records([])
    assert plan == Plan(steps=[], mode="superset")


# --- Test 2: all records fail success heuristic ---------------------------


def test_all_records_failing_heuristic_returns_empty_plan() -> None:
    # Two records that the default heuristic will reject: each ends with
    # a *user* message rather than assistant.
    bad = SessionRecord(
        session_id="bad",
        agent=_agent(),
        messages=[
            text("user", "hi"),
            assistant_tool_use(ToolCall(name="search", arguments={}, id="x1")),
            user_tool_result(ToolResult(id="x1", content="ok")),
            text("user", "user gets last word"),
        ],
    )
    plan = infer_plan_from_records([bad, bad])
    assert plan == Plan(steps=[], mode="superset")


# --- Test 3: single record, single sequence -------------------------------


def test_single_record_produces_mirroring_plan() -> None:
    record = _make_record(session_id="r1", tool_names=["search", "summarize"])
    plan = infer_plan_from_records([record])
    assert plan == Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="summarize"),
        ],
        mode="superset",
    )


# --- Test 4: modal sequence wins ------------------------------------------


def test_modal_sequence_wins_over_minority() -> None:
    # 3 records run A then B, 2 records run A then C. A->B is modal.
    records = [
        _make_record(session_id="ab1", tool_names=["A", "B"]),
        _make_record(session_id="ac1", tool_names=["A", "C"]),
        _make_record(session_id="ab2", tool_names=["A", "B"]),
        _make_record(session_id="ac2", tool_names=["A", "C"]),
        _make_record(session_id="ab3", tool_names=["A", "B"]),
    ]
    plan = infer_plan_from_records(records)
    assert [step.tool_name for step in plan.steps] == ["A", "B"]


# --- Test 5: tiebreak determinism (earliest-first-occurrence wins) ---------


def test_tiebreak_picks_sequence_whose_first_occurrence_is_earliest() -> None:
    # Two A->B at indices 0, 1; two A->C at indices 2, 3. Both tied at
    # count = 2. A->B's first occurrence (index 0) precedes A->C's (index 2),
    # so A->B must win.
    records = [
        _make_record(session_id="ab1", tool_names=["A", "B"]),
        _make_record(session_id="ab2", tool_names=["A", "B"]),
        _make_record(session_id="ac1", tool_names=["A", "C"]),
        _make_record(session_id="ac2", tool_names=["A", "C"]),
    ]
    plan = infer_plan_from_records(records)
    assert [step.tool_name for step in plan.steps] == ["A", "B"]


def test_tiebreak_is_stable_when_input_order_reversed() -> None:
    # Reverse the previous test's input. Now A->C's first occurrence is
    # at index 0, so it must win.
    records = [
        _make_record(session_id="ac1", tool_names=["A", "C"]),
        _make_record(session_id="ac2", tool_names=["A", "C"]),
        _make_record(session_id="ab1", tool_names=["A", "B"]),
        _make_record(session_id="ab2", tool_names=["A", "B"]),
    ]
    plan = infer_plan_from_records(records)
    assert [step.tool_name for step in plan.steps] == ["A", "C"]


# --- Test 6: custom success predicate -------------------------------------


def test_custom_success_predicate_filters_records() -> None:
    # Three "interesting" records all run A->B, two "boring" records run
    # the more common X->Y. Without the predicate, X->Y wins (it would
    # be modal); with the predicate marking only A->B records as
    # successful, A->B wins.
    interesting = [
        _make_record(session_id="i1", tool_names=["A", "B"]),
        _make_record(session_id="i2", tool_names=["A", "B"]),
        _make_record(session_id="i3", tool_names=["A", "B"]),
    ]
    boring = [
        _make_record(session_id="b1", tool_names=["X", "Y"]),
        _make_record(session_id="b2", tool_names=["X", "Y"]),
        _make_record(session_id="b3", tool_names=["X", "Y"]),
        _make_record(session_id="b4", tool_names=["X", "Y"]),
    ]

    # Sanity check: without the predicate, X->Y wins because it's modal.
    default_plan = infer_plan_from_records(interesting + boring)
    assert [step.tool_name for step in default_plan.steps] == ["X", "Y"]

    # With a predicate that accepts only the interesting subset, A->B wins.
    only_interesting = {r.session_id for r in interesting}

    def _is_interesting(record: SessionRecord) -> bool:
        return record.session_id in only_interesting

    filtered_plan = infer_plan_from_records(
        interesting + boring,
        success=_is_interesting,
    )
    assert [step.tool_name for step in filtered_plan.steps] == ["A", "B"]


# --- Test 7: default heuristic rejects orphan tool_use --------------------


def test_default_heuristic_rejects_orphan_tool_use() -> None:
    # Record contains an assistant tool_use whose id has no matching
    # tool_result. Heuristic must reject it.
    orphan = SessionRecord(
        session_id="orphan",
        agent=_agent(),
        messages=[
            text("user", "hi"),
            assistant_tool_use(ToolCall(name="search", arguments={}, id="missing")),
            text("assistant", "done"),
        ],
    )
    paired = _make_record(session_id="paired", tool_names=["X", "Y"])

    # If the orphan were accepted, "search" would compete; with the
    # heuristic rejecting it, the plan must reflect only the paired record.
    plan = infer_plan_from_records([orphan, paired])
    assert [step.tool_name for step in plan.steps] == ["X", "Y"]


# --- Test 8: default heuristic rejects error tool_results -----------------


def test_default_heuristic_rejects_error_tool_results() -> None:
    errored = SessionRecord(
        session_id="err",
        agent=_agent(),
        messages=[
            text("user", "hi"),
            assistant_tool_use(ToolCall(name="search", arguments={}, id="c1")),
            user_tool_result(ToolResult(id="c1", content="boom", is_error=True)),
            text("assistant", "I failed"),
        ],
    )
    good = _make_record(session_id="good", tool_names=["X", "Y"])
    plan = infer_plan_from_records([errored, good])
    assert [step.tool_name for step in plan.steps] == ["X", "Y"]


# --- Test 9: default heuristic rejects non-assistant-terminated -----------


def test_default_heuristic_rejects_non_assistant_terminated() -> None:
    # Last message is a user message: heuristic should reject.
    bad = SessionRecord(
        session_id="bad",
        agent=_agent(),
        messages=[
            text("user", "hi"),
            assistant_tool_use(ToolCall(name="search", arguments={}, id="c1")),
            user_tool_result(ToolResult(id="c1", content="ok")),
            # Trailing user message means the assistant didn't get the
            # last word — the run was probably cut short.
            text("user", "follow up"),
        ],
    )
    good = _make_record(session_id="good", tool_names=["A", "B"])
    plan = infer_plan_from_records([bad, good])
    assert [step.tool_name for step in plan.steps] == ["A", "B"]


# --- Test 10: mode override ----------------------------------------------


def test_mode_override_propagates_to_plan() -> None:
    record = _make_record(session_id="r1", tool_names=["search"])
    plan = infer_plan_from_records([record], mode="strict")
    assert plan.mode == "strict"
    # Empty branch must also honour the override.
    empty_strict = infer_plan_from_records([], mode="strict")
    assert empty_strict.mode == "strict"


# --- Bonus coverage: multiple tool_uses per assistant message -------------


def test_multiple_tool_uses_per_message_are_collected_in_order() -> None:
    """An assistant message can carry several tool_use blocks; the
    inference helper must walk every block of every assistant message
    in order, not just the first."""
    call_a = ToolCall(name="A", arguments={}, id="m1-a")
    call_b = ToolCall(name="B", arguments={}, id="m1-b")
    multi = Message(
        role="assistant",
        content=[
            ContentBlock(type="tool_use", tool_use=call_a),
            ContentBlock(type="tool_use", tool_use=call_b),
        ],
    )
    record = SessionRecord(
        session_id="multi",
        agent=_agent(),
        messages=[
            text("user", "go"),
            multi,
            user_tool_result(ToolResult(id="m1-a", content="ok")),
            user_tool_result(ToolResult(id="m1-b", content="ok")),
            text("assistant", "done"),
        ],
    )
    plan = infer_plan_from_records([record])
    assert [step.tool_name for step in plan.steps] == ["A", "B"]


# --- Bonus coverage: surviving record with no tool_uses ------------------


def test_surviving_records_with_no_tool_uses_yield_empty_plan() -> None:
    """A record can pass the success heuristic (assistant-terminated, no
    errors, no orphans because no tool_uses) but contribute no sequence.
    Inference should treat the population as having no usable sequences
    and return an empty plan, not crash."""
    chatty = SessionRecord(
        session_id="chatty",
        agent=_agent(),
        messages=[text("user", "hi"), text("assistant", "hello there")],
    )
    plan = infer_plan_from_records([chatty])
    assert plan == Plan(steps=[], mode="superset")


# --- Bonus coverage: empty messages list rejected by heuristic ------------


def test_default_heuristic_rejects_empty_messages() -> None:
    empty = SessionRecord(session_id="e", agent=_agent(), messages=[])
    good = _make_record(session_id="good", tool_names=["X"])
    plan = infer_plan_from_records([empty, good])
    assert [step.tool_name for step in plan.steps] == ["X"]
