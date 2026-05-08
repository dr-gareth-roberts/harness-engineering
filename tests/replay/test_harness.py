from __future__ import annotations

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.memory import SessionRecord
from harness.prompts import ContentBlock, Message, text
from harness.replay import EvalCase, compare_sessions, run_eval
from harness.tools import Dispatcher, ToolCall, ToolResult


def make_orchestrator(reply: str) -> Orchestrator:
    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", reply)

    return Orchestrator(Dispatcher(), HookRunner(), fake_runner)


def make_record(messages: list[Message], session_id: str = "s") -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        agent=SubAgent(name="x", system_prompt=""),
        messages=messages,
    )


# ---------------------------------------------------------------------------
# run_eval


async def test_run_eval_executes_each_case() -> None:
    cases = [
        EvalCase(name="greet", prompts=["hi", "again"]),
        EvalCase(name="probe", prompts=["status?"]),
    ]
    orch = make_orchestrator("ok")
    agent = SubAgent(name="t", system_prompt="")

    results = await run_eval(cases, orchestrator=orch, agent=agent)

    assert [r.case.name for r in results] == ["greet", "probe"]
    assert results[0].duration_ms >= 0

    greet = results[0].record.messages
    assert [m.role for m in greet] == ["user", "assistant", "user", "assistant"]
    assert [m.content[0].text for m in greet] == ["hi", "ok", "again", "ok"]


async def test_run_eval_isolates_cases() -> None:
    cases = [
        EvalCase(name="a", prompts=["hi"]),
        EvalCase(name="b", prompts=["bye"]),
    ]
    orch = make_orchestrator("ok")
    agent = SubAgent(name="t", system_prompt="")

    results = await run_eval(cases, orchestrator=orch, agent=agent)
    a_msgs = [m.content[0].text for m in results[0].record.messages]
    b_msgs = [m.content[0].text for m in results[1].record.messages]
    assert a_msgs == ["hi", "ok"]
    assert b_msgs == ["bye", "ok"]


# ---------------------------------------------------------------------------
# compare_sessions


def test_identical_records_match() -> None:
    record = make_record([text("user", "hi"), text("assistant", "ok")])
    diff = compare_sessions(record, record)
    assert diff.matches is True
    assert all(t.matches for t in diff.turns)
    assert len(diff.turns) == 2


def test_differing_text_does_not_match() -> None:
    a = make_record([text("user", "hi"), text("assistant", "yes")])
    b = make_record([text("user", "hi"), text("assistant", "no")])
    diff = compare_sessions(a, b, name="case-x")
    assert diff.matches is False
    assert diff.name == "case-x"
    assert diff.turns[0].matches is True
    assert diff.turns[1].matches is False
    assert diff.turns[1].a_text == "yes"
    assert diff.turns[1].b_text == "no"


def test_length_mismatch_fills_missing_side_with_none() -> None:
    a = make_record([text("user", "hi"), text("assistant", "ok")])
    b = make_record([text("user", "hi")])
    diff = compare_sessions(a, b)
    assert diff.matches is False
    assert len(diff.turns) == 2
    assert diff.turns[1].matches is False
    assert diff.turns[1].b_text is None


def test_multi_block_text_concatenates_for_comparison() -> None:
    """A message with two text blocks must compare equal to one combined block."""
    a = make_record(
        [
            text("user", "hi"),
            Message(
                role="assistant",
                content=[
                    ContentBlock(type="text", text="Hello "),
                    ContentBlock(type="text", text="World"),
                ],
            ),
        ]
    )
    b = make_record(
        [
            text("user", "hi"),
            Message(
                role="assistant",
                content=[ContentBlock(type="text", text="Hello World")],
            ),
        ]
    )
    diff = compare_sessions(a, b)
    assert diff.matches is True


def test_tool_use_id_differences_are_ignored() -> None:
    """Identical tool_use blocks with different ids should compare equal."""
    call_a = ToolCall(name="echo", arguments={"text": "hi"}, id="tu_111")
    call_b = ToolCall(name="echo", arguments={"text": "hi"}, id="tu_222")
    result_a = ToolResult(id="tu_111", content="hi", is_error=False)
    result_b = ToolResult(id="tu_222", content="hi", is_error=False)

    a_msgs = [
        text("user", "echo"),
        Message(role="assistant", content=[ContentBlock(type="tool_use", tool_use=call_a)]),
        Message(role="user", content=[ContentBlock(type="tool_result", tool_result=result_a)]),
    ]
    b_msgs = [
        text("user", "echo"),
        Message(role="assistant", content=[ContentBlock(type="tool_use", tool_use=call_b)]),
        Message(role="user", content=[ContentBlock(type="tool_result", tool_result=result_b)]),
    ]

    diff = compare_sessions(make_record(a_msgs), make_record(b_msgs))
    assert diff.matches is True


def test_tool_use_argument_difference_is_caught() -> None:
    """Arguments still matter — IDs are the only ignored axis."""
    call_a = ToolCall(name="echo", arguments={"text": "hi"}, id="tu_1")
    call_b = ToolCall(name="echo", arguments={"text": "bye"}, id="tu_1")

    a_msgs = [
        Message(role="assistant", content=[ContentBlock(type="tool_use", tool_use=call_a)]),
    ]
    b_msgs = [
        Message(role="assistant", content=[ContentBlock(type="tool_use", tool_use=call_b)]),
    ]
    diff = compare_sessions(make_record(a_msgs), make_record(b_msgs))
    assert diff.matches is False
