from __future__ import annotations

import time

from harness.agents import SubAgent
from harness.memory import SessionRecord
from harness.prompts import assistant_tool_use, text, user_tool_result
from harness.tools import ToolCall, ToolResult


def make_agent() -> SubAgent:
    return SubAgent(name="x", system_prompt="you are x")


def test_round_trip_preserves_all_fields() -> None:
    record = SessionRecord(
        session_id="s1",
        agent=make_agent(),
        messages=[text("user", "hi"), text("assistant", "hello")],
        metadata={"version": 3, "owner": "alice"},
    )
    serialized = record.model_dump_json()
    revived = SessionRecord.model_validate_json(serialized)

    assert revived.session_id == "s1"
    assert revived.agent.name == "x"
    assert [m.content[0].text for m in revived.messages] == ["hi", "hello"]
    assert revived.metadata == {"version": 3, "owner": "alice"}


def test_round_trip_preserves_tool_use_blocks() -> None:
    call = ToolCall(name="echo", arguments={"text": "hi"}, id="c1")
    result = ToolResult(id="c1", content="hi", is_error=False)
    record = SessionRecord(
        session_id="s2",
        agent=make_agent(),
        messages=[text("user", "echo hi"), assistant_tool_use(call), user_tool_result(result)],
    )
    revived = SessionRecord.model_validate_json(record.model_dump_json())

    assistant_block = revived.messages[1].content[0]
    assert assistant_block.type == "tool_use"
    assert assistant_block.tool_use is not None
    assert assistant_block.tool_use.name == "echo"

    result_block = revived.messages[2].content[0]
    assert result_block.type == "tool_result"
    assert result_block.tool_result is not None
    assert result_block.tool_result.content == "hi"


def test_touched_advances_updated_at_only() -> None:
    record = SessionRecord(session_id="s3", agent=make_agent())
    original_created = record.created_at
    original_updated = record.updated_at
    time.sleep(0.001)

    refreshed = record.touched()
    assert refreshed.created_at == original_created
    assert refreshed.updated_at > original_updated
