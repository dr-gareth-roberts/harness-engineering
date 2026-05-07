from __future__ import annotations

from harness.prompts import Message, assistant_tool_use, text, user_tool_result
from harness.tools import ToolCall, ToolResult


def test_text_helper_shape() -> None:
    msg = text("user", "hello", cache=True)
    assert msg.role == "user"
    assert len(msg.content) == 1
    block = msg.content[0]
    assert block.type == "text"
    assert block.text == "hello"
    assert block.cache is True


def test_text_default_no_cache() -> None:
    msg = text("system", "instructions")
    assert msg.content[0].cache is False


def test_assistant_tool_use_helper() -> None:
    call = ToolCall(name="echo", arguments={"text": "hi"}, id="c1")
    msg = assistant_tool_use(call)
    assert msg.role == "assistant"
    assert msg.content[0].type == "tool_use"
    assert msg.content[0].tool_use == call


def test_user_tool_result_helper() -> None:
    res = ToolResult(id="c1", content="hi", is_error=False)
    msg = user_tool_result(res)
    assert msg.role == "user"
    assert msg.content[0].type == "tool_result"
    assert msg.content[0].tool_result == res


def test_round_trip_preserves_cache_flag() -> None:
    msg = text("user", "x", cache=True)
    revived = Message.model_validate(msg.model_dump())
    assert revived.content[0].cache is True
