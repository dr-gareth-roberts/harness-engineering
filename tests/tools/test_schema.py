from __future__ import annotations

from pydantic import BaseModel

from harness.tools import Tool, ToolCall
from harness.tools.dispatcher import Dispatcher


class EchoIn(BaseModel):
    text: str


def echo_handler(args: EchoIn) -> str:
    return args.text


async def async_echo_handler(args: EchoIn) -> str:
    return args.text.upper()


def test_json_schema_shape() -> None:
    tool = Tool(
        name="echo",
        description="Echo back the input.",
        input_model=EchoIn,
        handler=echo_handler,
    )
    schema = tool.json_schema()
    assert schema["name"] == "echo"
    assert schema["description"] == "Echo back the input."
    assert schema["input_schema"]["properties"]["text"]["type"] == "string"
    assert schema["input_schema"]["required"] == ["text"]


async def test_sync_handler_dispatches() -> None:
    tool = Tool(
        name="echo",
        description="Echo.",
        input_model=EchoIn,
        handler=echo_handler,
    )
    dispatcher = Dispatcher([tool])
    result = await dispatcher.dispatch(ToolCall(name="echo", arguments={"text": "hi"}, id="c1"))
    assert result.is_error is False
    assert result.content == "hi"
    assert result.id == "c1"


async def test_async_handler_dispatches() -> None:
    tool = Tool(
        name="echo",
        description="Echo.",
        input_model=EchoIn,
        handler=async_echo_handler,
    )
    dispatcher = Dispatcher([tool])
    result = await dispatcher.dispatch(ToolCall(name="echo", arguments={"text": "hi"}))
    assert result.content == "HI"
