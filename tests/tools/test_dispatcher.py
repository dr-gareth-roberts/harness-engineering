from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness.tools import Dispatcher, Tool, ToolCall


class AddIn(BaseModel):
    a: int
    b: int


def add(args: AddIn) -> int:
    return args.a + args.b


def make_dispatcher() -> Dispatcher:
    return Dispatcher(
        [Tool(name="add", description="Add two ints.", input_model=AddIn, handler=add)]
    )


async def test_happy_path() -> None:
    dispatcher = make_dispatcher()
    result = await dispatcher.dispatch(ToolCall(name="add", arguments={"a": 2, "b": 3}))
    assert result.is_error is False
    assert result.content == 5


async def test_validation_error_returns_error_result() -> None:
    dispatcher = make_dispatcher()
    result = await dispatcher.dispatch(
        ToolCall(name="add", arguments={"a": "not-an-int", "b": 3})
    )
    assert result.is_error is True
    assert "a" in str(result.content)


async def test_unknown_tool_returns_error_result() -> None:
    dispatcher = make_dispatcher()
    result = await dispatcher.dispatch(ToolCall(name="missing", arguments={}))
    assert result.is_error is True
    assert "missing" in str(result.content)


async def test_handler_exception_returns_error_result() -> None:
    class BoomIn(BaseModel):
        pass

    def boom(args: BoomIn) -> None:
        raise RuntimeError("kaboom")

    dispatcher = Dispatcher(
        [Tool(name="boom", description="Always fails.", input_model=BoomIn, handler=boom)]
    )
    result = await dispatcher.dispatch(ToolCall(name="boom", arguments={}))
    assert result.is_error is True
    assert "kaboom" in str(result.content)


def test_duplicate_registration_raises() -> None:
    dispatcher = make_dispatcher()
    with pytest.raises(ValueError, match="already registered"):
        dispatcher.register(
            Tool(name="add", description="Dup.", input_model=AddIn, handler=add)
        )


def test_tools_schema_lists_all() -> None:
    dispatcher = make_dispatcher()
    schemas = dispatcher.tools_schema()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "add"
