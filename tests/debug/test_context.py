from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from harness.debug.context import DebugContext
from harness.prompts.messages import ContentBlock, Message, text
from harness.tools.dispatcher import Dispatcher
from harness.tools.schema import Tool, ToolCall

# ---------- helpers


class _EchoArgs(BaseModel):
    payload: str


def _echo_tool() -> Tool:
    def handler(args: _EchoArgs) -> str:
        return f"echo:{args.payload}"

    return Tool(
        name="echo",
        description="echoes its payload back",
        input_model=_EchoArgs,
        handler=handler,
    )


def _make_messages_with_tool_use() -> list[Message]:
    call = ToolCall(name="delete", arguments={"path": "/tmp/x"}, id="tu_1")
    return [
        text("user", "delete the temp dir"),
        Message(
            role="assistant",
            content=[ContentBlock(type="tool_use", tool_use=call)],
        ),
    ]


# ---------- tests for spec test #1: DebugContext exposure


def test_debug_context_exposes_messages_last_call_and_turn_index() -> None:
    msgs = _make_messages_with_tool_use()
    last_call = ToolCall(name="delete", arguments={"path": "/tmp/x"}, id="tu_1")
    ctx = DebugContext(msgs, last_call=last_call, turn_index=1)

    assert ctx.messages == msgs
    assert ctx.last_call is not None
    assert ctx.last_call.name == "delete"
    assert ctx.last_call.arguments == {"path": "/tmp/x"}
    assert ctx.turn_index == 1


def test_debug_context_messages_is_a_defensive_copy() -> None:
    msgs = [text("user", "hi")]
    ctx = DebugContext(msgs)

    # Mutating the source list should not bleed into the context view.
    msgs.append(text("user", "bye"))

    assert len(ctx.messages) == 1
    assert ctx.messages[0].content[0].text == "hi"


def test_debug_context_messages_property_returns_fresh_copy() -> None:
    """Caller should not be able to corrupt internal state via .messages."""
    ctx = DebugContext([text("user", "a")])
    snapshot = ctx.messages
    snapshot.append(text("user", "b"))
    assert len(ctx.messages) == 1


# ---------- tests for spec test #2: mutate replaces next message


def test_mutate_queues_replacement_message() -> None:
    ctx = DebugContext([text("user", "hi")])
    replacement = text("assistant", "stop")

    assert ctx.pending_mutation is None
    ctx.mutate(replacement)
    assert ctx.pending_mutation is replacement


def test_mutate_idempotent_only_last_wins() -> None:
    ctx = DebugContext([text("user", "hi")])
    ctx.mutate(text("assistant", "first"))
    ctx.mutate(text("assistant", "second"))

    pending = ctx.pending_mutation
    assert pending is not None
    assert pending.content[0].text == "second"


def test_mutate_rejects_non_messages() -> None:
    ctx = DebugContext([text("user", "hi")])
    with pytest.raises(TypeError):
        ctx.mutate("not a message")  # type: ignore[arg-type]


# ---------- tests for spec test #3: fire dispatches without advancing


async def test_fire_dispatches_through_dispatcher_and_returns_result() -> None:
    dispatcher = Dispatcher([_echo_tool()])
    ctx = DebugContext([text("user", "hi")], dispatcher=dispatcher)

    result = await ctx.fire("echo", {"payload": "hello"})

    assert result.is_error is False
    assert result.content == "echo:hello"


async def test_fire_does_not_advance_conversation() -> None:
    dispatcher = Dispatcher([_echo_tool()])
    ctx = DebugContext([text("user", "hi")], dispatcher=dispatcher)
    before = list(ctx.messages)

    await ctx.fire("echo", {"payload": "x"})

    assert ctx.messages == before
    assert ctx.pending_mutation is None


async def test_fire_without_dispatcher_raises_clear_error() -> None:
    ctx = DebugContext([text("user", "hi")])
    with pytest.raises(RuntimeError, match="Dispatcher"):
        await ctx.fire("echo", {"payload": "x"})


# ---------- inspect / resume / abort


def test_inspect_runs_callable_and_returns_value() -> None:
    ctx = DebugContext([text("user", "a"), text("user", "b")])
    out = ctx.inspect(lambda c: len(c.messages))
    assert out == 2


def test_inspect_passes_context_argument() -> None:
    ctx = DebugContext([text("user", "hi")], turn_index=4)
    captured: list[Any] = []
    ctx.inspect(lambda c: captured.append(c.turn_index))
    assert captured == [4]


def test_resume_marks_resumed() -> None:
    ctx = DebugContext([text("user", "a")])
    assert ctx.resumed is False
    ctx.resume()
    assert ctx.resumed is True


def test_abort_marks_aborted_and_blocks_resume() -> None:
    ctx = DebugContext([text("user", "a")])
    ctx.abort()
    assert ctx.aborted is True
    with pytest.raises(RuntimeError):
        ctx.resume()
