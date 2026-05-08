from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from pydantic import BaseModel

from harness.agents import SubAgent
from harness.hooks import HookRunner
from harness.policy import AllowList, attach_pre_tool_policies
from harness.prompts import Message, attach_file, text
from harness.runner.anthropic import (
    AnthropicRunner,
    _serialize_tool_content,
    _translate_in,
)
from harness.tools import Dispatcher, Tool
from tests.runner.fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeTextBlock,
    FakeToolUseBlock,
)

# ---------------------------------------------------------------------------
# helpers


class EchoIn(BaseModel):
    text: str


def _echo_dispatcher(*, calls: list[str] | None = None) -> tuple[Dispatcher, list[str]]:
    log: list[str] = calls if calls is not None else []

    def echo(args: EchoIn) -> str:
        log.append(args.text)
        return args.text

    return (
        Dispatcher(
            [Tool(name="echo", description="Echo it back.", input_model=EchoIn, handler=echo)]
        ),
        log,
    )


def _agent(*, allowed_tools: list[str] | None = None) -> SubAgent:
    return SubAgent(
        name="t",
        system_prompt="You are a small test agent.",
        model="claude-opus-4-7",
        allowed_tools=allowed_tools if allowed_tools is not None else ["echo"],
    )


# ---------------------------------------------------------------------------
# translation


def test_translate_in_extracts_system_messages() -> None:
    msgs: list[Message] = [
        text("system", "be helpful"),
        text("user", "hi"),
        text("system", "be terse"),
        text("user", "again"),
    ]
    api_messages, system = _translate_in(msgs)

    assert system == "be helpful\n\nbe terse"
    assert [m["role"] for m in api_messages] == ["user", "user"]
    assert api_messages[0]["content"] == [{"type": "text", "text": "hi"}]


def test_translate_in_propagates_cache_marker() -> None:
    msgs: list[Message] = [text("user", "cached", cache=True)]
    api_messages, _ = _translate_in(msgs)
    block = api_messages[0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}


def test_translate_in_renders_file_block_as_text(tmp_path: Any) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello")
    msgs: list[Message] = [
        Message(role="user", content=[attach_file(p)]),
    ]
    api_messages, _ = _translate_in(msgs)
    block = api_messages[0]["content"][0]
    assert block["type"] == "text"
    assert "<file path=" in block["text"]
    assert "hello" in block["text"]


def test_serialize_tool_content_handles_dict_and_list() -> None:
    assert _serialize_tool_content("plain") == "plain"
    assert _serialize_tool_content(42) == "42"
    assert _serialize_tool_content({"a": 1}) == '{"a": 1}'
    assert _serialize_tool_content([1, 2]) == "[1, 2]"


# ---------------------------------------------------------------------------
# loop behaviour


async def test_no_tool_happy_path() -> None:
    response = FakeMessage(
        content=[FakeTextBlock(text="hello there")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    result = await runner(_agent(), [text("user", "hi")])

    assert result.role == "assistant"
    assert len(result.content) == 1
    assert result.content[0].text == "hello there"


async def test_one_iteration_tool_loop_dispatches_and_continues() -> None:
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="echoed: hi")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, log = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    result = await runner(_agent(), [text("user", "echo hi")])
    assert log == ["hi"]
    assert result.content[0].text == "echoed: hi"

    # The second request must include the assistant's tool_use turn AND a
    # user turn carrying the tool_result.
    second = client.messages.requests[1]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["user", "assistant", "user"]
    tool_result_block = second["messages"][-1]["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["tool_use_id"] == "tu_1"
    assert tool_result_block["content"] == "hi"
    assert tool_result_block["is_error"] is False


async def test_hook_block_short_circuits_dispatch() -> None:
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, log = _echo_dispatcher()
    hooks = HookRunner()
    attach_pre_tool_policies(hooks, AllowList.of({"approved-only"}))
    runner = AnthropicRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]

    await runner(_agent(allowed_tools=["echo"]), [text("user", "go")])

    assert log == []  # dispatcher's handler never ran
    second = client.messages.requests[1]
    tool_result_block = second["messages"][-1]["content"][0]
    assert tool_result_block["is_error"] is True
    assert "echo" in tool_result_block["content"]


async def test_max_iterations_cap_raises() -> None:
    responses = [
        FakeMessage(
            content=[FakeToolUseBlock(id=f"tu_{i}", name="echo", input={"text": "x"})],
            stop_reason="tool_use",
        )
        for i in range(5)
    ]
    client = FakeAsyncAnthropic(responses=responses)
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,
        max_iterations=3,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="max_iterations=3"):
        await runner(_agent(), [text("user", "loop")])


async def test_unexpected_stop_reason_raises() -> None:
    response = FakeMessage(content=[FakeTextBlock(text="...")], stop_reason="refusal")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="refusal"):
        await runner(_agent(), [text("user", "x")])


async def test_allowed_tools_filter_excludes_unlisted_tools() -> None:
    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    await runner(_agent(allowed_tools=[]), [text("user", "x")])
    request = client.messages.requests[0]
    assert "tools" not in request


async def test_request_includes_thinking_and_effort_when_set() -> None:
    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        thinking_mode="adaptive",
        effort="high",
    )

    await runner(_agent(), [text("user", "x")])
    request = client.messages.requests[0]
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "high"}


async def test_thinking_omitted_when_disabled() -> None:
    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,
        thinking_mode="disabled",  # type: ignore[arg-type]
    )

    await runner(_agent(), [text("user", "x")])
    assert "thinking" not in client.messages.requests[0]


# ---------------------------------------------------------------------------
# missing dep


def test_missing_anthropic_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.delitem(sys.modules, "harness.runner.anthropic", raising=False)

    with pytest.raises(ImportError, match=r"harness-engineering\[anthropic\]"):
        importlib.import_module("harness.runner.anthropic")
