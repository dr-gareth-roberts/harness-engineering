from __future__ import annotations

import importlib
import json
import sys

import pytest
from pydantic import BaseModel

from harness.agents import SubAgent
from harness.hooks import HookRunner
from harness.policy import AllowList, attach_pre_tool_policies
from harness.prompts import Message, text
from harness.runner.openai_compat import (
    OpenAICompatRunner,
    _serialize_tool_content,
    _translate_in,
    _translate_tools,
)
from harness.tools import Dispatcher, Tool
from tests.runner.fakes_openai import (
    FakeAsyncOpenAI,
    FakeOAChoice,
    FakeOAFunction,
    FakeOAMessage,
    FakeOAResponse,
    FakeOAToolCall,
)

# ---------------------------------------------------------------------------
# helpers


class EchoIn(BaseModel):
    text: str


def _echo_dispatcher() -> tuple[Dispatcher, list[str]]:
    log: list[str] = []

    def echo(args: EchoIn) -> str:
        log.append(args.text)
        return args.text

    return (
        Dispatcher(
            [Tool(name="echo", description="Echo back.", input_model=EchoIn, handler=echo)]
        ),
        log,
    )


def _agent(*, allowed_tools: list[str] | None = None) -> SubAgent:
    return SubAgent(
        name="t",
        system_prompt="You are a small test agent.",
        model="gpt-test",
        allowed_tools=allowed_tools if allowed_tools is not None else ["echo"],
    )


# ---------------------------------------------------------------------------
# translation


def test_translate_tools_wraps_in_function_shape() -> None:
    schemas = [
        {"name": "echo", "description": "echoes", "input_schema": {"type": "object"}}
    ]
    out = _translate_tools(schemas)
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echoes",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_translate_in_keeps_system_as_message() -> None:
    msgs: list[Message] = [text("system", "be helpful"), text("user", "hi")]
    out = _translate_in(msgs)
    assert out == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]


def test_serialize_tool_content_handles_dicts() -> None:
    assert _serialize_tool_content({"a": 1}) == '{"a": 1}'
    assert _serialize_tool_content("plain") == "plain"
    assert _serialize_tool_content(42) == "42"


# ---------------------------------------------------------------------------
# loop behaviour


async def test_no_tool_happy_path() -> None:
    response = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content="hello there"),
                finish_reason="stop",
            )
        ]
    )
    client = FakeAsyncOpenAI(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = OpenAICompatRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    result = await runner(_agent(), [text("user", "hi")])
    assert result.role == "assistant"
    assert result.content[0].text == "hello there"


async def test_one_iteration_tool_loop_dispatches_and_continues() -> None:
    tool_use = FakeOAToolCall(
        id="call_1",
        function=FakeOAFunction(name="echo", arguments=json.dumps({"text": "hi"})),
    )
    response_1 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content=None, tool_calls=[tool_use]),
                finish_reason="tool_calls",
            )
        ]
    )
    response_2 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content="echoed: hi"),
                finish_reason="stop",
            )
        ]
    )
    client = FakeAsyncOpenAI(responses=[response_1, response_2])
    dispatcher, log = _echo_dispatcher()
    runner = OpenAICompatRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    result = await runner(_agent(), [text("user", "echo hi")])
    assert log == ["hi"]
    assert result.content[0].text == "echoed: hi"

    second = client.chat.completions.requests[1]
    roles = [m["role"] for m in second["messages"]]
    # Includes the assistant tool_calls turn AND the tool-result turn.
    assert "assistant" in roles
    assert "tool" in roles
    tool_msg = next(m for m in second["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "hi"


async def test_hook_block_short_circuits_dispatch() -> None:
    tool_use = FakeOAToolCall(
        id="call_1",
        function=FakeOAFunction(name="echo", arguments=json.dumps({"text": "hi"})),
    )
    response_1 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content=None, tool_calls=[tool_use]),
                finish_reason="tool_calls",
            )
        ]
    )
    response_2 = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response_1, response_2])
    dispatcher, log = _echo_dispatcher()
    hooks = HookRunner()
    attach_pre_tool_policies(hooks, AllowList.of({"approved-only"}))
    runner = OpenAICompatRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]

    await runner(_agent(allowed_tools=["echo"]), [text("user", "go")])

    assert log == []
    second = client.chat.completions.requests[1]
    tool_msg = next(m for m in second["messages"] if m["role"] == "tool")
    assert "echo" in tool_msg["content"]


async def test_max_iterations_cap_raises() -> None:
    responses = [
        FakeOAResponse(
            choices=[
                FakeOAChoice(
                    message=FakeOAMessage(
                        content=None,
                        tool_calls=[
                            FakeOAToolCall(
                                id=f"c{i}",
                                function=FakeOAFunction(
                                    name="echo",
                                    arguments=json.dumps({"text": "x"}),
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )
        for i in range(5)
    ]
    client = FakeAsyncOpenAI(responses=responses)
    dispatcher, _ = _echo_dispatcher()
    runner = OpenAICompatRunner(
        dispatcher, HookRunner(), client=client, max_iterations=3  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="max_iterations=3"):
        await runner(_agent(), [text("user", "loop")])


async def test_unexpected_finish_reason_raises() -> None:
    response = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content="..."),
                finish_reason="content_filter",
            )
        ]
    )
    client = FakeAsyncOpenAI(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = OpenAICompatRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="content_filter"):
        await runner(_agent(), [text("user", "x")])


async def test_allowed_tools_filter_excludes_unlisted_tools() -> None:
    response = FakeOAResponse(
        choices=[
            FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")
        ]
    )
    client = FakeAsyncOpenAI(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = OpenAICompatRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    await runner(_agent(allowed_tools=[]), [text("user", "x")])
    request = client.chat.completions.requests[0]
    assert "tools" not in request


async def test_system_prompt_is_prepended_as_system_message() -> None:
    response = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = OpenAICompatRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    await runner(_agent(), [text("user", "hi")])
    request = client.chat.completions.requests[0]
    assert request["messages"][0]["role"] == "system"
    assert "small test agent" in request["messages"][0]["content"]


# ---------------------------------------------------------------------------
# missing dep


def test_missing_openai_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.delitem(sys.modules, "harness.runner.openai_compat", raising=False)

    with pytest.raises(ImportError, match=r"harness-engineering\[openai-compat\]"):
        importlib.import_module("harness.runner.openai_compat")
