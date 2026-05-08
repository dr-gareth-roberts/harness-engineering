from __future__ import annotations

import importlib
import json
import sys

import pytest
from pydantic import BaseModel

from harness.agents import SubAgent
from harness.hooks import HookRunner, PostToolUse, PreToolUse
from harness.policy import AllowList, attach_pre_tool_policies
from harness.prompts import Message, text
from harness.runner.openai_compat import (
    OpenAICompatRunner,
    _serialize_tool_content,
    _translate_in,
    _translate_tools,
)
from harness.tools import Dispatcher, Tool
from harness.tools.schema import ToolCall, ToolResult
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
        Dispatcher([Tool(name="echo", description="Echo back.", input_model=EchoIn, handler=echo)]),
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
    schemas = [{"name": "echo", "description": "echoes", "input_schema": {"type": "object"}}]
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
        dispatcher,
        HookRunner(),
        client=client,
        max_iterations=3,  # type: ignore[arg-type]
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
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
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


# ---------------------------------------------------------------------------
# Speculator wiring (Wave 4 pre-step — mirrors AnthropicRunner)


class _StubSpeculator:
    """Minimal SpeculatorProtocol implementation for runner integration tests."""

    def __init__(self, hits: dict[str, ToolResult] | None = None) -> None:
        self.hits = dict(hits or {})
        self.begin_calls: list[dict[str, object]] = []
        self.try_resolve_calls: list[ToolCall] = []
        self.end_calls = 0

    async def begin(
        self,
        *,
        history: list[Message],
        agent: SubAgent,
        dispatcher: Dispatcher,
        hooks: HookRunner,
    ) -> None:
        self.begin_calls.append({"history_len": len(history), "agent_name": agent.name})

    async def try_resolve(self, call: ToolCall) -> ToolResult | None:
        self.try_resolve_calls.append(call)
        return self.hits.get(call.name)

    async def end(self) -> None:
        self.end_calls += 1


async def test_speculator_begin_and_end_fire_per_iteration_oai() -> None:
    response = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="hi"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    spec = _StubSpeculator()

    runner = OpenAICompatRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "hello")])

    assert len(spec.begin_calls) == 1
    assert spec.end_calls == 1
    assert spec.try_resolve_calls == []  # no tool_calls → no try_resolve


async def test_speculator_hit_skips_runner_dispatch_and_hooks_oai() -> None:
    """A try_resolve hit means the speculator already fired PreToolUse /
    dispatch / PostToolUse around its own dispatch. Runner must skip both."""
    tool_call = FakeOAToolCall(
        id="tc_1",
        function=FakeOAFunction(name="echo", arguments='{"text": "hi"}'),
    )
    response_1 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        ]
    )
    response_2 = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="done"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()

    cached = ToolResult(id="tc_1", content="(from speculation)", is_error=False)
    spec = _StubSpeculator(hits={"echo": cached})

    pre_calls: list[ToolCall] = []
    post_calls: list[tuple[ToolCall, ToolResult]] = []
    hooks = HookRunner()
    hooks.register(PreToolUse, lambda e: pre_calls.append(e.call) or None)
    hooks.register(
        PostToolUse,
        lambda e: post_calls.append((e.call, e.result)) or None,
    )

    runner = OpenAICompatRunner(
        dispatcher,
        hooks,
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "echo hi")])

    assert len(spec.try_resolve_calls) == 1
    assert spec.try_resolve_calls[0].name == "echo"
    # Runner did NOT fire its own hooks for the speculatively-resolved call.
    assert pre_calls == []
    assert post_calls == []
    assert dispatch_log == []
    # Two iterations -> two begin/end pairs.
    assert len(spec.begin_calls) == 2
    assert spec.end_calls == 2


async def test_speculator_miss_falls_back_to_runner_hooks_and_dispatch_oai() -> None:
    """When try_resolve returns None, the runner takes over normally."""
    tool_call = FakeOAToolCall(
        id="tc_1",
        function=FakeOAFunction(name="echo", arguments='{"text": "hi"}'),
    )
    response_1 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        ]
    )
    response_2 = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()
    spec = _StubSpeculator()  # no hits -> always miss

    pre_calls: list[ToolCall] = []
    hooks = HookRunner()
    hooks.register(PreToolUse, lambda e: pre_calls.append(e.call) or None)

    runner = OpenAICompatRunner(
        dispatcher,
        hooks,
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "echo hi")])

    assert len(spec.try_resolve_calls) == 1
    assert len(pre_calls) == 1
    assert dispatch_log == ["hi"]


async def test_speculator_end_fires_even_on_iteration_error_oai() -> None:
    """begin/end must be paired even when the iteration raises."""
    bad = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content="..."),
                finish_reason="content_filter",  # unhandled
            )
        ]
    )
    client = FakeAsyncOpenAI(responses=[bad])
    dispatcher, _ = _echo_dispatcher()
    spec = _StubSpeculator()

    runner = OpenAICompatRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    with pytest.raises(RuntimeError, match=r"Unexpected finish_reason"):
        await runner(_agent(), [text("user", "hi")])

    assert len(spec.begin_calls) == 1
    assert spec.end_calls == 1


async def test_running_history_grows_per_iteration_oai() -> None:
    """begin.history reflects in-loop turns the caller never sees."""
    tool_call = FakeOAToolCall(
        id="tc_1",
        function=FakeOAFunction(name="echo", arguments='{"text": "hi"}'),
    )
    response_1 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        ]
    )
    response_2 = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="done"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response_1, response_2])
    dispatcher, _ = _echo_dispatcher()
    spec = _StubSpeculator()

    runner = OpenAICompatRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "echo hi")])

    # Iter 1: just the user input. Iter 2: user + assistant(tool_calls)
    # + user(tool_result).
    assert spec.begin_calls[0]["history_len"] == 1
    assert spec.begin_calls[1]["history_len"] == 3
