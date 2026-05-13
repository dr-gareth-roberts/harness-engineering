from __future__ import annotations

import importlib
import json
import logging
import sys
from typing import Any, cast

import pytest
from pydantic import BaseModel

from harness.agents import SubAgent
from harness.hooks import HookRunner, PostToolUse, PreToolUse
from harness.policy import AllowList, attach_pre_tool_policies
from harness.prompts import ContentBlock, Message, text
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
        client=client,  # type: ignore[arg-type]
        max_iterations=3,
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

    with pytest.raises(ImportError, match=r"harness-engineering-toolkit\[openai-compat\]"):
        importlib.import_module("harness.runner.openai_compat")


# ---------------------------------------------------------------------------
# Speculator wiring (Wave 4 pre-step — mirrors AnthropicRunner)


class _StubSpeculator:
    """Minimal SpeculatorProtocol implementation for runner integration tests."""

    def __init__(self, hits: dict[str, ToolResult] | None = None) -> None:
        self.hits = dict(hits or {})
        self.begin_calls: list[dict[str, object]] = []
        self.observe_calls: list[ToolCall] = []
        self.cancel_unobserved_calls = 0
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

    async def observe(self, call: ToolCall) -> None:
        self.observe_calls.append(call)

    async def cancel_unobserved(self) -> None:
        self.cancel_unobserved_calls += 1

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
    hooks.register(PreToolUse, lambda e: pre_calls.append(e.call))
    hooks.register(
        PostToolUse,
        lambda e: post_calls.append((e.call, e.result)),
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
    hooks.register(PreToolUse, lambda e: pre_calls.append(e.call))

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


# ---------------------------------------------------------------------------
# Wave 10 #6: per-iteration timeout


async def test_oa_runner_timeout_raises_when_create_takes_too_long() -> None:
    """timeout_s wraps chat.completions.create in asyncio.wait_for. A fake
    that sleeps longer than the timeout raises TimeoutError."""
    response = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response], create_delay=0.2)
    dispatcher, _ = _echo_dispatcher()

    runner = OpenAICompatRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        timeout_s=0.05,
    )

    with pytest.raises(TimeoutError):
        await runner(_agent(), [text("user", "hi")])


async def test_oa_runner_no_timeout_completes_normally_with_slow_call() -> None:
    response = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response], create_delay=0.05)
    dispatcher, _ = _echo_dispatcher()

    runner = OpenAICompatRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
    )

    result = await runner(_agent(), [text("user", "hi")])
    assert isinstance(result, Message)


# ---------------------------------------------------------------------------
# Wave 10 #5: HookDecision.replacement (OpenAICompat parity)


async def test_oa_pre_tool_use_replacement_skips_dispatch() -> None:
    from harness.hooks.events import HookDecision

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

    hooks = HookRunner()
    hooks.register(
        PreToolUse,
        lambda e: HookDecision(replacement=ToolResult(content="injected", is_error=False)),
    )

    runner = OpenAICompatRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]
    await runner(_agent(), [text("user", "echo hi")])

    assert dispatch_log == []
    second = client.chat.completions.requests[1]
    tool_result_msg = next(m for m in second["messages"] if m.get("role") == "tool")
    assert tool_result_msg["tool_call_id"] == "tc_1"
    assert tool_result_msg["content"] == "injected"


# ---------------------------------------------------------------------------
# Wave 10 #3: speculator observe + cancel_unobserved on OpenAICompat


async def test_oa_runner_calls_observe_for_each_tool_call_in_response() -> None:
    """OpenAICompat parity with AnthropicRunner Wave 6: each emitted
    tool_call surfaces to speculator.observe() before dispatch begins."""
    tc1 = FakeOAToolCall(
        id="tc_1",
        function=FakeOAFunction(name="echo", arguments='{"text": "first"}'),
    )
    tc2 = FakeOAToolCall(
        id="tc_2",
        function=FakeOAFunction(name="echo", arguments='{"text": "second"}'),
    )
    response_1 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content=None, tool_calls=[tc1, tc2]),
                finish_reason="tool_calls",
            )
        ]
    )
    response_2 = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
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
    await runner(_agent(), [text("user", "echo")])

    # observe fired twice in iteration 1 (one per tool_call), zero in
    # iteration 2 (text-only response).
    assert [c.id for c in spec.observe_calls] == ["tc_1", "tc_2"]
    # cancel_unobserved fires once per iteration that had any tool_calls
    # — and once per iteration regardless if speculator was active.
    # Pre-Wave-10 #3, this would be 0; now it's 2 (once per iteration).
    assert spec.cancel_unobserved_calls == 2


async def test_oa_runner_with_speculator_none_does_not_observe() -> None:
    """speculator=None path skips the observe loop cleanly."""
    tc = FakeOAToolCall(
        id="tc_1",
        function=FakeOAFunction(name="echo", arguments='{"text": "hi"}'),
    )
    response_1 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content=None, tool_calls=[tc]),
                finish_reason="tool_calls",
            )
        ]
    )
    response_2 = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="done"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()

    runner = OpenAICompatRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        # speculator=None
    )
    result = await runner(_agent(), [text("user", "echo hi")])

    assert dispatch_log == ["hi"]
    assert isinstance(result, Message)


# ---------------------------------------------------------------------------
# M1.10 — empty-content assistant rows are filtered before the wire
#
# Some OpenAI-compatible servers (vLLM, llama.cpp) reject assistant
# messages that carry neither text content nor tool_calls. The
# translator must drop those rows; a strict-fake server here simulates
# the rejection so a regression would fail loudly.


class _StrictNoEmptyAssistantCompletions:
    """Fake `chat.completions` that rejects empty-content assistant rows
    with no tool_calls — mirrors vLLM / llama.cpp behavior."""

    def __init__(self, responses: list[FakeOAResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> FakeOAResponse:
        messages = kwargs.get("messages") or []
        assert isinstance(messages, list)
        for m in messages:
            assert isinstance(m, dict)
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            tool_calls = m.get("tool_calls")
            content_is_empty = content in (None, "") or (isinstance(content, list) and not content)
            if content_is_empty and not tool_calls:
                raise RuntimeError(
                    "fake backend rejected assistant message with empty content and no tool_calls"
                )
        if not self._responses:
            raise RuntimeError("strict fake: no canned responses left")
        self.requests.append(kwargs)
        return self._responses.pop(0)


class _StrictNoEmptyAssistantChat:
    def __init__(self, responses: list[FakeOAResponse]) -> None:
        self.completions = _StrictNoEmptyAssistantCompletions(responses)


class _StrictNoEmptyAssistantClient:
    def __init__(self, responses: list[FakeOAResponse]) -> None:
        self.chat = _StrictNoEmptyAssistantChat(responses)


def test_translate_in_drops_empty_assistant_with_no_tool_calls() -> None:
    """Direct translator test: an assistant Message with no text and no
    tool_use blocks must not appear in the output payload."""
    msgs: list[Message] = [
        text("user", "hello"),
        Message(role="assistant", content=[]),  # entirely empty assistant
        text("user", "follow-up"),
    ]
    out = _translate_in(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["user", "user"]


def test_translate_in_keeps_assistant_with_tool_calls_only() -> None:
    """Empty text but with tool_use blocks is valid — keep the row so
    the tool_call_id chain stays intact."""
    msgs: list[Message] = [
        text("user", "echo"),
        Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name="echo", arguments={"text": "hi"}, id="tc_1"),
                ),
            ],
        ),
        Message(
            role="user",
            content=[
                ContentBlock(
                    type="tool_result",
                    tool_result=ToolResult(id="tc_1", content="hi"),
                ),
            ],
        ),
    ]
    out = _translate_in(msgs)
    assistant_rows = [m for m in out if m["role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == ""
    assert assistant_rows[0]["tool_calls"][0]["function"]["name"] == "echo"


async def test_strict_backend_does_not_receive_empty_assistant_rows() -> None:
    """Integration test against a server that rejects empty-content
    assistants. The runner must not pass one through, even after a
    tool loop."""
    tool_use = FakeOAToolCall(
        id="tc_1",
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
        choices=[FakeOAChoice(message=FakeOAMessage(content="echoed: hi"), finish_reason="stop")]
    )
    client = _StrictNoEmptyAssistantClient(responses=[response_1, response_2])
    dispatcher, log = _echo_dispatcher()
    runner = OpenAICompatRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    # If the empty-row filter ever regresses, the strict fake will raise
    # on the second iteration (where the assistant tool_calls row is
    # appended) or on a prior assistant-only round trip.
    result = await runner(_agent(), [text("user", "echo hi")])
    assert log == ["hi"]
    assert result.content[0].text == "echoed: hi"


async def test_strict_backend_rejects_when_assistant_message_is_replayed_with_no_text() -> None:
    """Pass a prior assistant Message with no text and no tool_use as
    part of the input. With the filter, the strict backend accepts the
    payload; without it, the backend would raise."""
    msgs: list[Message] = [
        text("user", "hi"),
        Message(role="assistant", content=[]),  # would be rejected pre-fix
        text("user", "go on"),
    ]
    response = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
    )
    client = _StrictNoEmptyAssistantClient(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = OpenAICompatRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    result = await runner(_agent(), msgs)
    assert result.content[0].text == "ok"
    # And confirm the request that did land has no empty-content assistant.
    sent = cast(list[dict[str, Any]], client.chat.completions.requests[0]["messages"])
    for m in sent:
        if m["role"] == "assistant":
            assert m["content"] or m.get("tool_calls")


# ---------------------------------------------------------------------------
# M1.24 — malformed tool-call JSON surfaces as an is_error ToolResult
#
# Previously the runner silently called the tool with empty args. Now
# the parse failure becomes a visible trajectory entry and emits a
# WARNING log.


async def test_malformed_tool_call_json_synthesizes_error_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool_call with non-JSON arguments must NOT reach dispatch.
    Instead the runner returns a ToolResult(is_error=True, ...) and
    emits a WARNING."""
    bad_tool_call = FakeOAToolCall(
        id="tc_bad",
        function=FakeOAFunction(name="echo", arguments="{not valid json"),
    )
    response_1 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content=None, tool_calls=[bad_tool_call]),
                finish_reason="tool_calls",
            )
        ]
    )
    response_2 = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="recovered"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()

    pre_calls: list[ToolCall] = []
    post_calls: list[tuple[ToolCall, ToolResult]] = []
    hooks = HookRunner()
    hooks.register(PreToolUse, lambda e: pre_calls.append(e.call))
    hooks.register(PostToolUse, lambda e: post_calls.append((e.call, e.result)))

    runner = OpenAICompatRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING, logger="harness.runner.openai_compat"):
        result = await runner(_agent(), [text("user", "do it")])

    # Tool was NOT dispatched, and PreToolUse / PostToolUse hooks did
    # not fire — the parse-failure path is its own short circuit.
    assert dispatch_log == []
    assert pre_calls == []
    assert post_calls == []

    # The error ToolResult must reach the model on the next iteration.
    second_request_messages = client.chat.completions.requests[1]["messages"]
    tool_msg = next(m for m in second_request_messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "tc_bad"
    assert "malformed tool-call JSON" in tool_msg["content"]

    # The final assistant response is unaffected (the model "recovered").
    assert result.content[0].text == "recovered"

    # A WARNING log entry was emitted naming the tool and id.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("malformed tool-call JSON" in r.getMessage() for r in warnings)
    assert any("tc_bad" in r.getMessage() for r in warnings)


async def test_malformed_tool_call_json_does_not_fire_speculator_try_resolve() -> None:
    """The dispatch short-circuit also skips the speculator hit-check —
    the failure is a runner-level concern, not a prediction one."""
    bad_tool_call = FakeOAToolCall(
        id="tc_bad",
        function=FakeOAFunction(name="echo", arguments="{broken"),
    )
    response_1 = FakeOAResponse(
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content=None, tool_calls=[bad_tool_call]),
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
    await runner(_agent(), [text("user", "go")])

    # observe still fires (it has its own try/except and parses
    # independently), but try_resolve does NOT — dispatch was bypassed.
    assert spec.try_resolve_calls == []


# ---------------------------------------------------------------------------
# M1.25 — caller-supplied system message is not duplicated


async def test_caller_system_message_skips_agent_system_prepend() -> None:
    """If the caller already passes a system message, the runner must
    not prepend `agent.system_prompt` on top of it."""
    response = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = OpenAICompatRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    caller_system = "Only follow the caller's instructions."
    await runner(
        _agent(),
        [text("system", caller_system), text("user", "hi")],
    )

    sent = client.chat.completions.requests[0]["messages"]
    system_rows = [m for m in sent if m["role"] == "system"]
    assert len(system_rows) == 1
    assert system_rows[0]["content"] == caller_system
    # The default agent system prompt must NOT appear anywhere.
    assert not any("small test agent" in (m.get("content") or "") for m in sent)


async def test_agent_system_prompt_still_used_when_caller_has_none() -> None:
    """Sanity check: the absence-of-caller-system path is unchanged —
    the agent's prompt is still prepended."""
    response = FakeOAResponse(
        choices=[FakeOAChoice(message=FakeOAMessage(content="ok"), finish_reason="stop")]
    )
    client = FakeAsyncOpenAI(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = OpenAICompatRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    await runner(_agent(), [text("user", "hi")])
    sent = client.chat.completions.requests[0]["messages"]
    system_rows = [m for m in sent if m["role"] == "system"]
    assert len(system_rows) == 1
    assert "small test agent" in system_rows[0]["content"]
