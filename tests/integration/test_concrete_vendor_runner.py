"""Concrete-vendor runner integration — Orchestrator + real runner + faked SDK.

Cross-package smoke test that exercises the *real* harness runner code path
(`AnthropicRunner` / `OpenAICompatRunner`) through `Orchestrator.run`, with
only the vendor SDK surface (the network/client boundary) faked.

This is the integration-honesty companion to the runner-protocol tests in
`tests/integration/test_orchestrator_runner_hooks_policy.py` (which uses a
local fake-runner callable) and to `tests/runner/test_anthropic.py` /
`tests/runner/test_openai_compat.py` (which call the runner directly,
bypassing `Orchestrator.run`). Codex review S3 flagged that the existing
integration suite did not exercise a concrete vendor runner end-to-end;
this file is that exercise.

What "real" vs "faked" means here:

- Real: `Orchestrator`, `Dispatcher`, `HookRunner`, `Tool`, `SubAgent`,
  `AnthropicRunner`, `OpenAICompatRunner`, the harness `Message` /
  `ToolCall` / `ToolResult` types, the runner's translation helpers
  and tool-use loop, and the hook firing inside the runner.
- Faked: the vendor SDK client only — `FakeAsyncAnthropic` mimics
  `anthropic.AsyncAnthropic.messages.stream(...)`, `FakeAsyncOpenAI`
  mimics `openai.AsyncOpenAI.chat.completions.create(...)`. No
  network IO; the canned responses drive the loop.

The honesty pin: the `PreToolUse` capture handler is registered on the
same `HookRunner` instance that is passed into the concrete runner's
constructor. The handler firing therefore proves the *runner itself*
emitted the event during its real internal `self.hooks.emit(...)` call
— not a re-implementation thereof.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner, PreToolUse
from harness.prompts import text
from harness.runner.anthropic import AnthropicRunner
from harness.runner.openai_compat import OpenAICompatRunner
from harness.tools import Dispatcher, Tool
from harness.tools.schema import ToolCall
from tests.runner.fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeTextBlock,
    FakeToolUseBlock,
)
from tests.runner.fakes_openai import (
    FakeAsyncOpenAI,
    FakeOAChoice,
    FakeOAFunction,
    FakeOAMessage,
    FakeOAResponse,
    FakeOAToolCall,
)


class _EchoArgs(BaseModel):
    """Single-field input model for the `echo` tool used by both tests."""

    text: str


def _echo_dispatcher() -> tuple[Dispatcher, list[str]]:
    """Build a dispatcher with one echo-style tool plus the call log.

    The handler appends its received `text` arg to `log` so the test can
    prove the *real* dispatcher actually ran the tool (rather than the
    runner skipping it or a hook short-circuiting via `replacement`).
    """
    log: list[str] = []

    def echo(args: _EchoArgs) -> str:
        log.append(args.text)
        return args.text

    return (
        Dispatcher(
            [
                Tool(
                    name="echo",
                    description="Echo the supplied text back.",
                    input_model=_EchoArgs,
                    handler=echo,
                ),
            ]
        ),
        log,
    )


async def test_anthropic_runner_end_to_end_through_orchestrator(
    make_agent: Callable[..., SubAgent],
) -> None:
    """Run a single user turn through `Orchestrator.run` against a real
    `AnthropicRunner` whose SDK boundary is a `FakeAsyncAnthropic`.

    The two canned vendor responses drive the standard tool-use loop:
    iteration 1 emits a `tool_use` block, iteration 2 emits the terminal
    text. The hook handler registered on the same `HookRunner` instance
    that the runner holds must see the `PreToolUse` event with the tool
    call the model emitted; the dispatcher's handler log must record the
    real dispatch; the orchestrator's return value must be the terminal
    assistant text the runner translated out of response 2.
    """
    # Canned vendor responses driving the runner's two-iteration loop.
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="echoed: hi")],
        stop_reason="end_turn",
    )
    fake_client = FakeAsyncAnthropic(responses=[response_1, response_2])

    dispatcher, dispatch_log = _echo_dispatcher()

    hooks = HookRunner()
    pre_tool_calls: list[ToolCall] = []

    def _capture_pre_tool(event: PreToolUse) -> None:
        # Observational — return None so the runner's dispatch path
        # continues (returning a `HookDecision(block=True)` here would
        # short-circuit dispatch and the dispatch_log would stay empty,
        # breaking the test's honesty guarantee).
        pre_tool_calls.append(event.call)

    hooks.register(PreToolUse, _capture_pre_tool)

    # Real `AnthropicRunner` constructed with the *real* dispatcher and
    # hooks; the `# type: ignore[arg-type]` matches the existing
    # `tests/runner/test_anthropic.py` pattern for injecting a fake
    # client without satisfying the SDK's exact type.
    runner = AnthropicRunner(
        dispatcher,
        hooks,
        client=cast(Any, fake_client),
    )

    orchestrator = Orchestrator(dispatcher, hooks, runner)
    agent = make_agent(model="claude-opus-4-7", allowed_tools=["echo"])

    reply = await orchestrator.run(agent, [text("user", "please echo hi")])

    # The dispatcher's real handler ran — the tool input came through
    # the runner's translation + the orchestrator's hand-off, not from
    # a local re-implementation.
    assert dispatch_log == ["hi"]

    # The hook registered on the same `HookRunner` instance the runner
    # holds saw the `PreToolUse` event with the call args from the
    # vendor response.
    assert len(pre_tool_calls) == 1
    captured = pre_tool_calls[0]
    assert captured.name == "echo"
    assert captured.arguments == {"text": "hi"}
    assert captured.id == "tu_1"

    # The orchestrator's return value is the terminal assistant text the
    # runner translated out of response 2.
    assert reply.role == "assistant"
    assert len(reply.content) == 1
    assert reply.content[0].type == "text"
    assert reply.content[0].text == "echoed: hi"

    # The runner made exactly two SDK requests — one per loop iteration.
    # The second request carries the assistant tool_use turn and a user
    # tool_result turn synthesised from the dispatched result, proving
    # the real runner's tool-use-loop bookkeeping (not a fake's) ran.
    assert len(fake_client.messages.requests) == 2
    second_request = fake_client.messages.requests[1]
    tool_result_block = second_request["messages"][-1]["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["tool_use_id"] == "tu_1"
    assert tool_result_block["content"] == "hi"
    assert tool_result_block["is_error"] is False


async def test_openai_compat_runner_end_to_end_through_orchestrator(
    make_agent: Callable[..., SubAgent],
) -> None:
    """Run a single user turn through `Orchestrator.run` against a real
    `OpenAICompatRunner` whose SDK boundary is a `FakeAsyncOpenAI`.

    Same shape as the Anthropic test but exercises the *OpenAI* runner's
    translation: harness `Message` -> OpenAI chat.completions request,
    and OpenAI tool_calls response -> harness assistant `Message` with
    `tool_use` block. The hook handler and dispatcher are the same real
    instances injected into the runner, so a `PreToolUse` firing proves
    the runner's own internal `self.hooks.emit(...)` ran.
    """
    # Canned OpenAI-shape responses: response 1 emits a function tool
    # call, response 2 emits the terminal text.
    tool_call = FakeOAToolCall(
        id="call_1",
        function=FakeOAFunction(name="echo", arguments=json.dumps({"text": "hi"})),
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
        choices=[
            FakeOAChoice(
                message=FakeOAMessage(content="echoed: hi"),
                finish_reason="stop",
            )
        ]
    )
    fake_client = FakeAsyncOpenAI(responses=[response_1, response_2])

    dispatcher, dispatch_log = _echo_dispatcher()

    hooks = HookRunner()
    pre_tool_calls: list[ToolCall] = []

    def _capture_pre_tool(event: PreToolUse) -> None:
        pre_tool_calls.append(event.call)

    hooks.register(PreToolUse, _capture_pre_tool)

    runner = OpenAICompatRunner(
        dispatcher,
        hooks,
        client=cast(Any, fake_client),
    )

    orchestrator = Orchestrator(dispatcher, hooks, runner)
    agent = make_agent(model="gpt-test", allowed_tools=["echo"])

    reply = await orchestrator.run(agent, [text("user", "please echo hi")])

    # Real dispatcher executed the real handler.
    assert dispatch_log == ["hi"]

    # `PreToolUse` fired on the *runner's* `HookRunner` (same instance
    # passed to the constructor); the captured call carries the args
    # decoded from the OpenAI tool_calls JSON.
    assert len(pre_tool_calls) == 1
    captured = pre_tool_calls[0]
    assert captured.name == "echo"
    assert captured.arguments == {"text": "hi"}
    assert captured.id == "call_1"

    # Terminal assistant message is the second response's text, after
    # the real runner's `_translate_out` walked the OpenAI choice.
    assert reply.role == "assistant"
    assert len(reply.content) == 1
    assert reply.content[0].type == "text"
    assert reply.content[0].text == "echoed: hi"

    # Two requests, one per iteration. The second carries the assistant
    # tool_calls turn (translated from the harness assistant Message)
    # plus a `role=tool` entry with the dispatched result — proving the
    # real OpenAICompatRunner's request-building ran, not a stand-in.
    assert len(fake_client.chat.completions.requests) == 2
    second_request = fake_client.chat.completions.requests[1]
    roles = [m["role"] for m in second_request["messages"]]
    assert "assistant" in roles
    assert "tool" in roles
    tool_msg = next(m for m in second_request["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "hi"
