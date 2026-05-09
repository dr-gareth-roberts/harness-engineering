"""Streaming runner + Orchestrator.run_stream tests (Wave 13a #9).

Pins event ordering, terminal MessageEnd, speculator-during-stream
interaction, and the TypeError raised by Orchestrator.run_stream when
the runner doesn't satisfy `StreamingRunner`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner, SessionEnd, SessionStart
from harness.prompts import Message, text
from harness.prompts.messages import ContentBlock
from harness.runner.anthropic import AnthropicRunner
from harness.streaming import (
    MessageEnd,
    StreamEvent,
    StreamingRunner,
    TextDelta,
    ToolUseEnd,
    ToolUseStart,
)
from harness.tools import Dispatcher, Tool
from harness.tools.schema import ToolCall, ToolResult
from tests.runner.fakes import (
    FakeAsyncAnthropic,
    FakeContentBlockStopEvent,
    FakeMessage,
    FakeTextBlock,
    FakeTextEvent,
    FakeToolUseBlock,
)


class _EchoIn(BaseModel):
    text: str


def _echo_dispatcher() -> tuple[Dispatcher, list[str]]:
    log: list[str] = []

    def echo(args: _EchoIn) -> str:
        log.append(args.text)
        return args.text

    return (
        Dispatcher(
            [Tool(name="echo", description="Echo it back.", input_model=_EchoIn, handler=echo)]
        ),
        log,
    )


def _agent() -> SubAgent:
    return SubAgent(
        name="t",
        system_prompt="",
        model="claude-test",
        allowed_tools=["echo"],
    )


# ---------------------------------------------------------------------------
# Anthropic run_stream — text-only path


async def test_anthropic_run_stream_yields_text_deltas_then_message_end() -> None:
    """A text-only response: the runner yields TextDelta per text-event,
    then exactly one terminal MessageEnd."""
    response = FakeMessage(
        content=[FakeTextBlock(text="hello world")],
        stop_reason="end_turn",
        events=[
            FakeTextEvent(text="hello "),
            FakeTextEvent(text="world"),
            FakeContentBlockStopEvent(index=0, content_block=FakeTextBlock(text="hello world")),
        ],
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    events: list[StreamEvent] = []
    async for event in runner.run_stream(_agent(), [text("user", "hi")]):
        events.append(event)

    # Expect: 2 TextDelta + 1 MessageEnd, in that order.
    assert [type(e).__name__ for e in events] == ["TextDelta", "TextDelta", "MessageEnd"]
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas == ["hello ", "world"]
    [end] = [e for e in events if isinstance(e, MessageEnd)]
    assert end.message.content[0].text == "hello world"


# ---------------------------------------------------------------------------
# Anthropic run_stream — tool-use path


async def test_anthropic_run_stream_yields_tool_use_start_and_end_around_dispatch() -> None:
    """For a tool-use block, run_stream yields ToolUseStart (pre-dispatch),
    runs the dispatcher, yields ToolUseEnd (with the result), then either
    continues the loop or yields MessageEnd."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(
        content=[tool_use],
        stop_reason="tool_use",
        events=[FakeContentBlockStopEvent(index=0, content_block=tool_use)],
    )
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="echoed")],
        stop_reason="end_turn",
        events=[FakeTextEvent(text="echoed")],
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    events: list[StreamEvent] = []
    async for event in runner.run_stream(_agent(), [text("user", "echo hi")]):
        events.append(event)

    # Order pinned: ToolUseStart → ToolUseEnd → TextDelta → MessageEnd.
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolUseStart", "ToolUseEnd", "TextDelta", "MessageEnd"]

    [start_event] = [e for e in events if isinstance(e, ToolUseStart)]
    [end_event] = [e for e in events if isinstance(e, ToolUseEnd)]
    assert start_event.call.id == "tu_1"
    assert start_event.call.name == "echo"
    assert end_event.call.id == "tu_1"
    assert end_event.result.content == "hi"
    assert end_event.result.is_error is False
    # The dispatcher's handler ran.
    assert dispatch_log == ["hi"]


# ---------------------------------------------------------------------------
# StreamingRunner Protocol structural check


def test_anthropic_runner_satisfies_streaming_runner_protocol() -> None:
    """`AnthropicRunner` must satisfy the runtime-checkable
    StreamingRunner Protocol so `isinstance(runner, StreamingRunner)`
    works in `Orchestrator.run_stream`."""
    runner = AnthropicRunner(
        Dispatcher(),
        HookRunner(),
        client=FakeAsyncAnthropic(responses=[]),  # type: ignore[arg-type]
    )
    assert isinstance(runner, StreamingRunner)


def test_callable_runner_does_not_satisfy_streaming_runner_protocol() -> None:
    """A plain `Callable[..., Awaitable[Message]]` runner without
    `run_stream` must NOT satisfy the StreamingRunner Protocol — that's
    the gate `Orchestrator.run_stream` uses to raise TypeError."""

    async def callable_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", "hi")

    assert not isinstance(callable_runner, StreamingRunner)


# ---------------------------------------------------------------------------
# Orchestrator.run_stream — TypeError on non-streaming runner


async def test_orchestrator_run_stream_raises_typeerror_for_non_streaming_runner() -> None:
    async def callable_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", "hi")

    orch = Orchestrator(Dispatcher(), HookRunner(), callable_runner)
    agent = SubAgent(name="t", system_prompt="", model="m")

    # Generator pattern: TypeError fires when we try to consume the
    # first event. asynchronous generator functions defer the body
    # until the first `__anext__`, so we have to iterate to surface.
    iterator = orch.run_stream(agent, [text("user", "hi")])
    try:
        async for _event in iterator:
            pass
    except TypeError as exc:
        assert "StreamingRunner" in str(exc)
    else:
        raise AssertionError("expected TypeError")


# ---------------------------------------------------------------------------
# Orchestrator.run_stream — full delegation + lifecycle hooks


async def test_orchestrator_run_stream_emits_session_hooks_around_runner_stream() -> None:
    """SessionStart fires before the first stream event; SessionEnd
    fires after the last (or after exception). MessageEnd is the
    runner's terminal event; SessionEnd is the orchestrator's."""
    response = FakeMessage(
        content=[FakeTextBlock(text="ok")],
        stop_reason="end_turn",
        events=[FakeTextEvent(text="ok")],
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    seen: list[str] = []
    hooks = HookRunner()
    hooks.register(SessionStart, lambda e: seen.append("SessionStart"))
    hooks.register(SessionEnd, lambda e: seen.append("SessionEnd"))

    orch = Orchestrator(dispatcher, hooks, runner)
    events: list[StreamEvent] = []
    async for event in orch.run_stream(_agent(), [text("user", "hi")]):
        seen.append(type(event).__name__)
        events.append(event)

    # SessionStart fires before any stream event; SessionEnd fires
    # after MessageEnd (the runner's terminal).
    assert seen[0] == "SessionStart"
    assert seen[-1] == "SessionEnd"
    assert "MessageEnd" in seen
    assert seen.index("MessageEnd") < seen.index("SessionEnd")


async def test_orchestrator_run_stream_propagates_runner_events_in_order() -> None:
    """Round-trip: every event the runner yields must reach the
    orchestrator caller in the same order."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(
        content=[FakeTextBlock(text="thinking..."), tool_use],
        stop_reason="tool_use",
        events=[
            FakeTextEvent(text="thinking..."),
            FakeContentBlockStopEvent(index=0, content_block=FakeTextBlock(text="thinking...")),
            FakeContentBlockStopEvent(index=1, content_block=tool_use),
        ],
    )
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="done")],
        stop_reason="end_turn",
        events=[FakeTextEvent(text="done")],
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    orch = Orchestrator(dispatcher, HookRunner(), runner)

    kinds: list[str] = []
    async for event in orch.run_stream(_agent(), [text("user", "echo hi")]):
        kinds.append(type(event).__name__)

    # Note: the FakeContentBlockStopEvent for a text block doesn't
    # produce a TextDelta on its own — only FakeTextEvent does. So the
    # order is: TextDelta("thinking...") → ToolUseStart → ToolUseEnd →
    # TextDelta("done") → MessageEnd.
    assert kinds == ["TextDelta", "ToolUseStart", "ToolUseEnd", "TextDelta", "MessageEnd"]


# ---------------------------------------------------------------------------
# Speculator-during-stream interaction


class _StubSpeculator:
    """Minimal SpeculatorProtocol stub that just records calls."""

    def __init__(self, hits: dict[str, ToolResult] | None = None) -> None:
        self.hits = dict(hits or {})
        self.begin_calls = 0
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
        self.begin_calls += 1

    async def observe(self, call: ToolCall) -> None:
        self.observe_calls.append(call)

    async def cancel_unobserved(self) -> None:
        self.cancel_unobserved_calls += 1

    async def try_resolve(self, call: ToolCall) -> ToolResult | None:
        self.try_resolve_calls.append(call)
        return self.hits.get(call.name)

    async def end(self) -> None:
        self.end_calls += 1


async def test_speculator_lifecycle_during_run_stream() -> None:
    """The full speculator lifecycle (begin / observe / cancel_unobserved
    / try_resolve / end) must fire in run_stream just as it does in
    __call__. This is the targeted speculator-during-stream test the
    advisor flagged before declaring done."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(
        content=[tool_use],
        stop_reason="tool_use",
        events=[FakeContentBlockStopEvent(index=0, content_block=tool_use)],
    )
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="done")],
        stop_reason="end_turn",
        events=[FakeTextEvent(text="done")],
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()
    spec = _StubSpeculator()  # no hits configured -> always miss

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )

    events: list[StreamEvent] = []
    async for event in runner.run_stream(_agent(), [text("user", "echo hi")]):
        events.append(event)

    # Speculator API surface, per iteration:
    # iter 1: begin → observe(echo) → cancel_unobserved → try_resolve(echo) → end
    # iter 2: begin → cancel_unobserved (no observe — text only) → end
    assert spec.begin_calls == 2
    assert spec.cancel_unobserved_calls == 2
    assert spec.end_calls == 2
    assert [c.id for c in spec.observe_calls] == ["tu_1"]
    assert [c.id for c in spec.try_resolve_calls] == ["tu_1"]
    # Miss → runner's own dispatch ran.
    assert dispatch_log == ["hi"]


async def test_speculator_hit_short_circuits_dispatch_in_run_stream() -> None:
    """When the speculator returns a hit on try_resolve, the runner's
    own dispatch is skipped — but the ToolUseEnd event still fires
    (with the speculation's result)."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(
        content=[tool_use],
        stop_reason="tool_use",
        events=[FakeContentBlockStopEvent(index=0, content_block=tool_use)],
    )
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="done")],
        stop_reason="end_turn",
        events=[FakeTextEvent(text="done")],
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()
    cached = ToolResult(id="tu_1", content="(speculation)", is_error=False)
    spec = _StubSpeculator(hits={"echo": cached})

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )

    events: list[StreamEvent] = []
    async for event in runner.run_stream(_agent(), [text("user", "echo hi")]):
        events.append(event)

    [end_event] = [e for e in events if isinstance(e, ToolUseEnd)]
    assert end_event.result.content == "(speculation)"
    # Dispatcher's own handler did NOT run — speculation provided the result.
    assert dispatch_log == []


# ---------------------------------------------------------------------------
# MessageEnd uniqueness


async def test_message_end_fires_exactly_once_at_terminal() -> None:
    """Across a multi-iteration tool-use run, MessageEnd must fire
    exactly once — at the very end, not after each iteration."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(
        content=[tool_use],
        stop_reason="tool_use",
        events=[FakeContentBlockStopEvent(index=0, content_block=tool_use)],
    )
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="done")],
        stop_reason="end_turn",
        events=[FakeTextEvent(text="done")],
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    events: list[StreamEvent] = []
    async for event in runner.run_stream(_agent(), [text("user", "echo hi")]):
        events.append(event)

    message_ends = [e for e in events if isinstance(e, MessageEnd)]
    assert len(message_ends) == 1
    assert events[-1] is message_ends[0]


# ---------------------------------------------------------------------------
# Sanity: AnthropicRunner.__call__ still works (no regression)


async def test_anthropic_call_still_works_after_run_stream_addition() -> None:
    """The non-streaming `__call__` path is the one Wave 6/10/etc
    pinned with 50+ tests. Run a basic case to confirm we didn't
    regress it while adding `run_stream`."""
    response = FakeMessage(
        content=[FakeTextBlock(text="hello")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    result = await runner(_agent(), [text("user", "hi")])
    assert isinstance(result, Message)
    assert result.content[0].text == "hello"


# Helper exists to make ContentBlock importable for tests that need to
# build messages with image blocks (none here yet, but kept for parity
# with other test modules).
def _content_block_factory() -> AsyncIterator[ContentBlock]:
    raise NotImplementedError
