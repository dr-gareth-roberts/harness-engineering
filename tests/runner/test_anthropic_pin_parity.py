"""Behavior-pinning parity tests for `AnthropicRunner.__call__` vs `run_stream`.

M3.1a (test-first phase of M3.1 in `audit/RELEASE-TODO.md`). The runner's
two ~150-line tool-use loops (`__call__` :286-471 and `run_stream`
:502-666 in `src/harness/runner/anthropic.py`) currently duplicate the
tool-loop / speculator / hook / cache-cap / pause-refusal logic. M3.1b
will deduplicate; this file pins the **exact observable contract** of
both paths so a future extraction cannot silently drift them.

Parity collapse model
=====================

For every scenario, both surfaces are driven against the **same**
fake-vendor inputs (one fresh `FakeAsyncAnthropic` per path — the fake
pops responses on each `stream()` call). Outputs are compared via a
`collapse_stream` helper that locates the terminal `MessageEnd` and
returns its `Message`. Equality then uses Pydantic structural `==` on
`Message` — no bespoke comparator.

The tests deliberately observe **only** what a future refactor cannot
hide:

- The `Message` returned by `__call__`.
- The sequence of `StreamEvent`s yielded by `run_stream`.
- Hook events fired into a `HookRunner` (lists of captured event
  instances).
- Speculator method-call shapes (counts + ordered args).
- `PrefixWatcher.fingerprint` call ordering and the dict shapes
  passed in.

No internal state of either surface is inspected. The pin survives any
refactor that preserves the contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from harness.agents import SubAgent
from harness.hooks import HookRunner, PauseTurn, PostAssistantMessage, Refusal
from harness.prompts import Message, text
from harness.prompts.messages import ContentBlock
from harness.runner.anthropic import (
    AnthropicRunner,
    CacheBreakpointLimitExceeded,
)
from harness.streaming import (
    MessageEnd,
    StreamEvent,
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

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


class _EchoIn(BaseModel):
    text: str


def _echo_dispatcher() -> tuple[Dispatcher, list[str]]:
    """A dispatcher with a single `echo` tool plus an execution log.

    The log is the side channel used to assert dispatch happened (or
    didn't) along each path independently.
    """
    log: list[str] = []

    def echo(args: _EchoIn) -> str:
        log.append(args.text)
        return args.text

    return (
        Dispatcher(
            [
                Tool(
                    name="echo",
                    description="Echo it back.",
                    input_model=_EchoIn,
                    handler=echo,
                    idempotent=True,
                )
            ]
        ),
        log,
    )


def _agent() -> SubAgent:
    return SubAgent(
        name="t",
        system_prompt="You are a small test agent.",
        model="claude-test",
        allowed_tools=["echo"],
    )


def _collapse_stream(events: list[StreamEvent]) -> Message:
    """Reduce a stream-event sequence to the `Message` the caller sees.

    `run_stream` always yields exactly one terminal `MessageEnd` per
    invocation (pinned in `tests/runner/test_streaming.py`); its
    `message` field is the equivalent of `__call__`'s return value.
    """
    ends = [e for e in events if isinstance(e, MessageEnd)]
    assert len(ends) == 1, f"expected exactly one MessageEnd, got {len(ends)}"
    return ends[0].message


async def _drive_stream(
    runner: AnthropicRunner,
    agent: SubAgent,
    messages: list[Message],
) -> list[StreamEvent]:
    """Drain `run_stream` into a list. Kept as a one-liner helper so
    parity tests stay free of `async for` boilerplate."""
    events: list[StreamEvent] = []
    async for event in runner.run_stream(agent, messages):
        events.append(event)
    return events


class _StubPrefixWatcher:
    """Records every `fingerprint(request)` call.

    Snapshots a stable view of the request — message count + role
    sequence + per-message block-type sequence — because the runner
    mutates the same `request` dict in place across iterations. Storing
    references would collapse all snapshots to the post-loop state.
    """

    def __init__(self) -> None:
        self.fingerprint_calls: list[dict[str, Any]] = []

    async def fingerprint(self, request: dict[str, Any]) -> None:
        snapshot: dict[str, Any] = {
            "message_count": len(request.get("messages", [])),
            "roles": [m["role"] for m in request.get("messages", [])],
            "block_types": [
                [b.get("type") for b in (m.get("content") or []) if isinstance(b, dict)]
                for m in request.get("messages", [])
            ],
        }
        self.fingerprint_calls.append(snapshot)


class _StubSpeculator:
    """Test stub satisfying `SpeculatorProtocol` — records every call.

    Configurable per-call hit/miss via `hits` (name -> ToolResult). The
    same shape used in `tests/runner/test_anthropic.py`; redeclared here
    so this file stays standalone (parity tests should not depend on
    the order other test files load).
    """

    def __init__(self, hits: dict[str, ToolResult] | None = None) -> None:
        self.hits = dict(hits or {})
        self.begin_calls: list[int] = []  # history length per call
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
        self.begin_calls.append(len(history))

    async def observe(self, call: ToolCall) -> None:
        self.observe_calls.append(call)

    async def cancel_unobserved(self) -> None:
        self.cancel_unobserved_calls += 1

    async def try_resolve(self, call: ToolCall) -> ToolResult | None:
        self.try_resolve_calls.append(call)
        return self.hits.get(call.name)

    async def end(self) -> None:
        self.end_calls += 1


# ---------------------------------------------------------------------------
# Scenario 1: single text response (no tool use)
# ---------------------------------------------------------------------------


def _scenario_text_only_responses() -> list[FakeMessage]:
    """One assistant turn: text-only, end_turn. Both surfaces should
    produce a single-block assistant `Message` carrying `"hello world"`.

    `events` is scripted explicitly so `run_stream` emits `TextDelta`s
    (the auto-derived default produces only `content_block_stop`
    events, which `run_stream` does not turn into deltas)."""
    block = FakeTextBlock(text="hello world")
    return [
        FakeMessage(
            content=[block],
            stop_reason="end_turn",
            events=[
                FakeTextEvent(text="hello "),
                FakeTextEvent(text="world"),
                FakeContentBlockStopEvent(index=0, content_block=block),
            ],
        )
    ]


async def test_parity_single_text_response_returns_same_message() -> None:
    """Pin: `__call__`'s `Message` equals the `MessageEnd.message` from
    `run_stream`. Concatenated `TextDelta.text` equals the assistant's
    final text."""
    dispatcher, _ = _echo_dispatcher()

    call_client = FakeAsyncAnthropic(responses=_scenario_text_only_responses())
    call_runner = AnthropicRunner(dispatcher, HookRunner(), client=call_client)  # type: ignore[arg-type]
    call_result = await call_runner(_agent(), [text("user", "hi")])

    stream_client = FakeAsyncAnthropic(responses=_scenario_text_only_responses())
    stream_runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=stream_client,  # type: ignore[arg-type]
    )
    stream_events = await _drive_stream(stream_runner, _agent(), [text("user", "hi")])

    # Path-equivalence: collapsed stream message == direct call message.
    assert call_result == _collapse_stream(stream_events)

    # The text-delta concatenation must agree with the final text.
    deltas = "".join(e.text for e in stream_events if isinstance(e, TextDelta))
    final_text = "".join(b.text or "" for b in call_result.content if b.type == "text")
    assert deltas == final_text == "hello world"


# ---------------------------------------------------------------------------
# Scenario 2: single tool_use, then a wrap-up text response
# ---------------------------------------------------------------------------


def _scenario_single_tool_use_responses() -> list[FakeMessage]:
    """Two SDK turns:
      1. assistant emits `tool_use(echo, text="hi")` with stop_reason=tool_use
      2. assistant emits text "echoed: hi" with stop_reason=end_turn

    The runner's loop dispatches `echo` between the two turns; both
    surfaces must arrive at the same final assistant message."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    wrap_up = FakeTextBlock(text="echoed: hi")
    return [
        FakeMessage(
            content=[tool_use],
            stop_reason="tool_use",
            events=[FakeContentBlockStopEvent(index=0, content_block=tool_use)],
        ),
        FakeMessage(
            content=[wrap_up],
            stop_reason="end_turn",
            events=[
                FakeTextEvent(text="echoed: hi"),
                FakeContentBlockStopEvent(index=0, content_block=wrap_up),
            ],
        ),
    ]


async def test_parity_single_tool_use_produces_same_final_message() -> None:
    dispatcher_call, log_call = _echo_dispatcher()
    dispatcher_stream, log_stream = _echo_dispatcher()

    call_client = FakeAsyncAnthropic(responses=_scenario_single_tool_use_responses())
    call_runner = AnthropicRunner(
        dispatcher_call,
        HookRunner(),
        client=call_client,  # type: ignore[arg-type]
    )
    call_result = await call_runner(_agent(), [text("user", "echo hi")])

    stream_client = FakeAsyncAnthropic(responses=_scenario_single_tool_use_responses())
    stream_runner = AnthropicRunner(
        dispatcher_stream,
        HookRunner(),
        client=stream_client,  # type: ignore[arg-type]
    )
    stream_events = await _drive_stream(stream_runner, _agent(), [text("user", "echo hi")])

    # Final message identical between paths.
    assert call_result == _collapse_stream(stream_events)

    # Both paths actually ran the echo handler.
    assert log_call == ["hi"]
    assert log_stream == ["hi"]

    # The tool_use call surfaces in the stream as ToolUseStart / ToolUseEnd
    # carrying the same id the model emitted.
    [start] = [e for e in stream_events if isinstance(e, ToolUseStart)]
    [end] = [e for e in stream_events if isinstance(e, ToolUseEnd)]
    assert start.call.id == "tu_1"
    assert end.call.id == "tu_1"
    assert end.result.content == "hi"
    assert end.result.is_error is False


# ---------------------------------------------------------------------------
# Scenario 3: multiple tool_uses in a single assistant turn
# ---------------------------------------------------------------------------


def _scenario_multi_tool_use_responses() -> list[FakeMessage]:
    """First assistant turn emits THREE tool_use blocks in one message
    (Anthropic supports parallel tool use). Both surfaces must dispatch
    all three, in declared order, and arrive at the same final wrap-up.
    """
    tu_1 = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "a"})
    tu_2 = FakeToolUseBlock(id="tu_2", name="echo", input={"text": "b"})
    tu_3 = FakeToolUseBlock(id="tu_3", name="echo", input={"text": "c"})
    wrap_up = FakeTextBlock(text="done")
    return [
        FakeMessage(
            content=[tu_1, tu_2, tu_3],
            stop_reason="tool_use",
            events=[
                FakeContentBlockStopEvent(index=0, content_block=tu_1),
                FakeContentBlockStopEvent(index=1, content_block=tu_2),
                FakeContentBlockStopEvent(index=2, content_block=tu_3),
            ],
        ),
        FakeMessage(
            content=[wrap_up],
            stop_reason="end_turn",
            events=[
                FakeTextEvent(text="done"),
                FakeContentBlockStopEvent(index=0, content_block=wrap_up),
            ],
        ),
    ]


async def test_parity_multiple_tool_uses_dispatch_in_order() -> None:
    dispatcher_call, log_call = _echo_dispatcher()
    dispatcher_stream, log_stream = _echo_dispatcher()

    call_client = FakeAsyncAnthropic(responses=_scenario_multi_tool_use_responses())
    call_runner = AnthropicRunner(
        dispatcher_call,
        HookRunner(),
        client=call_client,  # type: ignore[arg-type]
    )
    call_result = await call_runner(_agent(), [text("user", "echo three times")])

    stream_client = FakeAsyncAnthropic(responses=_scenario_multi_tool_use_responses())
    stream_runner = AnthropicRunner(
        dispatcher_stream,
        HookRunner(),
        client=stream_client,  # type: ignore[arg-type]
    )
    stream_events = await _drive_stream(stream_runner, _agent(), [text("user", "echo three times")])

    # Final message identical.
    assert call_result == _collapse_stream(stream_events)

    # All three calls dispatched on both paths, in the model's emit
    # order — pinned by both the dispatcher's side-channel log and the
    # stream's ToolUseStart/End pair sequence.
    assert log_call == ["a", "b", "c"]
    assert log_stream == ["a", "b", "c"]

    starts = [e for e in stream_events if isinstance(e, ToolUseStart)]
    ends = [e for e in stream_events if isinstance(e, ToolUseEnd)]
    assert [s.call.id for s in starts] == ["tu_1", "tu_2", "tu_3"]
    assert [e.call.id for e in ends] == ["tu_1", "tu_2", "tu_3"]


# ---------------------------------------------------------------------------
# Scenario 4: pause_turn handling
# ---------------------------------------------------------------------------


def _scenario_pause_turn_responses() -> list[FakeMessage]:
    """One assistant turn ending in `pause_turn` (server-side budget
    exhaustion). The runner returns the partial message + fires
    `PauseTurn` on both surfaces.

    Per the M3.1 audit note, this is "where the two paths most subtly
    differ" — we pin equality so any future drift fails loudly."""
    block = FakeTextBlock(text="working on it")
    return [
        FakeMessage(
            content=[block],
            stop_reason="pause_turn",
            events=[
                FakeTextEvent(text="working on it"),
                FakeContentBlockStopEvent(index=0, content_block=block),
            ],
        )
    ]


async def test_parity_pause_turn_fires_event_and_returns_partial_message() -> None:
    dispatcher_call, _ = _echo_dispatcher()
    dispatcher_stream, _ = _echo_dispatcher()

    pause_events_call: list[PauseTurn] = []
    hooks_call = HookRunner()
    hooks_call.register(PauseTurn, lambda e: pause_events_call.append(e))

    pause_events_stream: list[PauseTurn] = []
    hooks_stream = HookRunner()
    hooks_stream.register(PauseTurn, lambda e: pause_events_stream.append(e))

    call_client = FakeAsyncAnthropic(responses=_scenario_pause_turn_responses())
    call_runner = AnthropicRunner(
        dispatcher_call,
        hooks_call,
        client=call_client,  # type: ignore[arg-type]
    )
    call_result = await call_runner(_agent(), [text("user", "do a long thing")])

    stream_client = FakeAsyncAnthropic(responses=_scenario_pause_turn_responses())
    stream_runner = AnthropicRunner(
        dispatcher_stream,
        hooks_stream,
        client=stream_client,  # type: ignore[arg-type]
    )
    stream_events = await _drive_stream(stream_runner, _agent(), [text("user", "do a long thing")])

    # Final assistant message is identical.
    assert call_result == _collapse_stream(stream_events)

    # Both paths fire exactly one PauseTurn event, with matching message
    # and reason. Pydantic equality covers structural compare.
    assert len(pause_events_call) == 1
    assert len(pause_events_stream) == 1
    assert pause_events_call[0].message == pause_events_stream[0].message
    assert pause_events_call[0].reason == pause_events_stream[0].reason == "pause_turn"

    # Stream's MessageEnd is the terminal event: no further events
    # (no extra TextDelta, no second iteration) follow it.
    assert isinstance(stream_events[-1], MessageEnd)


# ---------------------------------------------------------------------------
# Scenario 5: refusal handling
# ---------------------------------------------------------------------------


def _scenario_refusal_responses() -> list[FakeMessage]:
    """One assistant turn with `refusal` stop_reason. Runner returns
    the refusal-only message + fires `Refusal` on both surfaces.

    Anthropic's refusal stop usually carries a textual explanation;
    pin that the text passes through unchanged on both paths."""
    block = FakeTextBlock(text="I can't help with that.")
    return [
        FakeMessage(
            content=[block],
            stop_reason="refusal",
            events=[
                FakeTextEvent(text="I can't help with that."),
                FakeContentBlockStopEvent(index=0, content_block=block),
            ],
        )
    ]


async def test_parity_refusal_fires_event_and_returns_refusal_message() -> None:
    dispatcher_call, _ = _echo_dispatcher()
    dispatcher_stream, _ = _echo_dispatcher()

    refusal_events_call: list[Refusal] = []
    hooks_call = HookRunner()
    hooks_call.register(Refusal, lambda e: refusal_events_call.append(e))

    refusal_events_stream: list[Refusal] = []
    hooks_stream = HookRunner()
    hooks_stream.register(Refusal, lambda e: refusal_events_stream.append(e))

    call_client = FakeAsyncAnthropic(responses=_scenario_refusal_responses())
    call_runner = AnthropicRunner(
        dispatcher_call,
        hooks_call,
        client=call_client,  # type: ignore[arg-type]
    )
    call_result = await call_runner(_agent(), [text("user", "do a thing")])

    stream_client = FakeAsyncAnthropic(responses=_scenario_refusal_responses())
    stream_runner = AnthropicRunner(
        dispatcher_stream,
        hooks_stream,
        client=stream_client,  # type: ignore[arg-type]
    )
    stream_events = await _drive_stream(stream_runner, _agent(), [text("user", "do a thing")])

    # Final messages match.
    assert call_result == _collapse_stream(stream_events)

    # Both paths fire exactly one Refusal event with the same message
    # body.
    assert len(refusal_events_call) == 1
    assert len(refusal_events_stream) == 1
    assert refusal_events_call[0].message == refusal_events_stream[0].message

    # Refusal text passes through verbatim.
    assert "can't help" in (call_result.content[0].text or "")


# ---------------------------------------------------------------------------
# Scenario 6: Speculator hit pre-empts dispatch on both surfaces
# ---------------------------------------------------------------------------


async def test_parity_speculator_hit_uses_speculative_result_on_both_paths() -> None:
    """A speculator pre-resolves the upcoming `echo` call. Both paths
    must consume the speculative `ToolResult` and skip their own
    dispatch + hook cycle for that call.

    Pin the observable agreement:
    - both paths produce the same final assistant message;
    - both paths leave the dispatcher's handler unrun;
    - both paths consult `try_resolve` exactly once per tool_use block;
    - both paths fire begin/end one per iteration (here: 2).
    """
    dispatcher_call, log_call = _echo_dispatcher()
    dispatcher_stream, log_stream = _echo_dispatcher()

    cached = ToolResult(id="tu_1", content="(from speculation)", is_error=False)
    spec_call = _StubSpeculator(hits={"echo": cached})
    spec_stream = _StubSpeculator(hits={"echo": cached})

    call_client = FakeAsyncAnthropic(responses=_scenario_single_tool_use_responses())
    call_runner = AnthropicRunner(
        dispatcher_call,
        HookRunner(),
        client=call_client,  # type: ignore[arg-type]
        speculator=spec_call,
    )
    call_result = await call_runner(_agent(), [text("user", "echo hi")])

    stream_client = FakeAsyncAnthropic(responses=_scenario_single_tool_use_responses())
    stream_runner = AnthropicRunner(
        dispatcher_stream,
        HookRunner(),
        client=stream_client,  # type: ignore[arg-type]
        speculator=spec_stream,
    )
    stream_events = await _drive_stream(stream_runner, _agent(), [text("user", "echo hi")])

    # Final messages agree.
    assert call_result == _collapse_stream(stream_events)

    # Neither path touched the real dispatcher for the speculative call.
    assert log_call == []
    assert log_stream == []

    # Speculator surface called identically on both paths: one
    # try_resolve per tool_use block, two begin/end pairs across the
    # two iterations (tool_use iter + wrap-up iter).
    assert [c.id for c in spec_call.try_resolve_calls] == ["tu_1"]
    assert [c.id for c in spec_stream.try_resolve_calls] == ["tu_1"]
    assert len(spec_call.begin_calls) == len(spec_stream.begin_calls) == 2
    assert spec_call.end_calls == spec_stream.end_calls == 2

    # The stream emits a ToolUseEnd carrying the speculative result —
    # not the dispatcher's output.
    [end_event] = [e for e in stream_events if isinstance(e, ToolUseEnd)]
    assert end_event.result.content == "(from speculation)"


# ---------------------------------------------------------------------------
# Scenario 7: PrefixWatcher fingerprint parity
# ---------------------------------------------------------------------------


async def test_parity_prefix_watcher_fingerprint_calls_match_in_shape_and_order() -> None:
    """Both surfaces call `watcher.fingerprint(request)` once per
    iteration, on identical request shapes, in the same order.

    Pin the per-call snapshot (role + block-type structure) — if the
    extracted state machine accidentally fingerprints at a different
    point in the loop (e.g. after instead of before the mutation), the
    snapshots diverge."""
    dispatcher_call, _ = _echo_dispatcher()
    dispatcher_stream, _ = _echo_dispatcher()

    watcher_call = _StubPrefixWatcher()
    watcher_stream = _StubPrefixWatcher()

    call_client = FakeAsyncAnthropic(responses=_scenario_single_tool_use_responses())
    call_runner = AnthropicRunner(
        dispatcher_call,
        HookRunner(),
        client=call_client,  # type: ignore[arg-type]
        prefix_watcher=watcher_call,
    )
    call_result = await call_runner(_agent(), [text("user", "echo hi")])

    stream_client = FakeAsyncAnthropic(responses=_scenario_single_tool_use_responses())
    stream_runner = AnthropicRunner(
        dispatcher_stream,
        HookRunner(),
        client=stream_client,  # type: ignore[arg-type]
        prefix_watcher=watcher_stream,
    )
    stream_events = await _drive_stream(stream_runner, _agent(), [text("user", "echo hi")])

    # Final messages match.
    assert call_result == _collapse_stream(stream_events)

    # Same number of fingerprint calls (one per iteration; two
    # iterations: the tool_use turn + the wrap-up turn).
    assert len(watcher_call.fingerprint_calls) == 2
    assert len(watcher_stream.fingerprint_calls) == 2

    # Per-call shape agreement: iteration N from `__call__` must match
    # iteration N from `run_stream`.
    assert watcher_call.fingerprint_calls == watcher_stream.fingerprint_calls

    # And the iteration progression is sensible: 1 user message, then
    # 3 (user, assistant, user-tool-result).
    assert watcher_call.fingerprint_calls[0]["message_count"] == 1
    assert watcher_call.fingerprint_calls[1]["message_count"] == 3
    assert watcher_call.fingerprint_calls[1]["roles"] == ["user", "assistant", "user"]

    # Sanity: the stream actually completed (collapse already asserts a
    # single MessageEnd, but anchor that here too).
    assert any(isinstance(e, MessageEnd) for e in stream_events)


# ---------------------------------------------------------------------------
# Scenario 8: cache-breakpoint limit raised before any SDK IO
# ---------------------------------------------------------------------------


async def test_parity_cache_breakpoint_limit_raises_same_error_on_both_paths() -> None:
    """Five `cache_control` markers exceed the documented limit (4).
    `__call__` raises directly; `run_stream` raises on the first
    `__anext__` (async-generator semantics — body deferred until
    iteration begins). The message must be identical.

    Pin that neither path makes an SDK call: the fake's
    `messages.requests` log stays empty on both."""
    dispatcher_call, _ = _echo_dispatcher()
    dispatcher_stream, _ = _echo_dispatcher()

    five_cached_messages = [
        Message(
            role="user",
            content=[ContentBlock(type="text", text=f"chunk-{i}", cache=True)],
        )
        for i in range(5)
    ]

    call_client = FakeAsyncAnthropic(
        responses=[FakeMessage(content=[FakeTextBlock(text="x")], stop_reason="end_turn")]
    )
    call_runner = AnthropicRunner(
        dispatcher_call,
        HookRunner(),
        client=call_client,  # type: ignore[arg-type]
    )
    with pytest.raises(CacheBreakpointLimitExceeded) as call_exc:
        await call_runner(_agent(), five_cached_messages)

    stream_client = FakeAsyncAnthropic(
        responses=[FakeMessage(content=[FakeTextBlock(text="x")], stop_reason="end_turn")]
    )
    stream_runner = AnthropicRunner(
        dispatcher_stream,
        HookRunner(),
        client=stream_client,  # type: ignore[arg-type]
    )
    with pytest.raises(CacheBreakpointLimitExceeded) as stream_exc:
        async for _event in stream_runner.run_stream(_agent(), five_cached_messages):
            pass

    # Same typed exception with the same message on both paths.
    assert str(call_exc.value) == str(stream_exc.value)
    assert "caps cache breakpoints at 4" in str(call_exc.value)

    # And neither path made an SDK call — the runner short-circuited
    # before stream().
    assert call_client.messages.requests == []
    assert stream_client.messages.requests == []


# ---------------------------------------------------------------------------
# Cross-cutting: PostAssistantMessage event parity across iterations
# ---------------------------------------------------------------------------


async def test_parity_post_assistant_message_fires_per_iteration_on_both_paths() -> None:
    """The runner fires `PostAssistantMessage` once per iteration on
    both surfaces. A two-iteration tool-use loop yields two events,
    structurally identical between paths.

    Not in the original 8 scenarios but pins the audit-cited "subtle
    differences" risk class — emission ordering around the loop body
    is exactly where a generator extraction could drift."""
    dispatcher_call, _ = _echo_dispatcher()
    dispatcher_stream, _ = _echo_dispatcher()

    events_call: list[Message] = []
    hooks_call = HookRunner()
    hooks_call.register(PostAssistantMessage, lambda e: events_call.append(e.message))

    events_stream: list[Message] = []
    hooks_stream = HookRunner()
    hooks_stream.register(PostAssistantMessage, lambda e: events_stream.append(e.message))

    call_client = FakeAsyncAnthropic(responses=_scenario_single_tool_use_responses())
    call_runner = AnthropicRunner(
        dispatcher_call,
        hooks_call,
        client=call_client,  # type: ignore[arg-type]
    )
    await call_runner(_agent(), [text("user", "echo hi")])

    stream_client = FakeAsyncAnthropic(responses=_scenario_single_tool_use_responses())
    stream_runner = AnthropicRunner(
        dispatcher_stream,
        hooks_stream,
        client=stream_client,  # type: ignore[arg-type]
    )
    await _drive_stream(stream_runner, _agent(), [text("user", "echo hi")])

    # Same count, structurally identical messages.
    assert len(events_call) == len(events_stream) == 2
    assert events_call == events_stream
