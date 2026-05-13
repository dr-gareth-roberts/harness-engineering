"""Tests for `Telemetry.session_scope` / `span_scope` correlation IDs.

These pin the contract described in the recorder's module docstring:
events emitted inside a `session_scope` get the trace_id; events
emitted inside a nested `span_scope` get a fresh span_id and the
previous span as parent_span_id; events emitted *outside* any scope
keep their default `None` IDs.

Wave 11 #11 — load-bearing for the Wave 11 #10 OTel span hierarchy
work that consumes these IDs.
"""

from __future__ import annotations

import asyncio

from harness.telemetry import (
    MemorySink,
    OrchestratorTurn,
    Telemetry,
    ToolDispatched,
)


def _tool_event() -> ToolDispatched:
    return ToolDispatched(
        tool_name="echo",
        call_id="c1",
        arguments={"x": 1},
        is_error=False,
        duration_ms=2.5,
    )


# ---------------------------------------------------------------------------
# Outside any scope


async def test_emit_without_scope_leaves_correlation_ids_none() -> None:
    sink = MemorySink()
    t = Telemetry(sink)
    await t.emit(_tool_event())

    [evt] = sink.events
    assert evt.trace_id is None
    assert evt.span_id is None
    assert evt.parent_span_id is None


# ---------------------------------------------------------------------------
# session_scope


async def test_session_scope_attaches_trace_id_to_emitted_events() -> None:
    sink = MemorySink()
    t = Telemetry(sink)

    async with t.session_scope():
        await t.emit(_tool_event())
        await t.emit(_tool_event())

    e1, e2 = sink.events
    assert e1.trace_id is not None
    assert e1.trace_id == e2.trace_id  # same scope → same trace
    assert e1.span_id is None  # no span_scope opened
    assert e1.parent_span_id is None


async def test_session_scope_accepts_caller_supplied_trace_id() -> None:
    """A caller can pass `trace_id=` to propagate an upstream system's
    trace context (e.g., a request-trace header)."""
    sink = MemorySink()
    t = Telemetry(sink)
    given = "upstream-trace-abc123"

    async with t.session_scope(trace_id=given) as yielded:
        assert yielded == given
        await t.emit(_tool_event())

    [evt] = sink.events
    assert evt.trace_id == given


async def test_session_scope_resets_after_exit() -> None:
    sink = MemorySink()
    t = Telemetry(sink)

    async with t.session_scope():
        await t.emit(_tool_event())
    # After scope exit, emissions should NOT carry the trace_id.
    await t.emit(_tool_event())

    inside, outside = sink.events
    assert inside.trace_id is not None
    assert outside.trace_id is None


# ---------------------------------------------------------------------------
# span_scope


async def test_span_scope_attaches_span_id_inherits_trace_id() -> None:
    sink = MemorySink()
    t = Telemetry(sink)

    async with t.session_scope(), t.span_scope():
        await t.emit(_tool_event())

    [evt] = sink.events
    assert evt.trace_id is not None
    assert evt.span_id is not None
    # Top-level span has no parent.
    assert evt.parent_span_id is None


async def test_nested_span_scope_records_parent() -> None:
    sink = MemorySink()
    t = Telemetry(sink)

    async with t.session_scope(), t.span_scope() as outer_span:
        await t.emit(_tool_event())  # emit at outer
        async with t.span_scope() as inner_span:
            await t.emit(_tool_event())  # emit at inner

    outer_evt, inner_evt = sink.events
    assert outer_evt.span_id == outer_span
    assert outer_evt.parent_span_id is None
    assert inner_evt.span_id == inner_span
    assert inner_evt.parent_span_id == outer_span
    # Both share the same trace.
    assert outer_evt.trace_id == inner_evt.trace_id


async def test_concurrent_span_scopes_in_sibling_tasks_do_not_collide() -> None:
    """Concurrent dispatches (tool calls running in parallel) must each
    get their own span_id without clobbering each other's `contextvars`.
    asyncio.create_task copies the context per spec."""
    sink = MemorySink()
    t = Telemetry(sink)

    async def _emit_in_span(label: str) -> str:
        async with t.span_scope() as span:
            # Brief await so the tasks actually interleave on the event loop.
            await asyncio.sleep(0)
            await t.emit(
                ToolDispatched(
                    tool_name=label,
                    call_id=label,
                    arguments={},
                    is_error=False,
                    duration_ms=0.0,
                )
            )
            return span

    async with t.session_scope():
        spans = await asyncio.gather(
            _emit_in_span("a"),
            _emit_in_span("b"),
            _emit_in_span("c"),
        )

    # Three distinct span_ids — no collision.
    assert len(set(spans)) == 3
    # Three events, each with the matching span_id.
    tool_events = [e for e in sink.events if isinstance(e, ToolDispatched)]
    assert {e.tool_name: e.span_id for e in tool_events} == dict(zip("abc", spans, strict=False))


# ---------------------------------------------------------------------------
# Orchestrator + Dispatcher integration


async def test_orchestrator_run_threads_trace_through_dispatcher_emit() -> None:
    """A full Orchestrator.run() with a Dispatcher emits both an
    OrchestratorTurn and a ToolDispatched. Both should carry the same
    trace_id; the ToolDispatched should be a child span of the turn."""
    from pydantic import BaseModel

    from harness.agents import Orchestrator, SubAgent
    from harness.hooks import HookRunner
    from harness.prompts import Message, text
    from harness.tools import Dispatcher, Tool
    from harness.tools.schema import ToolCall

    class _NoArgs(BaseModel):
        pass

    async def echo(_args: _NoArgs) -> str:
        return "ok"

    sink = MemorySink()
    telemetry = Telemetry(sink)

    dispatcher = Dispatcher(
        [Tool(name="noop", description="", input_model=_NoArgs, handler=echo)],
        telemetry=telemetry,
    )

    async def runner_with_dispatch(_agent: SubAgent, _msgs: list[Message]) -> Message:
        # Dispatch a tool call inside the runner so it's nested inside
        # the orchestrator's session_scope.
        await dispatcher.dispatch(ToolCall(name="noop", arguments={}, id="c1"))
        return text("assistant", "done")

    orch = Orchestrator(
        dispatcher,
        HookRunner(),
        runner_with_dispatch,
        telemetry=telemetry,
    )
    agent = SubAgent(name="t", system_prompt="", model="m")
    await orch.run(agent, [text("user", "hi")])

    # Two events: the dispatch (fired first, inside the run) and the
    # orchestrator turn (fired in the finally, also inside the scope).
    assert len(sink.events) == 2
    [tool_dispatched] = [e for e in sink.events if isinstance(e, ToolDispatched)]
    [orchestrator_turn] = [e for e in sink.events if isinstance(e, OrchestratorTurn)]

    # Both share the trace_id — they're part of the same session.
    assert tool_dispatched.trace_id is not None
    assert tool_dispatched.trace_id == orchestrator_turn.trace_id

    # The dispatch event has its own span_id and the orchestrator's
    # turn-level span as its parent.
    assert tool_dispatched.span_id is not None
    assert orchestrator_turn.span_id is not None
    assert tool_dispatched.span_id != orchestrator_turn.span_id
    assert tool_dispatched.parent_span_id == orchestrator_turn.span_id
