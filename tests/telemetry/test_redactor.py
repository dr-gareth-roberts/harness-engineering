"""Tests for the `Telemetry(redactor=...)` boundary (M2.7).

`ToolDispatched.arguments` flows verbatim to every sink today; the
privacy module's "audit events never carry matched values" invariant is
privacy-module-local. The `redactor=` kwarg gives callers a
telemetry-boundary scrubber that runs before sink fan-out so JSONL /
OTel / MultiSink all observe the same redacted view.

These tests pin:

- Default `redactor=None` preserves the pre-existing event shape (no
  regression for existing users).
- A redactor that scrubs `ToolDispatched.arguments` is applied before
  the sink — the sink observes the redacted event, not the original.
- The redactor runs *after* correlation IDs are threaded in, so the
  scrubber sees the full populated event.
- A `MultiSink` fan-out sees the same redacted event in every sink.
- A redactor that mutates the input (anti-pattern) is documented as
  not-our-problem — pin via a `model_copy`-based pure-data redactor.
"""

from __future__ import annotations

from harness.telemetry import (
    MemorySink,
    MultiSink,
    OrchestratorTurn,
    Redactor,
    Telemetry,
    TelemetryEvent,
    ToolDispatched,
)

_SENTINEL_SECRET = "sk-do-not-leak"


def _tool_event(**overrides: object) -> ToolDispatched:
    payload: dict[str, object] = {
        "tool_name": "fetch",
        "call_id": "c1",
        "arguments": {"url": "https://example.com", "api_key": _SENTINEL_SECRET},
        "is_error": False,
        "duration_ms": 1.0,
    }
    payload.update(overrides)
    return ToolDispatched(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Default — no redactor → no behavior change (regression test for M2.7).


async def test_default_redactor_none_preserves_event_shape() -> None:
    """No `redactor=` kwarg → events reach the sink verbatim. This is the
    contract for every caller that existed before M2.7."""
    sink = MemorySink()
    telemetry = Telemetry(sink)

    await telemetry.emit(_tool_event())

    [evt] = sink.events
    assert isinstance(evt, ToolDispatched)
    # Sentinel survives — the recorder did not scrub by default.
    assert evt.arguments == {
        "url": "https://example.com",
        "api_key": _SENTINEL_SECRET,
    }


async def test_explicit_redactor_none_preserves_event_shape() -> None:
    """`redactor=None` is the explicit form of the default. Pinning both
    so a future signature change can't silently flip the default."""
    sink = MemorySink()
    telemetry = Telemetry(sink, redactor=None)

    await telemetry.emit(_tool_event())

    [evt] = sink.events
    assert isinstance(evt, ToolDispatched)
    assert evt.arguments["api_key"] == _SENTINEL_SECRET


# ---------------------------------------------------------------------------
# Redactor applied before sink fan-out.


async def test_redactor_scrubs_arguments_before_sink() -> None:
    """A redactor that scrubs `ToolDispatched.arguments` for sensitive
    keys: the sink observes the redacted event, not the original."""
    sensitive_keys = {"password", "api_key", "token", "secret"}

    def scrub(event: TelemetryEvent) -> TelemetryEvent:
        if not isinstance(event, ToolDispatched):
            return event
        redacted = {
            k: ("[REDACTED]" if k.lower() in sensitive_keys else v)
            for k, v in event.arguments.items()
        }
        # model_copy returns a NEW instance — the input is left intact.
        return event.model_copy(update={"arguments": redacted})

    sink = MemorySink()
    telemetry = Telemetry(sink, redactor=scrub)

    # Construct outside the recorder so we can assert the original
    # wasn't mutated underneath us.
    original = _tool_event()
    await telemetry.emit(original)

    [evt] = sink.events
    assert isinstance(evt, ToolDispatched)
    assert evt.arguments == {
        "url": "https://example.com",
        "api_key": "[REDACTED]",
    }
    # Pure-data contract: the redactor returned a new event, so the
    # caller's reference still carries the original value.
    assert original.arguments["api_key"] == _SENTINEL_SECRET


async def test_redactor_can_be_type_annotated_with_redactor_alias() -> None:
    """The exported `Redactor` alias is the documented contract surface.
    A test that binds a function to `Redactor` keeps the alias real —
    if someone narrows the alias in a way that breaks user code, this
    test fails to typecheck."""

    def passthrough(event: TelemetryEvent) -> TelemetryEvent:
        return event

    redactor: Redactor = passthrough
    sink = MemorySink()
    telemetry = Telemetry(sink, redactor=redactor)
    await telemetry.emit(_tool_event())

    [evt] = sink.events
    assert isinstance(evt, ToolDispatched)


# ---------------------------------------------------------------------------
# Redactor sees correlation IDs (runs after threading, before sink).


async def test_redactor_sees_correlation_ids() -> None:
    """The redactor runs *after* the recorder fills in trace_id /
    span_id from contextvars — so a redactor that scrubs based on
    correlation ID can do so. Pins ordering."""
    seen_trace_ids: list[str | None] = []

    def capture_trace(event: TelemetryEvent) -> TelemetryEvent:
        seen_trace_ids.append(event.trace_id)
        return event

    sink = MemorySink()
    telemetry = Telemetry(sink, redactor=capture_trace)

    async with telemetry.session_scope() as trace_id:
        await telemetry.emit(_tool_event())

    assert seen_trace_ids == [trace_id]


# ---------------------------------------------------------------------------
# MultiSink fan-out sees the same redacted event.


async def test_redactor_runs_once_before_multisink_fanout() -> None:
    """Every sink in a `MultiSink` observes the same redacted event.
    The redactor sits at the recorder boundary, *upstream* of fan-out —
    so a `JSONLSink` for audit + a `MemorySink` for tests can't
    disagree on what was scrubbed."""
    calls: list[str] = []

    def scrub(event: TelemetryEvent) -> TelemetryEvent:
        calls.append("redactor")
        if not isinstance(event, ToolDispatched):
            return event
        return event.model_copy(update={"arguments": {k: "[REDACTED]" for k in event.arguments}})

    sink_a = MemorySink()
    sink_b = MemorySink()
    telemetry = Telemetry(MultiSink(sink_a, sink_b), redactor=scrub)

    await telemetry.emit(_tool_event())

    # Redactor ran exactly once — recorder-level, not per-sink.
    assert calls == ["redactor"]
    # Both sinks observed the same redacted shape.
    [a_evt] = sink_a.events
    [b_evt] = sink_b.events
    assert isinstance(a_evt, ToolDispatched)
    assert isinstance(b_evt, ToolDispatched)
    assert (
        a_evt.arguments
        == b_evt.arguments
        == {
            "url": "[REDACTED]",
            "api_key": "[REDACTED]",
        }
    )


# ---------------------------------------------------------------------------
# Type narrowing — a redactor can also leave non-matching events alone.


async def test_redactor_can_pass_through_unaffected_event_types() -> None:
    """A redactor that targets `ToolDispatched` only must leave
    `OrchestratorTurn` (and any other event) alone. Pins the
    isinstance-guarded pass-through pattern documented in the module
    docs."""

    def tool_only_scrub(event: TelemetryEvent) -> TelemetryEvent:
        if isinstance(event, ToolDispatched):
            return event.model_copy(update={"arguments": {}})
        return event

    sink = MemorySink()
    telemetry = Telemetry(sink, redactor=tool_only_scrub)

    await telemetry.emit(_tool_event())
    await telemetry.emit(OrchestratorTurn(agent_name="agent", duration_ms=1.0, error=None))

    tool_events = [e for e in sink.events if isinstance(e, ToolDispatched)]
    turn_events = [e for e in sink.events if isinstance(e, OrchestratorTurn)]
    assert len(tool_events) == 1
    assert tool_events[0].arguments == {}  # scrubbed
    assert len(turn_events) == 1
    assert turn_events[0].agent_name == "agent"  # untouched
