"""Tests for `harness.telemetry.otel.OpenTelemetrySink`.

Builds a self-contained in-process OTel pipeline (TracerProvider →
SimpleSpanProcessor → InMemorySpanExporter) so assertions read events
back without hitting a real backend. Each test isolates its own
TracerProvider — we never touch the global one.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import set_tracer_provider

from harness.telemetry import MemorySink, MultiSink, OrchestratorTurn, ToolDispatched
from harness.telemetry.otel import OpenTelemetrySink


@pytest.fixture
def otel_pipeline() -> Iterator[tuple[otel_trace.Tracer, InMemorySpanExporter]]:
    """Provide an isolated TracerProvider + in-memory exporter per test.

    Setting the provider globally is required because `get_current_span()`
    consults the global `OpenTelemetry.tracer_provider`. We restore the
    previous provider after the test so other tests are not affected.
    """
    previous_provider = otel_trace.get_tracer_provider()

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        tracer = provider.get_tracer("harness-test")
        yield tracer, exporter
    finally:
        provider.shutdown()
        # `set_tracer_provider` warns on second call; assigning back via the
        # private API keeps tests isolated without polluting OTel logs.
        otel_trace._TRACER_PROVIDER = previous_provider


def make_tool_event(name: str = "echo") -> ToolDispatched:
    return ToolDispatched(
        tool_name=name,
        call_id="c1",
        arguments={"x": 1},
        is_error=False,
        duration_ms=2.5,
    )


def make_turn_event(agent: str = "alpha") -> OrchestratorTurn:
    return OrchestratorTurn(agent_name=agent, duration_ms=12.5, error=None)


# -----------------------------------------------------------------------------
# Happy path: events ride on the active span
# -----------------------------------------------------------------------------


async def test_happy_path_attaches_events_to_current_span(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink(tracer=tracer)

    with tracer.start_as_current_span("test-root"):
        await sink.emit(make_tool_event("a"))
        await sink.emit(make_tool_event("b"))
        await sink.emit(make_turn_event("alpha"))

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, "sink must not create new spans"

    root = spans[0]
    assert root.name == "test-root"

    event_names = [e.name for e in root.events]
    assert event_names == ["tool.dispatched", "tool.dispatched", "orchestrator.turn"]


async def test_promoted_attributes_for_tool_dispatched(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink(tracer=tracer)

    event = make_tool_event("greet")
    with tracer.start_as_current_span("turn"):
        await sink.emit(event)

    [span] = exporter.get_finished_spans()
    [otel_event] = span.events

    attrs: dict[str, Any] = dict(otel_event.attributes or {})
    # Required identification attributes
    assert attrs["harness.event_id"] == str(event.event_id)
    assert attrs["harness.kind"] == "tool.dispatched"
    # Promoted scalar payload fields
    assert attrs["harness.tool_name"] == "greet"
    assert attrs["harness.duration_ms"] == 2.5
    assert attrs["harness.is_error"] is False
    assert attrs["harness.call_id"] == "c1"
    # Non-scalar fields are stringified so OTel exporters never choke
    assert attrs["harness.arguments"] == str({"x": 1})


async def test_event_timestamp_preserved_in_nanoseconds(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """The recorded `event.timestamp` must travel through to the OTel event,
    not be replaced by `now()` at the moment `add_event` runs."""
    tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink(tracer=tracer)

    event = make_tool_event()
    expected_ns = int(event.timestamp.timestamp() * 1_000_000_000)

    with tracer.start_as_current_span("root"):
        await sink.emit(event)

    [span] = exporter.get_finished_spans()
    [otel_event] = span.events
    assert otel_event.timestamp == expected_ns


# -----------------------------------------------------------------------------
# Events-not-spans guarantee
# -----------------------------------------------------------------------------


async def test_sink_never_calls_span_creating_apis() -> None:
    """`OpenTelemetrySink` must not start spans — it adds events to whatever
    span is currently active. A regression where someone wires `start_span`
    in would silently produce a flat list of zero-children spans, which is
    exactly the failure mode the design avoids.
    """
    tracer = MagicMock(spec=otel_trace.Tracer)
    sink = OpenTelemetrySink(tracer=tracer)

    await sink.emit(make_tool_event())

    tracer.start_span.assert_not_called()
    tracer.start_as_current_span.assert_not_called()


async def test_sink_calls_add_event_on_current_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the positive side of the spans/events guarantee — `add_event` IS
    called, with the event kind as the name."""
    tracer = MagicMock(spec=otel_trace.Tracer)
    fake_span = MagicMock()
    monkeypatch.setattr(otel_trace, "get_current_span", lambda: fake_span)

    sink = OpenTelemetrySink(tracer=tracer)
    await sink.emit(make_turn_event("alpha"))

    fake_span.add_event.assert_called_once()
    kwargs = fake_span.add_event.call_args.kwargs
    assert kwargs["name"] == "orchestrator.turn"
    assert kwargs["attributes"]["harness.agent_name"] == "alpha"
    assert kwargs["attributes"]["harness.duration_ms"] == 12.5
    assert isinstance(kwargs["timestamp"], int)


# -----------------------------------------------------------------------------
# Sink protocol structural compatibility
# -----------------------------------------------------------------------------


async def test_satisfies_sink_protocol_via_multisink(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """Structural typing: `OpenTelemetrySink` does not inherit from `Sink`,
    but it must satisfy the protocol. Wiring it into `MultiSink` (which is
    typed `*sinks: Sink`) is the most direct demonstration."""
    tracer, exporter = otel_pipeline
    memory = MemorySink()
    multi = MultiSink(OpenTelemetrySink(tracer=tracer), memory)

    event = make_tool_event("multi")
    with tracer.start_as_current_span("multi-test"):
        await multi.emit(event)

    [span] = exporter.get_finished_spans()
    assert [e.name for e in span.events] == ["tool.dispatched"]
    assert [e.tool_name for e in memory.events if isinstance(e, ToolDispatched)] == ["multi"]


# -----------------------------------------------------------------------------
# Missing extra
# -----------------------------------------------------------------------------


def test_constructor_raises_when_otel_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting `sys.modules["opentelemetry"] = None` makes Python's import
    system treat the package as unavailable on the next import attempt.
    The lazy import inside `__init__` must therefore raise, with a message
    that points the user at `[otel]`.
    """
    # Drop any cached opentelemetry submodules so the next `import opentelemetry`
    # attempts a fresh resolution rather than picking up the existing module.
    cached = [
        name for name in sys.modules if name == "opentelemetry" or name.startswith("opentelemetry.")
    ]
    for name in cached:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)

    # Reload our module so the lazy-import path inside __init__ runs against
    # the patched sys.modules. Importing `harness.telemetry.otel` itself must
    # not fail — only construction does.
    otel_mod = importlib.reload(sys.modules["harness.telemetry.otel"])

    with pytest.raises(ImportError, match=r"\[otel\]"):
        otel_mod.OpenTelemetrySink()


async def test_none_valued_payload_fields_are_skipped_not_emitted(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """OTel attribute values can't be None — the SDK logs a warning and
    drops the attribute. We skip None-valued payload fields explicitly
    so the operator's stderr stays clean.

    Concrete trigger: `OrchestratorTurn.error: str | None = None`. A
    successful turn carries `error=None`, and emitting that as
    `harness.error=None` produced the warning.
    """
    tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink(tracer=tracer)

    # OrchestratorTurn with default error=None — the regression case.
    event = OrchestratorTurn(agent_name="demo", duration_ms=42.0)
    assert event.error is None

    with tracer.start_as_current_span("turn"):
        await sink.emit(event)

    [span] = exporter.get_finished_spans()
    [otel_event] = span.events
    attrs: dict[str, Any] = dict(otel_event.attributes or {})

    # The error attribute must NOT be present (skipped, not emitted as None).
    assert "harness.error" not in attrs
    # Non-None scalar payload fields are still promoted.
    assert attrs["harness.agent_name"] == "demo"
    assert attrs["harness.duration_ms"] == 42.0
