"""Tests for `harness.telemetry.otel.OpenTelemetrySink`.

Builds a self-contained in-process OTel pipeline (TracerProvider →
SimpleSpanProcessor → InMemorySpanExporter) so assertions read spans
back without hitting a real backend. Each test isolates its own
TracerProvider — we never touch the global one.

M3.5 changed the sink's contract: it now synthesizes real OTel spans
from `TelemetryEvent`s rather than attaching them as flat events on
the ambient span. These tests pin the new contract.
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

from harness.telemetry import MemorySink, MultiSink, OrchestratorTurn, ToolDispatched
from harness.telemetry.otel import OpenTelemetrySink


@pytest.fixture
def otel_pipeline() -> Iterator[tuple[otel_trace.Tracer, InMemorySpanExporter]]:
    """Provide an isolated TracerProvider + in-memory exporter per test.

    Setting the provider globally is required because the sink's default
    tracer is resolved via `trace.get_tracer("harness")`, which consults
    the global provider. We restore the previous provider after the test
    so other tests are not affected.

    Restoration nuance: we snapshot the raw `_TRACER_PROVIDER` module
    global (which may be `None` if `set_tracer_provider` was never
    called) rather than `get_tracer_provider()`'s return (which returns
    `_PROXY_TRACER_PROVIDER` when `_TRACER_PROVIDER is None`). Writing
    the proxy back into `_TRACER_PROVIDER` would create a self-loop —
    `ProxyTracerProvider.get_tracer` delegates to `_TRACER_PROVIDER`,
    causing infinite recursion the next time anyone calls `get_tracer`.
    """
    previous_provider = otel_trace._TRACER_PROVIDER

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # `set_tracer_provider` is `do_once` — after the first call it
    # warns and refuses. Write the private global directly so each
    # test gets a fresh, isolated provider regardless of test order.
    otel_trace._TRACER_PROVIDER = provider
    try:
        tracer = provider.get_tracer("harness-test")
        yield tracer, exporter
    finally:
        provider.shutdown()
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
# Happy path: events synthesize their own spans
# -----------------------------------------------------------------------------


async def test_each_event_synthesizes_a_span_named_after_kind(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """The sink now creates a span per event, named after `event.kind`.

    Pre-M3.5 behavior (events attached to the ambient span) is gone:
    the sink owns its spans, end-to-end.
    """
    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()

    # No ambient span — the sink should still produce spans (events
    # without correlation IDs use the ambient context, which is empty
    # here, so they become root-level spans).
    await sink.emit(make_tool_event("a"))
    await sink.emit(make_tool_event("b"))
    await sink.emit(make_turn_event("alpha"))

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == [
        "tool.dispatched",
        "tool.dispatched",
        "orchestrator.turn",
    ]


async def test_promoted_attributes_for_tool_dispatched(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """Payload fields ride as `harness.*` attributes on the synthesized span."""
    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()

    event = make_tool_event("greet")
    await sink.emit(event)

    [span] = exporter.get_finished_spans()
    attrs: dict[str, Any] = dict(span.attributes or {})

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


async def test_event_timestamp_becomes_span_start_time(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """The recorded `event.timestamp` must seed `span.start_time`,
    not be replaced by `now()` when the span is created."""
    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()

    event = make_tool_event()
    expected_ns = int(event.timestamp.timestamp() * 1_000_000_000)

    await sink.emit(event)

    [span] = exporter.get_finished_spans()
    assert span.start_time == expected_ns


async def test_duration_ms_becomes_span_width(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """Events carrying `duration_ms` produce spans whose end_time
    matches `start_time + duration_ms`, so viewers show realistic
    durations rather than zero-width markers."""
    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()

    event = make_tool_event()  # duration_ms=2.5
    await sink.emit(event)

    [span] = exporter.get_finished_spans()
    assert span.end_time is not None
    assert span.start_time is not None
    width_ns = span.end_time - span.start_time
    assert width_ns == int(2.5 * 1_000_000)


# -----------------------------------------------------------------------------
# M1.7 regression: the `tracer` kwarg has real effect
# -----------------------------------------------------------------------------


async def test_explicit_tracer_kwarg_is_actually_used() -> None:
    """M1.7 lesson: an unused kwarg is a bug.

    The pre-1.0.3 sink accepted `tracer=` and silently ignored it; we
    removed the kwarg in 1.0.3 rather than ship a no-op. M3.5 re-adds
    the kwarg with real behavior. This test pins that behavior: a
    supplied tracer is the tracer the sink calls.
    """
    mock_tracer = MagicMock()
    sink = OpenTelemetrySink(tracer=mock_tracer)

    await sink.emit(make_turn_event("alpha"))

    mock_tracer.start_as_current_span.assert_called_once()
    call = mock_tracer.start_as_current_span.call_args
    assert call.kwargs["name"] == "orchestrator.turn"
    # start_time is forwarded, end_on_exit=False (so we can end with a
    # precise end_time below).
    assert isinstance(call.kwargs["start_time"], int)
    assert call.kwargs["end_on_exit"] is False


async def test_default_tracer_resolved_from_global_provider(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """When no tracer is supplied, the sink resolves one from the global
    provider via `trace.get_tracer("harness")`. The in-memory exporter
    wired to the global provider in this test fixture proves it."""
    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()  # no explicit tracer

    await sink.emit(make_turn_event("default-tracer"))

    [span] = exporter.get_finished_spans()
    assert span.name == "orchestrator.turn"
    attrs: dict[str, Any] = dict(span.attributes or {})
    assert attrs["harness.agent_name"] == "default-tracer"


# -----------------------------------------------------------------------------
# Correlation-ID-driven span tree synthesis
# -----------------------------------------------------------------------------


async def test_synthesized_span_inherits_trace_id_from_event(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """`event.trace_id` seeds the SpanContext's `trace_id`, so the
    synthesized span lives in the same OTel trace as the harness session.

    The harness uses 32-hex (128-bit) trace IDs for exactly this reason —
    they can be used as OTel trace IDs as-is.
    """
    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()

    event = make_tool_event()
    event.trace_id = "ab" * 16  # 32 hex chars = 128 bits
    event.span_id = "cd" * 8  # 16 hex chars = 64 bits
    await sink.emit(event)

    [span] = exporter.get_finished_spans()
    assert format(span.context.trace_id, "032x") == "ab" * 16
    # The event's span_id became the synthesized span's parent.
    assert span.parent is not None
    assert format(span.parent.span_id, "016x") == "cd" * 8


async def test_consecutive_events_share_a_trace(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """Two events with the same `trace_id` produce two synthesized
    spans in the same OTel trace. This is the core trace-continuity
    guarantee M3.5 ships: an OTel viewer shows the harness session as
    one connected trace, not a scatter of unrelated spans."""
    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()

    trace_id_hex = "12" * 16
    event_a = make_tool_event("a")
    event_a.trace_id = trace_id_hex
    event_a.span_id = "aa" * 8

    event_b = make_tool_event("b")
    event_b.trace_id = trace_id_hex
    event_b.span_id = "bb" * 8

    await sink.emit(event_a)
    await sink.emit(event_b)

    span_a, span_b = exporter.get_finished_spans()
    assert span_a.context.trace_id == span_b.context.trace_id
    assert format(span_a.context.trace_id, "032x") == trace_id_hex


async def test_events_without_trace_id_fall_back_to_ambient_context(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """An event without correlation IDs (e.g. emitted outside any
    `session_scope`) gets a span under whatever OTel context is
    currently active. This preserves pre-M3.5 "ride on the ambient
    span" behavior as graceful degradation when the recorder hasn't
    set up correlation IDs."""
    tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()

    with tracer.start_as_current_span("ambient-root") as root:
        event = make_tool_event()
        assert event.trace_id is None
        await sink.emit(event)

    spans_by_name = {s.name: s for s in exporter.get_finished_spans()}
    synthesized = spans_by_name["tool.dispatched"]
    root_finished = spans_by_name["ambient-root"]
    # The synthesized span's parent is the ambient root span.
    assert synthesized.parent is not None
    assert synthesized.parent.span_id == root.get_span_context().span_id
    # Both spans share the ambient root's trace_id.
    assert synthesized.context.trace_id == root_finished.context.trace_id


async def test_invalid_hex_trace_id_falls_back_gracefully(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """A non-hex `trace_id` (e.g. someone shoved a UUID with dashes into
    it from upstream) can't seed a `SpanContext`. The sink falls back
    rather than crashing — the recorder's failure-isolation contract
    requires that a sink never crash an orchestrator turn."""
    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()

    event = make_tool_event()
    event.trace_id = "not-hex-at-all"
    event.span_id = "also-bad"
    await sink.emit(event)

    [span] = exporter.get_finished_spans()
    assert span.name == "tool.dispatched"
    # No parent linkage from the invalid IDs — fell back to ambient
    # context (which is empty in this test, so the span is a root).
    assert span.parent is None


# -----------------------------------------------------------------------------
# Sink protocol structural compatibility
# -----------------------------------------------------------------------------


async def test_satisfies_sink_protocol_via_multisink(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """Structural typing: `OpenTelemetrySink` does not inherit from `Sink`,
    but it must satisfy the protocol. Wiring it into `MultiSink` (which is
    typed `*sinks: Sink`) is the most direct demonstration."""
    _tracer, exporter = otel_pipeline
    memory = MemorySink()
    multi = MultiSink(OpenTelemetrySink(), memory)

    event = make_tool_event("multi")
    await multi.emit(event)

    [span] = exporter.get_finished_spans()
    assert span.name == "tool.dispatched"
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
    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()

    # OrchestratorTurn with default error=None — the regression case.
    event = OrchestratorTurn(agent_name="demo", duration_ms=42.0)
    assert event.error is None

    await sink.emit(event)

    [span] = exporter.get_finished_spans()
    attrs: dict[str, Any] = dict(span.attributes or {})

    # The error attribute must NOT be present (skipped, not emitted as None).
    assert "harness.error" not in attrs
    # Non-None scalar payload fields are still promoted.
    assert attrs["harness.agent_name"] == "demo"
    assert attrs["harness.duration_ms"] == 42.0


# ---------------------------------------------------------------------------
# Wave 11 #11: correlation IDs flow through the recorder into span structure


async def test_correlation_ids_threaded_through_recorder_seed_span_context(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """An event emitted inside `Telemetry.session_scope + span_scope`
    carries `trace_id` / `span_id` / `parent_span_id`. After M3.5 those
    IDs drive the synthesized span's SpanContext (trace_id) and parent
    (span_id), not just decorative attributes.

    The `harness.*` attributes are also still present — querying spans
    by attribute is cheap, and dashboards built against the attribute
    names in pre-M3.5 should keep working.
    """
    from harness.telemetry import Telemetry

    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()
    telemetry = Telemetry(sink)

    async with (
        telemetry.session_scope() as trace_id,
        telemetry.span_scope() as span_id,
    ):
        await telemetry.emit(make_tool_event())

    [span] = exporter.get_finished_spans()
    # Trace continuity: synthesized span lives in the harness trace.
    assert format(span.context.trace_id, "032x") == trace_id
    # Parent linkage: synthesized span's parent is the harness scope.
    assert span.parent is not None
    assert format(span.parent.span_id, "016x") == span_id
    # Attributes are still present for dashboard compatibility.
    attrs: dict[str, Any] = dict(span.attributes or {})
    assert attrs["harness.trace_id"] == trace_id
    assert attrs["harness.span_id"] == span_id
    # Top-level span has no parent_span_id from the recorder.
    assert "harness.parent_span_id" not in attrs


async def test_nested_span_scope_records_parent_span_id_attribute(
    otel_pipeline: tuple[otel_trace.Tracer, InMemorySpanExporter],
) -> None:
    """Nested `span_scope`s populate `parent_span_id` on the event. The
    sink promotes that as the `harness.parent_span_id` attribute (OTel
    only carries one parent in a SpanContext, so the grandparent isn't
    structurally encoded — but it's preserved as an attribute so users
    can group / filter on it)."""
    from harness.telemetry import Telemetry

    _tracer, exporter = otel_pipeline
    sink = OpenTelemetrySink()
    telemetry = Telemetry(sink)

    async with (
        telemetry.session_scope(),
        telemetry.span_scope() as outer_span,
        telemetry.span_scope() as _inner_span,
    ):
        await telemetry.emit(make_tool_event())

    [span] = exporter.get_finished_spans()
    attrs: dict[str, Any] = dict(span.attributes or {})
    assert attrs["harness.parent_span_id"] == outer_span
