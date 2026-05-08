"""OpenTelemetry sink: harness telemetry events as flat OTel events.

Run with: `uv run python examples/otel.py`

`harness.telemetry.OpenTelemetrySink` translates each `TelemetryEvent`
into a flat OTel `Event` attached to the current span. The sink does
*not* synthesize spans from event durations — the existing telemetry
recorder doesn't track parent-child correlation, so faking the nesting
would produce a flat list of zero-children spans (uglier than events).
Span nesting is documented as deferred work; until the recorder grows
correlation IDs, this is the right shape.

This example builds an in-process OTel pipeline (`InMemorySpanExporter`),
opens one span, fires a `ToolDispatched` and an `OrchestratorTurn` event
through `OpenTelemetrySink`, then reads the exporter's captured spans to
prove the events landed where expected.

Requires the `[otel]` extra. Run with:
    uv sync --extra dev --extra otel
"""

from __future__ import annotations

import asyncio

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness.telemetry import OpenTelemetrySink, OrchestratorTurn, ToolDispatched


def _setup_otel() -> InMemorySpanExporter:
    """Configure an in-process OTel pipeline that captures spans in memory."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    return exporter


async def main() -> int:
    transcript: list[str] = []
    exporter = _setup_otel()
    sink = OpenTelemetrySink()

    tracer = otel_trace.get_tracer("examples.otel")

    # Open a span. Inside, fire two harness telemetry events through the
    # sink. The sink calls `span.add_event(...)` — never `start_span` —
    # so the current span owns both events as children.
    with tracer.start_as_current_span("demo-orchestrator-turn"):
        await sink.emit(
            ToolDispatched(
                tool_name="search",
                call_id="call-1",
                arguments={"query": "demo"},
                is_error=False,
                duration_ms=12.4,
            )
        )
        await sink.emit(
            OrchestratorTurn(
                agent_name="demo-agent",
                duration_ms=98.7,
                error=None,
            )
        )

    # The span is now finished + exported. Read it back from the exporter
    # and inspect the events the sink attached.
    spans = exporter.get_finished_spans()
    transcript.append(f"--- exported spans: {len(spans)} ---")
    for span in spans:
        transcript.append(f"  span: {span.name}")
        transcript.append(f"  events on span: {len(span.events)}")
        for event in span.events:
            # Promote a few attributes for readability — the sink prefixes
            # every harness payload field with `harness.`.
            attrs = dict(event.attributes or {})
            kind = attrs.get("harness.kind")
            duration = attrs.get("harness.duration_ms")
            tool = attrs.get("harness.tool_name") or attrs.get("harness.agent_name")
            transcript.append(
                f"    event name={event.name!r} kind={kind!r} "
                f"target={tool!r} duration_ms={duration}"
            )

    transcript.append("  no spans created by sink (events only) — span nesting deferred per design")

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
