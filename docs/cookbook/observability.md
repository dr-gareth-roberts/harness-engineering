# Observability with OpenTelemetry

## Problem

Your agent runs in production. You want to see, per session: which
tools fired, how long each dispatch took, how the dispatches relate
to each other (ordering, parallelism), and which session the events
belong to. Ideally in your existing OTel-compatible backend
(Jaeger, Tempo, Honeycomb, etc.) without writing custom export glue.

## Solution sketch

`harness.telemetry` ships a pluggable `Sink` Protocol. `JSONLSink`
and `MemorySink` are zero-dep; `OpenTelemetrySink` lives behind the
`[otel]` extra and emits each `TelemetryEvent` as a flat OTel `Event`
on the currently-active span.

Wave 11 added correlation IDs (`trace_id`, `span_id`,
`parent_span_id`) on every event. The `Telemetry` recorder
auto-generates them via `contextvars`, the `Orchestrator` opens a
session-scope per `run()`, the `Dispatcher` opens a span-scope per
`dispatch()`. Result: events emitted under one orchestrator turn
all share the same `trace_id`; tool dispatches nest under the
turn-span as `parent_span_id`. You can group / filter on
`harness.trace_id` in your backend.

## Working code

Install:

<!-- reason: shell example, not executed in the codeblock gate -->
<!--pytest.mark.skip-->
```bash
uv add 'harness-engineering-toolkit[otel]'
```

<!-- reason: illustrative; needs the [otel] + [anthropic] extras and uses `await` at module scope -->
<!--pytest.mark.skip-->
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from harness import (
    AnthropicRunner,
    Dispatcher,
    HookRunner,
    OpenTelemetrySink,
    Orchestrator,
    Telemetry,
)


# 1. Stand up an OTel TracerProvider — point it at your real backend
#    in production. ConsoleSpanExporter is just a sanity check.
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# 2. Build the harness telemetry recorder backed by the OTel sink.
telemetry = Telemetry(sink=OpenTelemetrySink())

# 3. Wire the recorder into both Dispatcher and Orchestrator.
dispatcher = Dispatcher([...], telemetry=telemetry)
orchestrator = Orchestrator(
    dispatcher,
    HookRunner(),
    AnthropicRunner(dispatcher, HookRunner()),
    telemetry=telemetry,
)

# 4. Drive normally. Wrap in your own root span (FastAPI middleware /
#    instrumented HTTP client / etc.) so the harness events attach to
#    something the exporter sees.
tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("user-request"):
    await orchestrator.run(agent, messages)
```

What the backend sees:

- One root span (`user-request`).
- A `harness.trace_id` attribute on every emitted event.
- An `orchestrator.turn` event with `harness.agent_name` /
  `harness.duration_ms`.
- One `tool.dispatched` event per dispatch, with `harness.tool_name`,
  `harness.is_error`, `harness.duration_ms`, `harness.span_id`,
  `harness.parent_span_id` (= the turn's span_id).

Filter for one session: `harness.trace_id = "abc..."`. Find all
tool calls slower than 200ms: `harness.kind = "tool.dispatched"
AND harness.duration_ms > 200`.

## Multi-sink (OTel + JSONL fallback)

Often you want OTel for live observability *and* a JSONL file for
post-hoc analysis or audit. `MultiSink` fans out:

<!-- reason: illustrative; constructing OpenTelemetrySink requires the [otel] extra -->
<!--pytest.mark.skip-->
```python
from harness import JSONLSink, MultiSink, OpenTelemetrySink, Telemetry

telemetry = Telemetry(
    sink=MultiSink(
        OpenTelemetrySink(),
        JSONLSink("./events.jsonl"),
    ),
)
```

If one sink raises, the recorder logs at WARNING and keeps going —
a misbehaving sink can't crash the orchestrator turn.

## Use the correlation IDs without OTel

Even without `[otel]` installed, the correlation IDs ride on every
event:

<!-- reason: illustrative; references undefined orchestrator / agent / messages and uses `await` at module scope -->
<!--pytest.mark.skip-->
```python
from harness import MemorySink, Telemetry

sink = MemorySink()
telemetry = Telemetry(sink=sink)

await orchestrator.run(agent, messages)

# Events are correlated by trace_id without OTel involvement.
turn_events = [e for e in sink.events if e.kind == "orchestrator.turn"]
trace_id = turn_events[0].trace_id

related = [e for e in sink.events if e.trace_id == trace_id]
print(f"{len(related)} events under trace {trace_id}")
```

## Caller-supplied trace_id (propagation)

If your upstream service passes a request-trace ID, propagate it:

<!-- reason: illustrative; `async with` at module scope and references undefined names -->
<!--pytest.mark.skip-->
```python
async with telemetry.session_scope(trace_id=request.headers["x-trace-id"]):
    await orchestrator.run(agent, messages)
```

The orchestrator's auto-opened scope respects the ambient
`trace_id` if one is already set, so external traces flow through.

## Gotchas

- **`OpenTelemetrySink` synthesizes one span per event.** Each
  `TelemetryEvent` becomes an OTel span whose name is `event.kind`,
  whose attributes are the harness-prefixed payload fields, and
  whose parent `SpanContext` is seeded from the recorder's
  `trace_id` / `span_id` — so harness events land in the same OTel
  trace as any upstream instrumentation. Events carrying
  `duration_ms` (e.g. `ToolDispatched`, `OrchestratorTurn`) get a
  realistic `end_time` rather than a zero-width span. When
  `event.trace_id` is absent (events emitted outside a
  `session_scope`), the sink falls back to whatever OTel context is
  ambient — graceful degradation matching the pre-1.2.0 "ride on
  the current span" behavior.
- **Concurrent dispatches keep distinct span_ids.** `asyncio.create_task`
  copies `contextvars`, so `asyncio.gather` over parallel
  `Dispatcher.dispatch` calls each get their own span. No collision.
- **Sink protocol is just `async emit(event) -> None`.** Anything
  satisfying that signature works; you don't need to inherit from a
  base class.

### Migration from pre-1.3.0

Before 1.3.0, `OpenTelemetrySink` emitted each event as a flat OTel
`Event` on the currently active span and silently no-op'd onto
`NonRecordingSpan` when no instrumented caller was wrapping the
harness call. 1.3.0 replaces that with the synthesized-span
contract above: callers who used to rely on `add_event` semantics
should now see real spans in their backend without any wrapping
required. Backends that grouped events by `harness.trace_id`
attribute continue to work — the attribute is still set on every
span alongside the structurally-seeded SpanContext.

## Related

- [`harness.telemetry`](../modules/telemetry.md) — module reference.
- [`examples/otel.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/otel.py)
  — runnable in-process OTel pipeline demo (`InMemorySpanExporter`).
- [Cookbook: Cache + speculate](cache-and-speculate.md) — pair OTel
  spans with prefix-cache audits to see drift events alongside
  request latency.
