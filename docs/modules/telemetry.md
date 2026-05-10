# `harness.telemetry`

Pluggable `Sink` Protocol plus `JSONLSink`, `MemorySink`, and
`MultiSink` (zero deps). `OpenTelemetrySink` lives behind the
`[otel]` extra. Wave 11 added `trace_id` / `span_id` /
`parent_span_id` correlation auto-propagated through orchestrator
and dispatcher via `contextvars`.

## When to reach for this

- You want structured telemetry of every tool dispatch and
  orchestrator turn.
- You want JSON-Lines output for later analysis (`JSONLSink`),
  in-memory capture for tests (`MemorySink`), or live OpenTelemetry
  export (`OpenTelemetrySink`).
- You want events from a session correlated by trace_id without
  threading IDs manually.

## Quick example

```python
from harness import JSONLSink, MultiSink, Telemetry
from harness.telemetry import OpenTelemetrySink  # under [otel]

telemetry = Telemetry(
    sink=MultiSink(
        OpenTelemetrySink(),               # live observability
        JSONLSink("./events.jsonl"),       # post-hoc analysis
    ),
)

dispatcher = Dispatcher([...], telemetry=telemetry)
orchestrator = Orchestrator(dispatcher, hooks, runner, telemetry=telemetry)

# Caller-supplied trace_id for upstream propagation:
async with telemetry.session_scope(trace_id=request_trace_id):
    await orchestrator.run(agent, messages)
```

## Gotchas

- **`OpenTelemetrySink` doesn't synthesize spans.** It emits events
  on the currently-active OTel span (resolved from the global
  context). Wrap your call in a real span (FastAPI middleware, etc.)
  so the events have a parent. Without one, `add_event` is a no-op
  on `NonRecordingSpan`.
- **Sink failures are logged at WARNING and swallowed.** A
  misbehaving sink can never crash an orchestrator turn; log the
  failure and keep going.
- **`session_scope` and `span_scope` are async-context-manager.**
  Concurrent dispatches each get their own span via `contextvars`
  copy on `asyncio.create_task`.
- **No span-tree synthesis from harness events.** Faithfully
  round-tripping harness span_ids would require a custom OTel
  `IdGenerator`; tracked as deferred.

## Related

- [Cookbook: Observability with OTel](../cookbook/observability.md)
- [`examples/otel.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/otel.py)
- [`harness.privacy`](privacy.md) — emits `DetectionEvent`s through the same sink.

## API reference

::: harness.telemetry
