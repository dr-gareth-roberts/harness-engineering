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

<!-- reason: illustrative; needs the [otel] extra, references undefined Dispatcher import, and uses `async with` at module scope -->
<!--pytest.mark.skip-->
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

## Redacting events at the sink boundary

`ToolDispatched.arguments` and any other event field carrying
user/model content flows **verbatim** to every sink by default. The
`harness.privacy` module's "audit events never carry matched values"
invariant is privacy-module-local — it does **not** apply to
`harness.telemetry`. If you wire `JSONLSink("./audit.jsonl")` expecting
audit-grade output, you'll capture secrets verbatim.

Two boundaries, two tools:

| Boundary | Surface | Tool |
| --- | --- | --- |
| Telemetry sink | `Telemetry.emit` → sinks | `redactor=` (this page) |
| Model I/O | runner inputs/outputs | [`harness.privacy.PrivacyBoundary`](privacy.md) |

Pass a `redactor` to `Telemetry` to scrub events at the recorder
boundary *before* fan-out to any sink. The redactor takes a
`TelemetryEvent` and returns a (possibly transformed) event of the
same type:

```python
from harness import JSONLSink, Telemetry
from harness.telemetry import Redactor, TelemetryEvent, ToolDispatched

_SENSITIVE_KEYS = {"password", "api_key", "token", "secret"}


def scrub_tool_arguments(event: TelemetryEvent) -> TelemetryEvent:
    """Redact sensitive-looking keys from `ToolDispatched.arguments`."""
    if not isinstance(event, ToolDispatched):
        return event
    redacted = {
        k: ("[REDACTED]" if k.lower() in _SENSITIVE_KEYS else v)
        for k, v in event.arguments.items()
    }
    # `model_copy(update=...)` returns a new instance — don't mutate
    # the input, sinks like `MemorySink` retain references.
    return event.model_copy(update={"arguments": redacted})


redactor: Redactor = scrub_tool_arguments
telemetry = Telemetry(
    sink=JSONLSink("./audit.jsonl"),
    redactor=redactor,
)
```

Contract:

- The redactor runs **after** correlation IDs are threaded in and
  **before** the sink sees the event — every sink in a `MultiSink`
  observes the same redacted view.
- The redactor should be **pure-data**: return a new instance (via
  `event.model_copy(update={...})`) rather than mutating the input.
  Sinks that retain references (e.g. `MemorySink`) would otherwise
  observe the mutation post-hoc.
- The recorder does **not** catch exceptions raised by the redactor —
  a bug in the scrubber is a configuration error, and silently
  dropping events on a redactor crash would be worse than a loud
  failure. Wrap your redactor's body in `try/except` if you want soft
  failure modes.
- Default `redactor=None` preserves the pre-existing behavior — no
  regression for callers that don't opt in.

> Telemetry sinks are **not** audit-grade by default. The `redactor=`
> kwarg covers the telemetry-boundary case (e.g. scrubbing
> `ToolDispatched.arguments` before they hit `JSONLSink`). For
> audit-grade redaction of model I/O across the runner boundary, use
> `harness.privacy.PrivacyBoundary` to wrap the runner — that's a
> different boundary with stronger guarantees (the boundary fully
> rewrites detected matches in the model-visible payload, not just in
> downstream telemetry).

## `OpenTelemetrySink` span synthesis

Each `TelemetryEvent` becomes a real OTel span — not a flat event on
the ambient span. The recorder's correlation IDs drive the mapping:

| Harness field | OTel field |
| --- | --- |
| `event.trace_id` (32-hex / 128-bit) | synthesized span's `trace_id` |
| `event.span_id` (16-hex / 64-bit) | synthesized span's **parent** span_id |
| `event.kind` (e.g. `"tool.dispatched"`) | synthesized span's `name` |
| `event.timestamp` | synthesized span's `start_time` |
| `event.duration_ms` (if present) | `end_time = start_time + duration_ms` |
| `event.parent_span_id` | `harness.parent_span_id` attribute |

The synthesized span's own `span_id` is minted by the configured
`IdGenerator` — OTel does not expose an API to override it. The
fidelity floor:

- **Trace continuity is faithful.** All events from one harness
  session live in one OTel trace.
- **One level of parent linkage is structurally encoded.** The
  synthesized span's parent is the harness scope that emitted it.
- **Deeper nesting emerges implicitly** from the harness scope
  ordering (and from the `harness.parent_span_id` attribute, which
  lets viewers reconstruct the full chain on query).

To unify with an upstream OTel trace, propagate the upstream
trace_id into `session_scope`:

<!-- reason: illustrative; references undefined `telemetry` / `upstream_trace_id_hex` and uses `async with` at module scope -->
<!--pytest.mark.skip-->
```python
async with telemetry.session_scope(trace_id=upstream_trace_id_hex):
    ...
```

The `tracer` kwarg pins the sink to a specific tracer instance
(useful for tests that want isolation from the global TracerProvider,
or for multi-tenant configurations):

```python
from harness.telemetry import OpenTelemetrySink
from opentelemetry import trace
sink = OpenTelemetrySink(tracer=trace.get_tracer("my-service"))
```

When omitted, the sink resolves a tracer from the global provider via
`trace.get_tracer("harness")`.

When an event has no correlation IDs (emitted outside any
`session_scope`), the sink falls back to the ambient OTel context —
preserving the pre-1.3.0 "ride on the surrounding span" behavior as
graceful degradation.

## Gotchas

- **Spans live in the harness trace, not the ambient one.** When you
  wrap a call in an upstream OTel span and use `session_scope()` to
  mint a fresh `trace_id`, the synthesized harness spans will be in a
  *different* OTel trace than your upstream span. To unify, pass the
  upstream trace_id into `session_scope(trace_id=...)`.
- **Sink failures are logged at WARNING and swallowed.** A
  misbehaving sink can never crash an orchestrator turn; log the
  failure and keep going.
- **`session_scope` and `span_scope` are async-context-manager.**
  Concurrent dispatches each get their own span via `contextvars`
  copy on `asyncio.create_task`.
- **`redactor=` is the sink boundary, not the model boundary.** If
  you need to keep secrets out of the model's input/output as well
  as out of telemetry, reach for
  [`harness.privacy.PrivacyBoundary`](privacy.md).

## Related

- [Cookbook: Observability with OTel](../cookbook/observability.md)
- [`examples/otel.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/otel.py)
- [`harness.privacy`](privacy.md) — emits `DetectionEvent`s through the same sink.

## API reference

::: harness.telemetry
