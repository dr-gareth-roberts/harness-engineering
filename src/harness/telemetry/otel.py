"""OpenTelemetry sink for `harness.telemetry`.

Emits each `TelemetryEvent` as a flat OTel `Event` attached to whatever span
is currently active when `emit()` is called. Does NOT create or nest spans.

Span nesting is deferred — implement once the telemetry recorder tracks
correlation IDs across `OrchestratorTurn` and `ToolDispatched`. Without
parent-child correlation in the event stream we cannot build proper span
trees from emissions; faking the nesting (one root span per turn, child
spans per dispatch — but they are not actually nested in the data model)
would be worse than admitting the limitation.

For events that carry their own duration (`ToolDispatched.duration_ms`,
`OrchestratorTurn.duration_ms`) the duration is promoted to the
`harness.duration_ms` attribute on the OTel event. We do not synthesise
spans from durations — that would produce a flat list of zero-children
spans, which is uglier than events.

Wire OpenTelemetry up at the boundary that creates the span (FastAPI
middleware, instrumented HTTP client, etc.); `OpenTelemetrySink` then
attaches harness events to whichever span is current. When no instrumented
caller is active, `Span.add_event` is a no-op on the OTel `NonRecordingSpan`
returned by `get_current_span()` — that is the desired behaviour.

Lazy imports `opentelemetry` from the constructor so importing this module
does not require the `[otel]` extra; only constructing the sink does.

Install with: `uv sync --extra otel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harness.telemetry.events import TelemetryEvent

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


# Pydantic / harness-internal fields that are already encoded into the OTel
# event itself (via `name=event.kind`, span timestamp, and the explicit
# `harness.event_id`/`harness.kind` attributes), so we skip them when
# promoting payload fields to attributes.
_RESERVED_FIELDS = frozenset({"event_id", "timestamp", "kind"})


class OpenTelemetrySink:
    """Telemetry sink that mirrors `TelemetryEvent`s onto the current OTel span.

    Each `emit(event)` call resolves the currently active span via
    `opentelemetry.trace.get_current_span()` and calls `span.add_event()`
    with:

      - `name` = `event.kind` (e.g. `"tool.dispatched"`, `"orchestrator.turn"`)
      - `timestamp` = `event.timestamp` converted to nanoseconds since epoch
      - `attributes` = a flat dict of harness-prefixed keys derived from
        `event.model_dump()`. Scalar fields (`str | int | float | bool | None`)
        are passed through; complex values are stringified so OTel exporters
        (which only accept scalar attribute values) never reject them.

    No spans are created. See module docstring for why.
    """

    def __init__(
        self,
        tracer_name: str = "harness",
        tracer: Tracer | None = None,
    ) -> None:
        # Lazy import: importing this module must not require the [otel] extra.
        # Only construction does. Keeping the import inside __init__ also makes
        # the missing-extra test (which monkeypatches sys.modules) work — Python
        # raises ImportError on the next import attempt after a None entry, so
        # the import has to happen after the monkeypatch.
        try:
            from opentelemetry import trace as _trace
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "OpenTelemetrySink requires the [otel] extra. Install with: uv sync --extra otel"
            ) from exc

        self._trace = _trace
        self._tracer: Tracer = tracer if tracer is not None else _trace.get_tracer(tracer_name)

    async def emit(self, event: TelemetryEvent) -> None:
        # `add_event` writes onto the currently active span resolved from the
        # global OTel context — not from `self._tracer`. The tracer is held
        # only because the constructor signature accepts it; we never call
        # `start_span` / `start_as_current_span` on it.
        span = self._trace.get_current_span()

        attributes: dict[str, Any] = {
            "harness.event_id": str(event.event_id),
            "harness.kind": event.kind,
        }
        for key, value in event.model_dump().items():
            if key in _RESERVED_FIELDS:
                continue
            attr_key = f"harness.{key}"
            if value is None or isinstance(value, str | int | float | bool):
                attributes[attr_key] = value
            else:
                # OTel attribute values must be scalar (or homogeneous sequences
                # of scalars). Stringify anything else so exporters never choke.
                attributes[attr_key] = str(value)

        # OTel Span.add_event takes a Unix-epoch nanosecond timestamp. Without
        # this conversion the event would carry whatever "now" is at the moment
        # add_event runs, not the event's recorded timestamp — defeating the
        # point of a sink whose input already has a timestamp.
        timestamp_ns = int(event.timestamp.timestamp() * 1_000_000_000)

        span.add_event(name=event.kind, attributes=attributes, timestamp=timestamp_ns)


__all__ = ["OpenTelemetrySink"]
