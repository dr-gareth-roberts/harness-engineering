"""OpenTelemetry sink for `harness.telemetry`.

Each `TelemetryEvent` is synthesized into a real OTel span whose name is
the event's `kind` (e.g. `"tool.dispatched"`, `"orchestrator.turn"`) and
whose attributes are the harness-prefixed payload fields. The recorder's
correlation IDs (Wave 11 #11 — `trace_id`, `span_id`, `parent_span_id`)
seed the parent `SpanContext`, so the synthesized span lives in the
same OTel trace as the harness session and links back to the harness
scope that emitted it.

**Span synthesis model (M3.5):**

- `event.trace_id` (32-hex / 128-bit) becomes the OTel `trace_id` of the
  synthesized span.
- `event.span_id` (16-hex / 64-bit) becomes the **parent** span_id of
  the synthesized span. The synthesized span's own span_id is minted by
  the configured `IdGenerator` — OTel does not expose an API to override
  it. This is the strict-improvement floor M3.5 ships: trace continuity
  and one level of parent linkage are faithful; deeper nesting emerges
  implicitly from harness scope ordering rather than being structurally
  encoded in every span's SpanContext.
- `event.parent_span_id` is the grandparent in this view — OTel only
  carries one parent per span, so the grandparent isn't structurally
  encoded but is preserved as the `harness.parent_span_id` attribute so
  users can group / filter on it in their backend.

When `event.trace_id` is absent (events emitted outside any
`session_scope`), the sink falls back to creating a span under whatever
OTel context is currently active — preserving the pre-M3.5 "ride on the
ambient span" behavior as graceful degradation.

The `tracer` constructor kwarg is honored: pass `tracer=my_tracer` to
pin the sink to a specific tracer instance (useful for tests that want
isolation from the global TracerProvider, or for multi-tenant
configurations). When omitted, the sink resolves a tracer from the
global provider via `trace.get_tracer("harness")`. (M1.7 lesson: the
kwarg used to be silently ignored; M3.5 makes it real.)

For events that carry their own duration (`ToolDispatched.duration_ms`,
`OrchestratorTurn.duration_ms`) the synthesized span's `end_time` is
set to `start_time + duration_ms` so OTel viewers show realistic
durations rather than zero-width spans.

Lazy imports `opentelemetry` from the constructor so importing this
module does not require the `[otel]` extra; only constructing the sink
does. Install with: `uv sync --extra otel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harness.telemetry.events import TelemetryEvent

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

# Pydantic / harness-internal fields that are already encoded into the
# OTel span itself (via `name=event.kind`, `span.start_time`, the
# explicit `harness.event_id`/`harness.kind` attributes, and the
# SpanContext seeded from `trace_id`/`span_id`/`parent_span_id`), so we
# skip them when promoting payload fields to attributes. Correlation
# IDs are intentionally re-emitted as `harness.*` attributes too —
# they're structurally redundant once the SpanContext is seeded, but
# users built dashboards against the attribute names in pre-M3.5
# versions and silently dropping them would be a contract break.
_RESERVED_FIELDS = frozenset({"event_id", "timestamp", "kind"})


class OpenTelemetrySink:
    """Telemetry sink that synthesizes OTel spans from `TelemetryEvent`s.

    Each `emit(event)` call:

    1. Seeds a parent `SpanContext` from the event's correlation IDs
       (`trace_id` + `span_id`). If absent, falls back to the ambient
       OTel context.
    2. Starts an OTel span via `tracer.start_as_current_span(...)` with
       `name=event.kind`, `start_time=event.timestamp` (ns), and the
       seeded context as parent.
    3. Sets `harness.*`-prefixed attributes for every scalar payload
       field; stringifies non-scalars so exporters never reject them.
    4. Ends the span; if the event carries `duration_ms`, the `end_time`
       is `start_time + duration_ms` (otherwise the span ends "now",
       producing a near-zero-width span which most viewers still render).

    The `tracer` kwarg pins the sink to a specific tracer instance.
    When omitted, the sink resolves a tracer via
    `trace.get_tracer("harness")` against the global provider.

    See module docstring for the trace_id / parent_span_id fidelity
    contract and the rationale for not overriding the synthesized span's
    own span_id.
    """

    def __init__(self, tracer: Tracer | None = None) -> None:
        # Lazy import: importing this module must not require the [otel] extra.
        # Only construction does. Keeping the import inside __init__ also makes
        # the missing-extra test (which monkeypatches sys.modules) work — Python
        # raises ImportError on the next import attempt after a None entry, so
        # the import has to happen after the monkeypatch.
        try:
            from opentelemetry import trace as _trace
            from opentelemetry.trace import (
                NonRecordingSpan,
                SpanContext,
                TraceFlags,
                set_span_in_context,
            )
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "OpenTelemetrySink requires the [otel] extra. Install with: uv sync --extra otel"
            ) from exc

        self._trace = _trace
        self._SpanContext = SpanContext
        self._NonRecordingSpan = NonRecordingSpan
        self._TraceFlags = TraceFlags
        self._set_span_in_context = set_span_in_context
        # An explicit tracer pins the sink to that tracer (M1.7 lesson:
        # this kwarg used to be silently ignored; M3.5 makes it real).
        # Falling back to the global provider is the convenient default
        # for callers who wire OTel up elsewhere.
        self._tracer: Tracer = tracer if tracer is not None else _trace.get_tracer("harness")

    async def emit(self, event: TelemetryEvent) -> None:
        # OTel span timestamps are Unix-epoch nanoseconds. Without this
        # conversion the synthesized span would start at the moment
        # `start_as_current_span` runs, not the event's recorded
        # timestamp — defeating the point of a sink whose input already
        # has a timestamp.
        start_ns = int(event.timestamp.timestamp() * 1_000_000_000)

        # Seed the parent SpanContext from the harness correlation IDs.
        # When `trace_id` is absent (event emitted outside any
        # `session_scope`), context=None lets OTel pick up whatever
        # ambient context is active — graceful degradation matching the
        # pre-M3.5 "ride on current span" behavior.
        parent_context = self._build_parent_context(event)

        # `end_on_exit=False` so we can set an explicit end_time below;
        # without it the SDK ends the span "now" on context-manager exit.
        span_cm = self._tracer.start_as_current_span(
            name=event.kind,
            context=parent_context,
            start_time=start_ns,
            end_on_exit=False,
        )
        with span_cm as span:
            span.set_attributes(self._build_attributes(event))
            end_ns = self._compute_end_ns(event, start_ns)
            span.end(end_time=end_ns)

    def _build_parent_context(self, event: TelemetryEvent) -> Any:
        """Build a parent OTel context from the event's correlation IDs.

        Returns `None` if `trace_id` is absent — letting OTel fall back to
        the ambient context (a real upstream span, or `NonRecordingSpan`
        when nothing's wrapped).
        """
        if event.trace_id is None or event.span_id is None:
            return None

        try:
            trace_id_int = int(event.trace_id, 16)
            span_id_int = int(event.span_id, 16)
        except ValueError:
            # A non-hex correlation ID (e.g. caller passed a UUID with
            # dashes) can't seed a SpanContext. Fall back rather than
            # raise — the recorder's failure-isolation contract says a
            # sink must never crash an orchestrator turn.
            return None

        if trace_id_int == 0 or span_id_int == 0:
            # OTel treats zero IDs as invalid and silently drops the
            # context. Fall back to the ambient context.
            return None

        parent_sc = self._SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=True,
            trace_flags=self._TraceFlags(self._TraceFlags.SAMPLED),
        )
        return self._set_span_in_context(self._NonRecordingSpan(parent_sc))

    def _build_attributes(self, event: TelemetryEvent) -> dict[str, Any]:
        """Flatten the event's payload into `harness.*`-prefixed scalar attributes."""
        attributes: dict[str, Any] = {
            "harness.event_id": str(event.event_id),
            "harness.kind": event.kind,
        }
        for key, value in event.model_dump().items():
            if key in _RESERVED_FIELDS:
                continue
            attr_key = f"harness.{key}"
            if value is None:
                # OTel attributes don't accept None — the SDK logs a warning
                # and drops the attribute. Skip explicitly so the warning
                # stays out of the operator's stderr.
                continue
            if isinstance(value, str | int | float | bool):
                attributes[attr_key] = value
            else:
                # OTel attribute values must be scalar (or homogeneous sequences
                # of scalars). Stringify anything else so exporters never choke.
                attributes[attr_key] = str(value)
        return attributes

    @staticmethod
    def _compute_end_ns(event: TelemetryEvent, start_ns: int) -> int:
        """Compute the span end_time in ns, honoring `duration_ms` if present.

        Events carrying their own duration (`ToolDispatched`,
        `OrchestratorTurn`) produce spans of realistic width in viewers.
        Events without a duration get a one-nanosecond span — a no-width
        marker that some viewers render as a point.
        """
        duration_ms = getattr(event, "duration_ms", None)
        if isinstance(duration_ms, int | float) and duration_ms >= 0:
            return start_ns + int(duration_ms * 1_000_000)
        # +1 so end > start; some exporters reject equal timestamps.
        return start_ns + 1


__all__ = ["OpenTelemetrySink"]
