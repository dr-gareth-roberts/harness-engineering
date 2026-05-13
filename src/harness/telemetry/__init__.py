from typing import TYPE_CHECKING, Any

from harness.telemetry.events import OrchestratorTurn, TelemetryEvent, ToolDispatched
from harness.telemetry.recorder import Redactor, Telemetry
from harness.telemetry.sinks import JSONLSink, MemorySink, MultiSink, NullSink, Sink

if TYPE_CHECKING:
    from harness.telemetry.otel import OpenTelemetrySink

__all__ = [
    "JSONLSink",
    "MemorySink",
    "MultiSink",
    "NullSink",
    "OpenTelemetrySink",
    "OrchestratorTurn",
    "Redactor",
    "Sink",
    "Telemetry",
    "TelemetryEvent",
    "ToolDispatched",
]


def __getattr__(name: str) -> Any:
    """Lazy-load OpenTelemetrySink so the [otel] extra is opt-in.

    Importing `harness.telemetry` succeeds without `opentelemetry-api`
    installed; only `from harness.telemetry import OpenTelemetrySink`
    triggers the (lazily-guarded) import that may raise `ImportError`.
    Mirrors the pattern `harness.runner` uses for AnthropicRunner /
    OpenAICompatRunner.
    """
    if name == "OpenTelemetrySink":
        from harness.telemetry.otel import OpenTelemetrySink

        return OpenTelemetrySink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
