from __future__ import annotations

import logging

from harness.telemetry.events import TelemetryEvent
from harness.telemetry.sinks import NullSink, Sink

logger = logging.getLogger(__name__)


class Telemetry:
    """Central recorder. Wraps a single Sink and isolates failures.

    The recorder never raises — sink errors are logged at WARNING and
    swallowed so a misbehaving sink can never crash an orchestrator turn
    or a tool dispatch. Pair with `MultiSink` to fan out to multiple
    backends.
    """

    def __init__(self, sink: Sink | None = None) -> None:
        self._sink: Sink = sink if sink is not None else NullSink()

    async def emit(self, event: TelemetryEvent) -> None:
        try:
            await self._sink.emit(event)
        except Exception:
            logger.warning("telemetry sink %r failed", self._sink, exc_info=True)
