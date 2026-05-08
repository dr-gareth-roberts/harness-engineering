from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol, TextIO

from harness.telemetry.events import TelemetryEvent

logger = logging.getLogger(__name__)


class Sink(Protocol):
    async def emit(self, event: TelemetryEvent) -> None: ...


class NullSink:
    async def emit(self, event: TelemetryEvent) -> None:
        return None


class MemorySink:
    """Appends events to an in-memory list. Useful for tests and inspection."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []
        self._lock = asyncio.Lock()

    async def emit(self, event: TelemetryEvent) -> None:
        async with self._lock:
            self.events.append(event)


class JSONLSink:
    """Writes one JSON line per event to a path or an open text stream.

    Backed by a path: opens in append mode for each emit. POSIX `O_APPEND`
    makes single writes atomic for typical event sizes, and a per-instance
    `asyncio.Lock` guards against interleaved writes when multiple
    coroutines emit concurrently (e.g. `Orchestrator.run_parallel`).

    Cross-process / cross-machine concurrency is out of scope — wrap with
    a real queue or rotation strategy if you need that.
    """

    def __init__(self, target: TextIO | Path | str) -> None:
        self._path: Path | None
        self._stream: TextIO | None
        if isinstance(target, str | Path):
            self._path = Path(target)
            self._stream = None
        else:
            self._path = None
            self._stream = target
        self._lock = asyncio.Lock()

    async def emit(self, event: TelemetryEvent) -> None:
        line = event.model_dump_json() + "\n"
        async with self._lock:
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
            else:
                assert self._stream is not None
                self._stream.write(line)
                self._stream.flush()


class MultiSink:
    """Fan-out wrapper. One failing sink does not stop the others."""

    def __init__(self, *sinks: Sink) -> None:
        self._sinks: tuple[Sink, ...] = sinks

    async def emit(self, event: TelemetryEvent) -> None:
        for sink in self._sinks:
            try:
                await sink.emit(event)
            except Exception:
                logger.warning("telemetry sink %r failed", sink, exc_info=True)
