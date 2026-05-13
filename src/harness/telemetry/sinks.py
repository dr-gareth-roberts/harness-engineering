from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import TracebackType
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

    Backed by a path: opens the file in append mode on the first emit and
    holds the handle for the sink's lifetime. POSIX `O_APPEND` makes single
    writes atomic for typical event sizes (up to `PIPE_BUF`: 512B on macOS,
    4096B on Linux), and a per-instance `asyncio.Lock` guards against
    interleaved writes when multiple coroutines emit concurrently
    (e.g. `Orchestrator.run_parallel`).

    Holding the handle within a single process is safe — every event is
    written and flushed under the lock, so partial-line tearing cannot
    occur. Cross-process / cross-machine concurrency is out of scope:
    multiple writers to the same path still race per the POSIX caveat
    above. Wrap with a real queue or rotation strategy if you need that.

    Lifecycle:

    - Lazy open: the file is opened on the first `emit`, so a sink that
      is constructed but never emitted does not touch the filesystem.
    - `close()` releases the handle. It is idempotent and safe to call
      multiple times. After `close()`, a subsequent `emit` reopens the
      file in append mode, preserving the "always-durable" contract.
    - `async with JSONLSink(p) as sink:` releases the handle on exit.
    - When constructed with a caller-owned stream (`JSONLSink(stream)`),
      `close()` is a true no-op: the caller retains ownership of the
      underlying stream.
    """

    def __init__(self, target: TextIO | Path | str) -> None:
        self._path: Path | None
        self._stream: TextIO | None
        self._owns_handle: bool
        if isinstance(target, str | Path):
            self._path = Path(target)
            self._stream = None
            self._owns_handle = True
        else:
            self._path = None
            self._stream = target
            self._owns_handle = False
        self._lock = asyncio.Lock()

    async def emit(self, event: TelemetryEvent) -> None:
        line = event.model_dump_json() + "\n"
        async with self._lock:
            if self._stream is None:
                # Lazy-open path-backed handles on first emit (or after close).
                # Guarded by the lock so concurrent first-emits don't race.
                assert self._path is not None
                self._stream = self._path.open("a", encoding="utf-8")
            self._stream.write(line)
            self._stream.flush()

    async def close(self) -> None:
        """Close the file handle if owned. Idempotent.

        For path-backed sinks, the next `emit` will reopen the file. For
        caller-owned streams, this is a no-op — the caller retains
        ownership and is responsible for closing the stream.
        """
        async with self._lock:
            if self._owns_handle and self._stream is not None:
                self._stream.close()
                self._stream = None

    async def __aenter__(self) -> JSONLSink:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


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
