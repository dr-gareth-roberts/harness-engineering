"""Fingerprint storage backends for the prefix-drift watcher.

Mirrors `harness.memory.store`'s `Protocol + InMemory + File` triple. A
`FingerprintRecord` captures one breakpoint's hash at one point in time;
the optional `full_prompt` field carries the JSON-serialized prompt
segment when the watcher's `full_capture` policy says to keep it.

`FileFingerprintStore` writes one JSON line per record (append-only),
matching `harness.telemetry.JSONLSink`'s on-disk format. The file IO is
synchronous inside `async` methods rather than reaching for `aiofiles`
to keep the dependency footprint at zero — the volumes here are tiny
(one record per model call) and the calling code is already inside
`asyncio.Lock` to prevent torn writes from concurrent emitters.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class FingerprintRecord(BaseModel):
    """One observation: the hash of a single breakpoint at a single moment.

    Ordering is by `timestamp` — within the same breakpoint, two records
    with the same hash mean "the prefix held stable across two calls"; a
    differing hash on adjacent records is what `audit` reports as drift.

    `full_prompt` is the JSON-serialized prompt segment (the same bytes
    that were hashed). It's only populated when the watcher's
    `full_capture` policy decided to keep it — tests for `"never"`
    asserts it's always `None`, tests for `"on_drift"` asserts it's
    populated only when the segment differs from the previous one.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    breakpoint_index: int
    hash: str
    full_prompt: str | None = None


class FingerprintStore(Protocol):
    """Write-once / read-many store of `FingerprintRecord`s.

    `iter_recent` returns records whose timestamp is at or after `since`;
    no ordering is promised — callers (like `audit`) sort by timestamp
    themselves. The async-iterator return type lets a future
    network-backed store stream rather than materialize the whole window.
    """

    async def append(self, record: FingerprintRecord) -> None: ...
    def iter_recent(self, *, since: datetime) -> AsyncIterator[FingerprintRecord]: ...


class InMemoryFingerprintStore:
    """List-backed store. Useful for tests and for in-process use where
    persistence across restarts isn't required."""

    def __init__(self) -> None:
        self._records: list[FingerprintRecord] = []
        self._lock = asyncio.Lock()

    async def append(self, record: FingerprintRecord) -> None:
        async with self._lock:
            self._records.append(record)

    async def iter_recent(self, *, since: datetime) -> AsyncIterator[FingerprintRecord]:
        async with self._lock:
            snapshot = list(self._records)
        for record in snapshot:
            if record.timestamp >= since:
                yield record


class FileFingerprintStore:
    """JSONL-backed store: one record per line, append-only.

    Pass either a path to a `.jsonl` file or a directory; in the
    directory case we use `<dir>/fingerprints.jsonl`. Reading is
    streaming (one `json.loads` per line); writes are guarded by a
    per-instance `asyncio.Lock` so concurrent appends from the same
    process don't interleave at the bytes-on-disk level.

    **Concurrent / cross-process writes**: this store opens the file in append
    mode, which gives `O_APPEND` atomicity for writes up to `PIPE_BUF` (512
    bytes on macOS, 4096 on Linux). Single-process multi-threaded access is
    safe: an `asyncio.Lock` serializes writes, and `iter_recent` tolerates a
    half-written final line. Across processes, records smaller than `PIPE_BUF`
    append atomically; larger records may interleave bytes from concurrent
    writers and corrupt the affected lines (`iter_recent` skips unparseable
    JSON, but the data is lost). This is a different profile from
    :class:`harness.memory.store.FileStore`, which atomically renames whole
    files and risks lost-update instead. If you need cross-process safety for
    large records, wrap `append` with an external file-lock (e.g.
    `fcntl.flock`) or restrict to a single writer per file. Cross-machine
    consistency is out of scope.
    """

    def __init__(self, path: Path | str) -> None:
        target = Path(path)
        if target.is_dir() or (not target.exists() and target.suffix == ""):
            target.mkdir(parents=True, exist_ok=True)
            target = target / "fingerprints.jsonl"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        self._path = target
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def append(self, record: FingerprintRecord) -> None:
        line = record.model_dump_json() + "\n"
        async with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

    async def iter_recent(self, *, since: datetime) -> AsyncIterator[FingerprintRecord]:
        async with self._lock:
            if not self._path.is_file():
                return
            lines = self._path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                # Tolerate a half-written final line from a concurrent writer.
                continue
            record = FingerprintRecord.model_validate(payload)
            if record.timestamp >= since:
                yield record
