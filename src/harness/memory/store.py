from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Protocol

from harness.memory.record import SessionRecord


class MemoryStore(Protocol):
    async def save(self, record: SessionRecord) -> None: ...
    async def load(self, session_id: str) -> SessionRecord | None: ...
    async def list(self, *, limit: int = 100) -> list[SessionRecord]:
        """Return up to ``limit`` records.

        Records MUST be returned ordered by ``updated_at`` descending (most
        recent first). Implementations that cannot guarantee recency ordering
        should return ALL records (ignore ``limit``) and let callers sort.

        Callers such as :class:`harness.speculate.cross_session.CrossSessionPredictor`
        depend on this recency contract to derive a "K most recent" slice
        without re-sorting.
        """
        ...

    async def delete(self, session_id: str) -> bool: ...


class InMemoryStore:
    """Dict-backed store. Deep-copies records on save and load so caller
    mutations never bleed into stored state."""

    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()

    async def save(self, record: SessionRecord) -> None:
        async with self._lock:
            self._records[record.session_id] = record.model_copy(deep=True)

    async def load(self, session_id: str) -> SessionRecord | None:
        async with self._lock:
            stored = self._records.get(session_id)
        return stored.model_copy(deep=True) if stored is not None else None

    async def list(self, *, limit: int = 100) -> list[SessionRecord]:
        async with self._lock:
            snapshot = [r.model_copy(deep=True) for r in self._records.values()]
        # Contract: recency-ordered (see MemoryStore.list).
        snapshot.sort(key=lambda r: r.updated_at, reverse=True)
        return snapshot[:limit]

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._records.pop(session_id, None) is not None


class FileStore:
    """One JSON file per session under `root`. Atomic writes prevent torn reads.

    `os.replace()` is atomic on POSIX (and on Windows for same-volume,
    no-open-handle paths). This means a concurrent reader will see either the
    previous record or the new one, never a half-written file. Without
    `fsync()` calls this is not full crash safety — on a hard power loss
    either version may be the survivor — but it does cover process crashes
    and SIGKILL during writes.

    **Concurrent / cross-process writes**: this store writes a per-session
    temp file (`<id>.json.tmp`) then atomically renames it via `os.replace()`.
    Single-process multi-threaded access is safe: an `asyncio.Lock` serializes
    writes, and `os.replace()` guarantees readers never observe a half-written
    file. Cross-process writers of the same `session_id` race on the shared
    tmp path and on `os.replace` ordering; the failure mode is **lost update**,
    not torn read — the surviving file is always a complete record from one
    writer, but the other writer's update may be silently overwritten. This is
    a different profile from :class:`harness.cache.store.FileFingerprintStore`,
    which appends and risks interleaving instead. If you need cross-process
    safety, wrap `save` with an external file-lock (e.g. `fcntl.flock`) or
    restrict to a single writer per `session_id`.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    # Characters that must never appear in a session_id, mapped to the class
    # that motivates the rejection. Keys are single chars; values are short
    # labels surfaced in the error message so operators can see why the id
    # was rejected without having to read source.
    _UNSAFE_CHARS: dict[str, str] = {
        "/": "path separator",
        "\\": "path separator",
        ":": "path/drive separator",
        ";": "shell metacharacter",
        "\n": "control character",
        "\r": "control character",
        "\x00": "null byte",
    }

    def _path_for(self, session_id: str) -> Path:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if session_id.startswith("."):
            raise ValueError(f"unsafe session_id: {session_id!r} (disallowed leading '.')")
        for ch in session_id:
            label = self._UNSAFE_CHARS.get(ch)
            if label is not None:
                raise ValueError(
                    f"unsafe session_id: {session_id!r} (disallowed character {ch!r}: {label})"
                )
            if ord(ch) < 32:
                raise ValueError(
                    f"unsafe session_id: {session_id!r} "
                    f"(disallowed character {ch!r}: control character)"
                )
        return self._root / f"{session_id}.json"

    async def save(self, record: SessionRecord) -> None:
        path = self._path_for(record.session_id)
        tmp = path.parent / (path.name + ".tmp")
        async with self._lock:
            tmp.write_text(record.model_dump_json(), encoding="utf-8")
            os.replace(tmp, path)

    async def load(self, session_id: str) -> SessionRecord | None:
        path = self._path_for(session_id)
        async with self._lock:
            if not path.is_file():
                return None
            text = path.read_text(encoding="utf-8")
        return SessionRecord.model_validate_json(text)

    async def list(self, *, limit: int = 100) -> list[SessionRecord]:
        async with self._lock:
            paths = [
                p
                for p in self._root.iterdir()
                if p.is_file() and p.suffix == ".json" and not p.name.endswith(".tmp")
            ]
            records = [
                SessionRecord.model_validate_json(p.read_text(encoding="utf-8")) for p in paths
            ]
        # Contract: recency-ordered (see MemoryStore.list).
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records[:limit]

    async def delete(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        async with self._lock:
            if not path.is_file():
                return False
            path.unlink()
            return True
