from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Protocol

from harness.memory.record import SessionRecord


class MemoryStore(Protocol):
    async def save(self, record: SessionRecord) -> None: ...
    async def load(self, session_id: str) -> SessionRecord | None: ...
    async def list(self, *, limit: int = 100) -> list[SessionRecord]: ...
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
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path_for(self, session_id: str) -> Path:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if "/" in session_id or "\\" in session_id or session_id.startswith("."):
            raise ValueError(f"unsafe session_id: {session_id!r}")
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
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records[:limit]

    async def delete(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        async with self._lock:
            if not path.is_file():
                return False
            path.unlink()
            return True
