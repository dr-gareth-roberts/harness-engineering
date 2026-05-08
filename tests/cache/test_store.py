"""Tests for `harness.cache.store` — fingerprint persistence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from harness.cache.store import (
    FileFingerprintStore,
    FingerprintRecord,
    InMemoryFingerprintStore,
)


def _make_record(
    *, breakpoint: int = 0, hash_: str = "abc", offset_hours: float = 0
) -> FingerprintRecord:
    return FingerprintRecord(
        timestamp=datetime.now(UTC) - timedelta(hours=offset_hours),
        breakpoint_index=breakpoint,
        hash=hash_,
        full_prompt=None,
    )


# Test 4: FingerprintStore round-trips records.
async def test_in_memory_store_round_trips_record() -> None:
    store = InMemoryFingerprintStore()
    record = _make_record(breakpoint=2, hash_="deadbeef")
    await store.append(record)

    recent = [r async for r in store.iter_recent(since=datetime.now(UTC) - timedelta(hours=1))]
    assert len(recent) == 1
    assert recent[0].breakpoint_index == 2
    assert recent[0].hash == "deadbeef"


async def test_file_store_round_trips_record_through_disk(tmp_path: Path) -> None:
    store = FileFingerprintStore(tmp_path)
    record = FingerprintRecord(
        timestamp=datetime.now(UTC),
        breakpoint_index=1,
        hash="cafef00d",
        full_prompt='{"system": "be helpful"}',
    )
    await store.append(record)

    # Re-open the same path; the second instance must see what the first wrote.
    store2 = FileFingerprintStore(tmp_path)
    recent = [r async for r in store2.iter_recent(since=datetime.now(UTC) - timedelta(hours=1))]
    assert len(recent) == 1
    assert recent[0].breakpoint_index == 1
    assert recent[0].hash == "cafef00d"
    assert recent[0].full_prompt == '{"system": "be helpful"}'


async def test_iter_recent_filters_by_window(tmp_path: Path) -> None:
    store = FileFingerprintStore(tmp_path)
    await store.append(_make_record(breakpoint=0, hash_="old", offset_hours=48))
    await store.append(_make_record(breakpoint=0, hash_="new", offset_hours=0))

    cutoff = datetime.now(UTC) - timedelta(hours=24)
    recent = [r async for r in store.iter_recent(since=cutoff)]
    assert [r.hash for r in recent] == ["new"]


async def test_in_memory_store_concurrent_appends_preserve_count() -> None:
    """Eight concurrent appends should yield exactly eight records — the
    `asyncio.Lock` prevents lost writes."""
    store = InMemoryFingerprintStore()
    records = [_make_record(breakpoint=i, hash_=f"h{i}") for i in range(8)]
    await asyncio.gather(*(store.append(r) for r in records))

    recent = [r async for r in store.iter_recent(since=datetime.now(UTC) - timedelta(hours=1))]
    assert len(recent) == 8
    assert {r.breakpoint_index for r in recent} == set(range(8))


async def test_file_store_accepts_explicit_jsonl_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.jsonl"
    store = FileFingerprintStore(explicit)
    await store.append(_make_record(hash_="fixed"))
    assert explicit.exists()
    assert "fixed" in explicit.read_text(encoding="utf-8")


async def test_file_store_tolerates_missing_path_until_first_write(tmp_path: Path) -> None:
    store = FileFingerprintStore(tmp_path / "fresh")
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    # No writes yet — iteration on a non-existent file must not crash.
    recent = [r async for r in store.iter_recent(since=cutoff)]
    assert recent == []
