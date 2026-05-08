from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from harness.agents import SubAgent
from harness.memory import FileStore, InMemoryStore, MemoryStore, SessionRecord
from harness.prompts import text

StoreFactory = Callable[[], Awaitable[MemoryStore]]


def make_record(session_id: str, **overrides: Any) -> SessionRecord:
    base: dict[str, Any] = {
        "session_id": session_id,
        "agent": SubAgent(name="x", system_prompt="hi", model="test-model"),
        "messages": [text("user", "hello")],
    }
    base.update(overrides)
    return SessionRecord(**base)


@pytest.fixture
def in_memory_factory() -> StoreFactory:
    async def _make() -> MemoryStore:
        return InMemoryStore()

    return _make


@pytest.fixture
def file_factory(tmp_path: Path) -> StoreFactory:
    async def _make() -> MemoryStore:
        return FileStore(tmp_path)

    return _make


@pytest.fixture(params=["in_memory", "file"])
def store_factory(request: pytest.FixtureRequest) -> StoreFactory:
    factory: StoreFactory = request.getfixturevalue(f"{request.param}_factory")
    return factory


# ---------------------------------------------------------------------------
# Core CRUD


async def test_save_then_load_returns_equivalent_record(store_factory: StoreFactory) -> None:
    store = await store_factory()
    record = make_record("s1")
    await store.save(record)
    loaded = await store.load("s1")

    assert loaded is not None
    assert loaded.session_id == "s1"
    assert [m.content[0].text for m in loaded.messages] == ["hello"]


async def test_load_missing_returns_none(store_factory: StoreFactory) -> None:
    store = await store_factory()
    assert await store.load("no-such-id") is None


async def test_list_returns_sorted_by_updated_at_desc(store_factory: StoreFactory) -> None:
    store = await store_factory()
    now = datetime.now(UTC)
    for i in range(3):
        await store.save(
            make_record(f"s{i}", updated_at=now - timedelta(hours=i)),
        )

    listed = await store.list()
    assert [r.session_id for r in listed] == ["s0", "s1", "s2"]


async def test_list_respects_limit(store_factory: StoreFactory) -> None:
    store = await store_factory()
    for i in range(5):
        await store.save(make_record(f"s{i}"))
    listed = await store.list(limit=2)
    assert len(listed) == 2


async def test_delete_returns_true_then_false(store_factory: StoreFactory) -> None:
    store = await store_factory()
    await store.save(make_record("s1"))
    assert await store.delete("s1") is True
    assert await store.delete("s1") is False
    assert await store.load("s1") is None


# ---------------------------------------------------------------------------
# Isolation


async def test_load_returns_isolated_copy(store_factory: StoreFactory) -> None:
    store = await store_factory()
    await store.save(make_record("s1"))

    loaded = await store.load("s1")
    assert loaded is not None
    loaded.messages.append(text("user", "mutated"))
    loaded.metadata["mutated"] = True

    re_loaded = await store.load("s1")
    assert re_loaded is not None
    assert len(re_loaded.messages) == 1
    assert re_loaded.metadata == {}


# ---------------------------------------------------------------------------
# Concurrency


async def test_concurrent_save_distinct_ids(store_factory: StoreFactory) -> None:
    store = await store_factory()
    records = [make_record(f"s{i}") for i in range(8)]
    await asyncio.gather(*(store.save(r) for r in records))

    assert {r.session_id for r in await store.list()} == {f"s{i}" for i in range(8)}


async def test_concurrent_save_same_id_yields_one_winner(store_factory: StoreFactory) -> None:
    """Eight concurrent saves with the same session_id must produce one of the
    eight inputs verbatim — never a torn or merged record."""
    store = await store_factory()
    distinct_metadata = [{"variant": i} for i in range(8)]
    records = [make_record("contest", metadata=m) for m in distinct_metadata]

    await asyncio.gather(*(store.save(r) for r in records))

    final = await store.load("contest")
    assert final is not None
    assert final.metadata in distinct_metadata


# ---------------------------------------------------------------------------
# FileStore-specific


async def test_filestore_rejects_unsafe_session_id(tmp_path: Path) -> None:
    store = FileStore(tmp_path)

    for unsafe in ("../etc", "a/b", r"a\b", ".hidden", ""):
        with pytest.raises(ValueError):
            await store.save(make_record(unsafe))


async def test_filestore_ignores_stray_tmp_files(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    await store.save(make_record("s1"))
    (tmp_path / "garbage.json.tmp").write_text("not json", encoding="utf-8")

    listed = await store.list()
    assert [r.session_id for r in listed] == ["s1"]
