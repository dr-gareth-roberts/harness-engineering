"""Tests for `harness.cache.audit` — drift detection + diff rendering + CLI snapshot.

Covers spec tests 5, 6, 7, 9.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from harness.cache.audit import audit
from harness.cache.cli import _cmd_cache_audit, _parse_duration, register
from harness.cache.store import (
    FileFingerprintStore,
    FingerprintRecord,
    InMemoryFingerprintStore,
)
from harness.cache.watcher import PrefixWatcher


def _record(
    *, ts: datetime, bp: int = 0, hash_: str = "h", full_prompt: str | None = None
) -> FingerprintRecord:
    return FingerprintRecord(timestamp=ts, breakpoint_index=bp, hash=hash_, full_prompt=full_prompt)


# Test 5: no drift events when all hashes for a breakpoint are equal.
async def test_audit_returns_no_drift_when_hashes_are_equal() -> None:
    store = InMemoryFingerprintStore()
    base = datetime.now(UTC) - timedelta(minutes=30)
    for i in range(4):
        await store.append(
            _record(ts=base + timedelta(minutes=i), bp=0, hash_="same", full_prompt="<body>")
        )

    report = await audit(store, window_hours=24)
    assert report.drift_events == []
    assert report.stable_prefixes == [0]


# Test 6: audit returns DriftEvent with diff when a hash changes.
async def test_audit_returns_drift_event_with_diff_when_hash_changes() -> None:
    store = InMemoryFingerprintStore()
    base = datetime.now(UTC) - timedelta(minutes=30)
    await store.append(
        _record(
            ts=base,
            bp=0,
            hash_="abc",
            full_prompt='{"system": "old"}',
        )
    )
    await store.append(
        _record(
            ts=base + timedelta(minutes=1),
            bp=0,
            hash_="def",
            full_prompt='{"system": "new"}',
        )
    )

    report = await audit(store, window_hours=24)
    assert len(report.drift_events) == 1
    event = report.drift_events[0]
    assert event.breakpoint_index == 0
    assert event.before_hash == "abc"
    assert event.after_hash == "def"
    # Diff body must reference both prompts.
    assert "old" in event.diff
    assert "new" in event.diff
    # No stable_prefixes — breakpoint 0 drifted.
    assert report.stable_prefixes == []


async def test_audit_orders_records_by_timestamp_within_breakpoint() -> None:
    """The store doesn't promise order; audit must sort before walking."""
    store = InMemoryFingerprintStore()
    base = datetime.now(UTC) - timedelta(minutes=30)
    # Append out of order.
    await store.append(
        _record(ts=base + timedelta(minutes=2), bp=0, hash_="C", full_prompt="three")
    )
    await store.append(_record(ts=base, bp=0, hash_="A", full_prompt="one"))
    await store.append(_record(ts=base + timedelta(minutes=1), bp=0, hash_="B", full_prompt="two"))

    report = await audit(store, window_hours=24)
    # Two transitions: A->B then B->C, in time order.
    assert len(report.drift_events) == 2
    assert report.drift_events[0].before_hash == "A"
    assert report.drift_events[0].after_hash == "B"
    assert report.drift_events[1].before_hash == "B"
    assert report.drift_events[1].after_hash == "C"


# Test 7: diff captures the actual changed lines (the timestamp/uuid regression).
async def test_audit_diff_surfaces_changed_timestamp_line() -> None:
    store = InMemoryFingerprintStore()
    base = datetime.now(UTC) - timedelta(minutes=30)
    await store.append(
        _record(
            ts=base,
            bp=0,
            hash_="h1",
            full_prompt='"current_time": "2026-04-12T13:58:11Z"\n"user_id": "u_abc"',
        )
    )
    await store.append(
        _record(
            ts=base + timedelta(minutes=1),
            bp=0,
            hash_="h2",
            full_prompt='"current_time": "2026-04-12T14:03:42Z"\n"user_id": "u_abc"',
        )
    )

    report = await audit(store, window_hours=24)
    assert len(report.drift_events) == 1
    diff = report.drift_events[0].diff
    # Both timestamps appear in the diff body (one as `-`, one as `+`).
    assert "2026-04-12T13:58:11Z" in diff
    assert "2026-04-12T14:03:42Z" in diff
    # And the hint heuristic flags the timestamp.
    assert report.drift_events[0].hint is not None
    assert "timestamp" in report.drift_events[0].hint


async def test_audit_diff_surfaces_changed_uuid_line() -> None:
    store = InMemoryFingerprintStore()
    base = datetime.now(UTC) - timedelta(minutes=30)
    await store.append(
        _record(
            ts=base,
            bp=0,
            hash_="h1",
            full_prompt='request_id="11111111-2222-3333-4444-555555555555"',
        )
    )
    await store.append(
        _record(
            ts=base + timedelta(minutes=1),
            bp=0,
            hash_="h2",
            full_prompt='request_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"',
        )
    )

    report = await audit(store, window_hours=24)
    assert len(report.drift_events) == 1
    assert report.drift_events[0].hint is not None
    assert "uuid" in report.drift_events[0].hint


async def test_audit_window_excludes_old_records() -> None:
    store = InMemoryFingerprintStore()
    old_ts = datetime.now(UTC) - timedelta(hours=48)
    await store.append(_record(ts=old_ts, bp=0, hash_="old"))
    new_ts = datetime.now(UTC) - timedelta(minutes=5)
    await store.append(_record(ts=new_ts, bp=0, hash_="new"))

    # 24-hour window: only the 5-minute-old record is in scope, so no
    # drift transition (single hash within window).
    report = await audit(store, window_hours=24)
    assert report.drift_events == []
    assert report.stable_prefixes == [0]


async def test_audit_backfills_before_prompt_from_earlier_record() -> None:
    """`full_capture="on_drift"` only stores the prompt on the *first*
    sighting and on each drift. When two stable observations precede a
    drift, audit must pull the "before" body from the earliest record.
    """
    store = InMemoryFingerprintStore()
    base = datetime.now(UTC) - timedelta(minutes=30)
    await store.append(_record(ts=base, bp=0, hash_="abc", full_prompt='"before"'))
    await store.append(_record(ts=base + timedelta(minutes=1), bp=0, hash_="abc", full_prompt=None))
    await store.append(
        _record(ts=base + timedelta(minutes=2), bp=0, hash_="def", full_prompt='"after"')
    )

    report = await audit(store, window_hours=24)
    assert len(report.drift_events) == 1
    diff = report.drift_events[0].diff
    assert "before" in diff
    assert "after" in diff


# ---------------------------------------------------------------------------
# CLI: parsing, registration, and the spec's snapshot test.


def test_parse_duration_handles_h_d_w_m() -> None:
    assert _parse_duration("24h") == 24.0
    assert _parse_duration("7d") == 168.0
    assert _parse_duration("1w") == 168.0
    assert _parse_duration("30m") == 0.5


def test_parse_duration_rejects_unknown_units() -> None:
    with pytest.raises(ValueError):
        _parse_duration("24x")
    with pytest.raises(ValueError):
        _parse_duration("nope")


def test_register_adds_cache_audit_subcommand() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    register(sub)
    # Parsing the subcommand shouldn't blow up.
    args = parser.parse_args(["cache-audit", "--store", "/tmp/x"])
    assert args.func is _cmd_cache_audit
    assert args.store == "/tmp/x"
    assert args.since == "24h"


# Test 9: CLI emits useful output for the common "timestamp leak" case.
async def test_cli_cmd_cache_audit_snapshot_for_timestamp_leak(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FileFingerprintStore(tmp_path)
    watcher = PrefixWatcher(store, full_capture="on_drift")

    for stamp in ("2026-04-12T13:58:11Z", "2026-04-12T14:03:42Z"):
        request: dict[str, Any] = {
            "model": "claude-opus-4-7",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": f'now is "{stamp}"',
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        }
        await watcher.fingerprint(request)

    # `_cmd_cache_audit` is sync (it uses `asyncio.run` internally); we
    # call it directly with a hand-built Namespace to avoid spawning a
    # subprocess.
    namespace = argparse.Namespace(store=str(tmp_path), since="24h")
    rc = await asyncio.to_thread(_cmd_cache_audit, namespace)

    assert rc == 0
    output = capsys.readouterr().out
    assert "DRIFT detected at breakpoint 0" in output
    assert "Hint:" in output
    assert "timestamp" in output
    # Diff body shows up under the "Diff (first 20 changed lines)" header.
    assert "Diff" in output
    assert "2026-04-12T13:58:11Z" in output
    assert "2026-04-12T14:03:42Z" in output


def test_cli_cmd_cache_audit_reports_no_drift_for_empty_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    namespace = argparse.Namespace(store=str(tmp_path), since="24h")
    rc = _cmd_cache_audit(namespace)
    assert rc == 0
    output = capsys.readouterr().out
    assert "no drift events" in output
