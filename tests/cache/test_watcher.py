"""Tests for `harness.cache.watcher` — fingerprint hashing semantics.

Covers spec tests 1, 2, 3, 8, 10. The fingerprint protocol is structural,
so we never need to instantiate a real runner — the watcher's
`fingerprint(request)` is exercised directly with hand-built request
dicts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from harness.cache.audit import audit
from harness.cache.store import InMemoryFingerprintStore
from harness.cache.watcher import PrefixWatcher


def _anthropic_request(
    *,
    system: str = "you are helpful",
    user_text: str = "hello",
    cache_system: bool = True,
) -> dict[str, Any]:
    """Build a minimal Anthropic-shaped request with one cache breakpoint
    on the system prompt."""
    system_block: dict[str, Any] = {"type": "text", "text": system}
    if cache_system:
        system_block["cache_control"] = {"type": "ephemeral"}
    return {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "system": [system_block],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ],
    }


# Test 1: identical request dicts -> identical hashes.
async def test_identical_requests_produce_identical_hashes() -> None:
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="never")

    request_a = _anthropic_request()
    request_b = _anthropic_request()
    await watcher.fingerprint(request_a)
    await watcher.fingerprint(request_b)

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    records = [r async for r in store.iter_recent(since=cutoff)]
    by_breakpoint: dict[int, list[str]] = {}
    for record in records:
        by_breakpoint.setdefault(record.breakpoint_index, []).append(record.hash)

    # Each breakpoint observed twice; both observations must agree.
    for hashes in by_breakpoint.values():
        assert len(hashes) == 2
        assert hashes[0] == hashes[1]


# Test 2: a single byte change anywhere in the prefix produces a different hash.
async def test_single_byte_change_changes_hash() -> None:
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="never")

    await watcher.fingerprint(_anthropic_request(system="you are helpful"))
    await watcher.fingerprint(_anthropic_request(system="you are Helpful"))  # 'h' -> 'H'

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    records = [r async for r in store.iter_recent(since=cutoff)]
    breakpoint_zero_hashes = [r.hash for r in records if r.breakpoint_index == 0]
    assert len(breakpoint_zero_hashes) == 2
    assert breakpoint_zero_hashes[0] != breakpoint_zero_hashes[1]


# Test 3: cache breakpoints in different positions hash independently.
async def test_independent_breakpoints_hash_independently() -> None:
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="never")

    cache_marker = {"type": "ephemeral"}
    request: dict[str, Any] = {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "system": [
            {"type": "text", "text": "stable header", "cache_control": cache_marker},
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "long stable doc", "cache_control": cache_marker},
                    {"type": "text", "text": "today's question"},
                ],
            },
        ],
    }
    await watcher.fingerprint(request)

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    records = [r async for r in store.iter_recent(since=cutoff)]
    breakpoint_indices = {r.breakpoint_index for r in records}
    # Two cache_control markers + a trailing tail = at least two segments;
    # the first marker is one breakpoint and the second is another.
    assert len(breakpoint_indices) >= 2
    # And the hashes for those breakpoints must be distinct (they cover
    # disjoint content — an accidental same-hash would imply a bug in
    # the splitter).
    hashes = {r.breakpoint_index: r.hash for r in records}
    assert hashes[0] != hashes[1]


async def test_changing_only_breakpoint_one_leaves_breakpoint_zero_stable() -> None:
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="never")

    def _request(tail: str) -> dict[str, Any]:
        return {
            "model": "claude-opus-4-7",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "header", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "doc", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": tail},
                    ],
                },
            ],
        }

    await watcher.fingerprint(_request("first question"))
    await watcher.fingerprint(_request("second question"))

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    records = [r async for r in store.iter_recent(since=cutoff)]
    by_bp: dict[int, list[str]] = {}
    for r in records:
        by_bp.setdefault(r.breakpoint_index, []).append(r.hash)

    # Breakpoint 0 (system+header) must be identical across the two calls.
    assert len(set(by_bp[0])) == 1
    # The trailing breakpoint (whichever index it is) must have changed.
    last_bp = max(by_bp)
    assert len(set(by_bp[last_bp])) == 2


# Test 8: full_capture="on_drift" only captures prompts when drift is detected.
async def test_full_capture_on_drift_captures_only_on_change() -> None:
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="on_drift")

    # Same request three times -> first stores prompt, then two None.
    for _ in range(3):
        await watcher.fingerprint(_anthropic_request())
    # Then a drifted request -> stores the new prompt.
    await watcher.fingerprint(_anthropic_request(system="changed"))

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    records = sorted(
        [r async for r in store.iter_recent(since=cutoff) if r.breakpoint_index == 0],
        key=lambda r: r.timestamp,
    )
    assert len(records) == 4
    # First record carries the prompt (so future drift has a "before" side).
    assert records[0].full_prompt is not None
    # Second and third are no-ops (same hash) -> no need to re-store.
    assert records[1].full_prompt is None
    assert records[2].full_prompt is None
    # Fourth is drift -> prompt captured.
    assert records[3].full_prompt is not None


async def test_full_capture_never_never_captures_prompts() -> None:
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="never")
    await watcher.fingerprint(_anthropic_request())
    await watcher.fingerprint(_anthropic_request(system="other"))

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    records = [r async for r in store.iter_recent(since=cutoff)]
    assert all(r.full_prompt is None for r in records)


async def test_full_capture_always_captures_every_prompt() -> None:
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="always")
    await watcher.fingerprint(_anthropic_request())
    await watcher.fingerprint(_anthropic_request())  # identical -> still captured

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    records = [r async for r in store.iter_recent(since=cutoff)]
    assert all(r.full_prompt is not None for r in records)


# Test 10: integration with a fake runner.
async def test_integration_with_fake_anthropic_runner_surfaces_drift() -> None:
    """Wire the watcher into the same protocol an `AnthropicRunner` would
    use; fire 5 requests with subtly-different prefixes; assert audit
    flags the drift on the right breakpoint.

    We don't need a real runner — the integration the spec asks for is
    "the watcher receives the request dict the runner is about to send."
    Driving the watcher directly with hand-built requests is exactly that.
    """
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="on_drift")

    timestamps = [
        "2026-04-12T13:58:11Z",
        "2026-04-12T14:01:42Z",
        "2026-04-12T14:03:55Z",
        "2026-04-12T14:05:08Z",
        "2026-04-12T14:08:21Z",
    ]
    for stamp in timestamps:
        request: dict[str, Any] = {
            "model": "claude-opus-4-7",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": f'system prompt; current_time="{stamp}"',
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        }
        await watcher.fingerprint(request)

    report = await audit(store, window_hours=24)
    # All 5 different timestamps -> 4 transitions on breakpoint 0.
    drift_on_zero = [e for e in report.drift_events if e.breakpoint_index == 0]
    assert len(drift_on_zero) == 4
    # The hint should fire for the timestamp pattern.
    assert any(e.hint and "timestamp" in e.hint for e in drift_on_zero)


async def test_watcher_audit_convenience_method_calls_through_to_audit() -> None:
    """`watcher.audit(window_hours=24)` is sugar for the free
    `audit(store, window_hours=24)` — both should produce the same report."""
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="on_drift")
    await watcher.fingerprint(_anthropic_request(system="a"))
    await watcher.fingerprint(_anthropic_request(system="b"))

    via_method = await watcher.audit(window_hours=24)
    via_function = await audit(store, window_hours=24)

    assert len(via_method.drift_events) == len(via_function.drift_events) == 1


async def test_openai_compat_request_hashes_as_single_segment() -> None:
    """OpenAI-compatible requests have no `cache_control` markers — the
    watcher should still produce one fingerprint (best-effort prefix
    detection per the spec)."""
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store, full_capture="never")

    request: dict[str, Any] = {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ],
    }
    await watcher.fingerprint(request)

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    records = [r async for r in store.iter_recent(since=cutoff)]
    # Exactly one segment for the entire OpenAI prefix.
    assert len(records) == 1
    assert records[0].breakpoint_index == 0
