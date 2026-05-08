"""Prefix-drift watcher: catch silent cache invalidators before the cost spike.

Run with: `uv run python examples/cache.py`

`harness.cache.PrefixWatcher` fingerprints rendered request prefixes per
cache breakpoint. When a "stable" prefix changes byte-for-byte — usually
because something leaked into it (a timestamp, a re-ordered dict, a
varying tool list) — the watcher's `audit` surfaces a `DriftEvent` with
a unified diff so you can see exactly what changed.

This example simulates a deployment where the system prompt accidentally
includes the current time. The first three requests fingerprint cleanly;
the next two have a leaking timestamp that mutates the rendered prefix.
The audit reports the drift and shows the offending lines.

No real model is called — we hand-build request dicts in the shape
`AnthropicRunner._build_request` would produce.
"""

from __future__ import annotations

import asyncio

from harness.cache import InMemoryFingerprintStore, PrefixWatcher


def _request(system_text: str, user: str) -> dict:
    """Hand-built request dict that mimics `AnthropicRunner._build_request`.

    The system block carries `cache_control: {"type": "ephemeral"}` so the
    watcher fingerprints the system prompt as its own cache breakpoint —
    independent of the (varying) user message.
    """
    return {
        "model": "demo-model",
        "max_tokens": 1024,
        "system": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ],
    }


async def main() -> int:
    transcript: list[str] = []
    store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(store=store, full_capture="on_drift")

    # Hold the user message constant across all five requests — we want the
    # demo focused on whether the *system-prompt* segment is stable. The
    # user-message segment naturally drifts when the conversation changes;
    # that's expected and not what this watcher is for.
    user = "search the corpus"

    # Three "stable" requests — identical system prompt and identical user
    # message. The system-prompt breakpoint's hash stays constant.
    transcript.append("--- phase 1: stable cached system prompt ---")
    stable_system = "You are a helpful assistant."
    for i in range(3):
        await watcher.fingerprint(_request(stable_system, user))
        transcript.append(f"  request {i}: fingerprinted (stable system prompt)")

    # Two more requests where someone accidentally embeds the current time
    # in the system prompt — the kind of leak that silently destroys cache
    # hit rate. The system-prompt segment's hash now changes per request.
    transcript.append("--- phase 2: timestamp leak in system prompt ---")
    for i, ts in enumerate(["2026-05-09T13:58:11Z", "2026-05-09T14:03:42Z"]):
        leaky_system = f'{stable_system}\n"current_time": {ts!r}'
        await watcher.fingerprint(_request(leaky_system, user))
        transcript.append(f"  request {3 + i}: fingerprinted (leaked ts {ts})")

    # Audit. Drift events surface the fingerprint mismatch + a unified diff
    # of the actual changed bytes so you can see the leak. The system-prompt
    # breakpoint should drift; the user-message breakpoint shouldn't.
    report = await watcher.audit(window_hours=24.0)
    transcript.append("--- audit ---")
    transcript.append(f"  stable breakpoints: {sorted(report.stable_prefixes)}")
    transcript.append(f"  drift events: {len(report.drift_events)}")
    for event in report.drift_events:
        transcript.append(
            f"  drift @ breakpoint {event.breakpoint_index}: "
            f"hash {event.before_hash[:8]} -> {event.after_hash[:8]}"
        )
        # First few diff lines so the leak is visible.
        for line in event.diff.splitlines()[:6]:
            transcript.append(f"    {line}")
        if event.hint:
            transcript.append(f"    hint: {event.hint}")

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
