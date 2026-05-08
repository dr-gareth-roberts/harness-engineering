"""`PrefixWatcher` — fingerprints prompt prefixes per cache breakpoint.

Walks the request dict the runner is about to send, splits it into the
segments separated by `cache_control` markers (Anthropic) or treats the
whole prefix as a single segment (OpenAI-compatible — the SDK protocol
exposes no breakpoint markers, so the best we can do is hash the entire
request as one block). Each segment is canonicalized to JSON with
`sort_keys=True` for deterministic byte-level fingerprints, then hashed
with SHA-256.

`fingerprint(request)` is called once per iteration of the runner's
tool-use loop, immediately before the SDK call. It satisfies the
`PrefixWatcherProtocol` structurally; runners receive it through the
`prefix_watcher=` constructor kwarg.

`full_capture` controls how aggressively we keep the underlying prompt
bytes for later diffing:

- `"never"`   — hashes only; cheapest. `audit` can detect drift but cannot
                show what changed.
- `"on_drift"` — keep the prompt bytes whenever this segment differs from
                the previously-seen segment for the same breakpoint. This
                is the default and the cheapest option that still lets
                `audit` produce a `unified_diff`. The very first record
                for any breakpoint stores its prompt (so future drift has
                a "before" side); subsequent identical records store
                `None` to keep storage flat; drift records store the new
                prompt.
- `"always"`  — store every prompt. Useful while debugging, expensive at
                scale. The `audit` `DriftEvent` diff renders identically.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from harness.cache.store import FingerprintRecord, FingerprintStore

FullCapture = Literal["always", "on_drift", "never"]


def _segments_for_anthropic(request: dict[str, Any]) -> list[Any]:
    """Split the Anthropic request into per-cache-breakpoint segments.

    Anthropic flags a cache breakpoint with `cache_control={"type":
    "ephemeral"}` on individual content blocks inside `system`,
    `messages[].content[]`, or `tools[]`. We walk those blocks in the
    fixed order `tools` -> `system` -> `messages`, and split the flat
    sequence at every block bearing a `cache_control` marker.

    Segment 0 contains everything up to and including the first marker;
    segment 1 contains everything between the first and second marker;
    and so on. If no markers are present we still return one segment
    (the whole prefix) so the watcher can hash *something* — that's how
    we cover the no-cache case for free.
    """
    flat: list[Any] = []
    boundary_indices: list[int] = []

    def _emit(item: dict[str, Any]) -> None:
        flat.append(item)
        # We wrap each block in `{"section": ..., "value": <block>}`. The
        # cache_control marker lives on the wrapped block, not the wrapper —
        # peek inside to detect a breakpoint.
        inner = item.get("value")
        if isinstance(inner, dict) and inner.get("cache_control") is not None:
            # The cache_control block is the *last* element of the segment
            # it terminates — record its index in the flat sequence.
            boundary_indices.append(len(flat) - 1)

    # tools first (they're the most prefix-stable, conventionally cached
    # at position 0).
    tools = request.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            _emit({"section": "tool", "value": tool})

    # system: can be a string or a list of content blocks.
    system = request.get("system")
    if isinstance(system, list):
        for block in system:
            _emit({"section": "system", "value": block})
    elif isinstance(system, str) or system is not None:
        _emit({"section": "system", "value": system})

    # messages: iterate each message's content blocks individually so a
    # mid-message `cache_control` correctly anchors a breakpoint.
    messages = request.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        _emit({"section": "message", "role": msg.get("role"), "value": block})
                else:
                    _emit({"section": "message", "role": msg.get("role"), "value": content})
            else:
                _emit({"section": "message", "value": msg})

    # Top-level scalars (model, max_tokens, …) live in segment 0 so a
    # model-version bump shows up as drift on breakpoint 0. We attach
    # them to the very front of the flat sequence.
    scalar_meta = {
        k: v
        for k, v in request.items()
        if k not in ("system", "messages", "tools") and not isinstance(v, list | dict)
    }
    if scalar_meta:
        flat.insert(0, {"section": "meta", "value": scalar_meta})
        # Shift any boundary indices that came from items appended later.
        boundary_indices = [i + 1 for i in boundary_indices]

    if not boundary_indices:
        return [flat]

    segments: list[Any] = []
    cursor = 0
    for end in boundary_indices:
        segments.append(flat[cursor : end + 1])
        cursor = end + 1
    if cursor < len(flat):
        # Trailing content after the last `cache_control` is its own
        # segment — covers the "you cached the system prompt but the
        # latest user message just got appended" case.
        segments.append(flat[cursor:])
    return segments


def _segments_for_openai(request: dict[str, Any]) -> list[Any]:
    """Treat an OpenAI-compatible request as one opaque segment.

    The Chat Completions protocol exposes no per-block cache markers,
    so any breakpoint detection would be guessing. Hashing the whole
    request prefix as a single segment still catches the common
    "system prompt drifted" / "tool list reordered" cases.
    """
    return [
        {
            "model": request.get("model"),
            "system_or_messages": request.get("messages"),
            "tools": request.get("tools"),
        }
    ]


def _segments(request: dict[str, Any]) -> list[Any]:
    """Pick the splitter heuristically.

    Anthropic requests look like `{"messages": [...], "system": ..., "tools": ...}`
    and may carry `cache_control` markers. OpenAI-compatible requests have the
    same surface keys but never carry `cache_control` markers — we detect
    Anthropic shape by the *presence* of any cache_control, and fall back
    to "single segment" otherwise. Both runners pass dicts through
    `fingerprint()`; tests cover each shape explicitly.
    """
    has_cache_control = _request_has_cache_control(request)
    if has_cache_control:
        return _segments_for_anthropic(request)
    return _segments_for_openai(request)


def _request_has_cache_control(request: dict[str, Any]) -> bool:
    """Return True if any nested block carries a `cache_control` key."""
    stack: list[Any] = [request]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "cache_control" in node:
                return True
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return False


def _canonical(segment: Any) -> str:
    """Deterministic JSON for hashing.

    `sort_keys=True` flattens dict-ordering noise; `default=str` makes
    Pydantic model instances and arbitrary objects hashable (we don't
    expect them in a runner-shaped dict, but defensively serializing
    them as `repr` is better than crashing inside the fingerprint
    callsite).
    """
    return json.dumps(segment, sort_keys=True, default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PrefixWatcher:
    """Fingerprints prompt prefixes per cache breakpoint.

    Construct once per `Runner`; pass it as the `prefix_watcher=` kwarg.
    Storage backend is pluggable — pass an `InMemoryFingerprintStore` for
    tests and a `FileFingerprintStore` for production durability.

    Per-call behaviour: each `fingerprint(request)` walks the request,
    extracts one segment per cache breakpoint, hashes each segment, and
    appends a `FingerprintRecord(timestamp, breakpoint_index, hash,
    full_prompt)` to the store. The watcher keeps an in-memory cache of
    the most recent `(hash, full_prompt)` per breakpoint to decide
    whether `full_capture="on_drift"` should record the new prompt.
    """

    def __init__(
        self,
        store: FingerprintStore,
        *,
        full_capture: FullCapture = "on_drift",
    ) -> None:
        self._store = store
        self._full_capture: FullCapture = full_capture
        # In-memory: most recent (hash, full_prompt) per breakpoint, used
        # to decide whether `on_drift` should snapshot a new prompt.
        self._last_seen: dict[int, tuple[str, str]] = {}

    async def fingerprint(self, request: dict[str, Any]) -> None:
        """Compute one hash per cache breakpoint; persist via the store.

        Called by `AnthropicRunner` / `OpenAICompatRunner` once per
        tool-use loop iteration immediately before the SDK call. The
        signature matches `PrefixWatcherProtocol`; the runner doesn't
        import `harness.cache` at all — structural typing handles it.
        """
        now = datetime.now(UTC)
        segments = _segments(request)

        for index, segment in enumerate(segments):
            canonical = _canonical(segment)
            digest = _sha256(canonical)

            full_prompt: str | None = self._decide_full_prompt(index, digest, canonical)
            record = FingerprintRecord(
                timestamp=now,
                breakpoint_index=index,
                hash=digest,
                full_prompt=full_prompt,
            )
            await self._store.append(record)
            self._last_seen[index] = (digest, canonical)

    def _decide_full_prompt(self, breakpoint_index: int, digest: str, canonical: str) -> str | None:
        if self._full_capture == "always":
            return canonical
        if self._full_capture == "never":
            return None
        # full_capture == "on_drift"
        previous = self._last_seen.get(breakpoint_index)
        if previous is None:
            # First sighting of this breakpoint — store the prompt so
            # any future drift has a "before" side to diff against.
            return canonical
        prev_digest, _ = previous
        if prev_digest != digest:
            # Drift detected — store the new prompt.
            return canonical
        # Same hash as last time — no need to re-store the prompt; the
        # earlier record (or the very first one) already carries it.
        return None
