"""In-memory response cache for ablation re-runs.

Each ablated input is canonicalized (JSON-serialized, deterministic
key ordering) and hashed with SHA-256. Combined with the
`target_message_index`, that hash forms a stable cache key — a re-run
against the same model and the same ablated input is, by assumption,
deterministic, so we never need to invoke the runner twice for it.

The implementation is deliberately tiny — a dict with `get`/`put`. Anything
fancier (LRU, persistence, distributed caches) belongs in a downstream
adapter, not this primitive.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from harness.prompts.messages import Message


def hash_messages(messages: list[Message]) -> str:
    """SHA-256 hex digest over a JSON serialization of `messages`.

    Pydantic's `model_dump` gives us a deterministic dict; we then JSON-encode
    with sorted keys to remove any ordering ambiguity. The digest is short
    enough for log lines but long enough that collisions are practically
    impossible for the cache sizes we expect (thousands of ablations, max).
    """
    payload: list[dict[str, Any]] = [m.model_dump(mode="json") for m in messages]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CacheKey = tuple[str, int]


class InMemoryAttributionCache:
    """Process-local cache mapping `(input_hash, target_index)` to a response.

    The stored value is the assistant `Message` returned by the runner for
    that ablated input. Callers that want persistence can wrap or replace
    this with their own implementation — the API is two methods.
    """

    def __init__(self) -> None:
        self._store: dict[CacheKey, Message] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: CacheKey) -> Message | None:
        value = self._store.get(key)
        if value is None:
            self._misses += 1
        else:
            self._hits += 1
        return value

    def put(self, key: CacheKey, value: Message) -> None:
        self._store[key] = value

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def __len__(self) -> int:
        return len(self._store)


__all__ = [
    "CacheKey",
    "InMemoryAttributionCache",
    "hash_messages",
]
