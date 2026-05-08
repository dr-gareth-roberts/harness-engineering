"""Causal provenance via leave-one-out ablation.

`attribute(session, target_message_index, runner, agent, ...)` re-runs the
runner once per input chunk with that chunk removed, then ranks chunks by
how much the response diverged from the original. Higher divergence ⇒
higher causal influence.

Three similarity metrics ship: `JaccardSimilarity` (token overlap, default,
zero deps), `LengthRatio` (cruder, zero deps), and `EmbeddingSimilarity`
(cosine over `sentence-transformers`; lazy-imported under the `[attribute]`
extra).

Two design knobs every caller will reach for:

- `granularity`: `"message"`, `"block"`, or `"sentence"`. Trade precision
  against the N+1 runner calls.
- `estimate_only=True`: counts the chunks (and hence the calls) without
  invoking the runner once. Use it to budget before committing.
"""

from harness.attribute.ablation import (
    AttributionChunk,
    AttributionResult,
    Granularity,
    attribute,
    chunk_session,
)
from harness.attribute.cache import (
    CacheKey,
    InMemoryAttributionCache,
    hash_messages,
)
from harness.attribute.similarity import (
    EmbeddingSimilarity,
    JaccardSimilarity,
    LengthRatio,
    Similarity,
)

__all__ = [
    "AttributionChunk",
    "AttributionResult",
    "CacheKey",
    "EmbeddingSimilarity",
    "Granularity",
    "InMemoryAttributionCache",
    "JaccardSimilarity",
    "LengthRatio",
    "Similarity",
    "attribute",
    "chunk_session",
    "hash_messages",
]
