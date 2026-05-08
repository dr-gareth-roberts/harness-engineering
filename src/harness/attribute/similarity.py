"""Similarity metrics for ablation-based attribution.

Each metric is a callable returning a value in `[0, 1]`, where `1.0` is
"identical" and `0.0` is "completely different". Higher similarity between
the original target and the ablated re-run means the removed chunk had less
influence; lower similarity means more influence.

Three implementations ship:

- `JaccardSimilarity` — token overlap. Zero deps.
- `LengthRatio` — character-length ratio. Zero deps; cruder.
- `EmbeddingSimilarity` — cosine similarity of sentence-transformer embeddings.
  Requires the `[attribute]` extra (`sentence-transformers>=2`); imports lazily
  so the rest of the package stays usable without that heavy dependency.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Similarity(Protocol):
    """Compare two strings, returning a similarity score in `[0, 1]`."""

    def __call__(self, a: str, b: str) -> float: ...


class JaccardSimilarity:
    """Token-overlap similarity using Jaccard index.

    Tokenization is lowercase, whitespace-split. The score is
    `|A ∩ B| / |A ∪ B|`. Identical strings score `1.0`; disjoint vocabularies
    score `0.0`. By convention, when both inputs are empty (empty union) the
    score is `1.0` — they are trivially "the same".
    """

    def __call__(self, a: str, b: str) -> float:
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        union = tokens_a | tokens_b
        if not union:
            return 1.0
        intersection = tokens_a & tokens_b
        return len(intersection) / len(union)


class LengthRatio:
    """Character-length ratio similarity.

    Returns `min(len(a), len(b)) / max(len(a), len(b))`. Crude but free —
    useful as a smoke check or when token-level vocabularies are degenerate
    (e.g., very short responses). Both empty inputs → `1.0`.
    """

    def __call__(self, a: str, b: str) -> float:
        len_a = len(a)
        len_b = len(b)
        if len_a == 0 and len_b == 0:
            return 1.0
        denominator = max(len_a, len_b)
        if denominator == 0:
            return 1.0
        return min(len_a, len_b) / denominator


_EMBEDDING_IMPORT_ERROR = (
    "EmbeddingSimilarity requires the [attribute] extra. Install with: uv sync --extra attribute"
)


class EmbeddingSimilarity:
    """Cosine similarity of sentence-transformer embeddings.

    Imports `sentence_transformers` lazily so the rest of the attribute
    package — and crucially its tests — work without the heavy torch install
    that the optional `[attribute]` extra pulls in.

    The lazy import lives inside `__init__`; if the import fails we re-raise
    with a clear, actionable `ImportError`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )
        except ImportError as exc:
            raise ImportError(_EMBEDDING_IMPORT_ERROR) from exc

        self._model_name = model_name
        self._model: Any = SentenceTransformer(model_name)

    def __call__(self, a: str, b: str) -> float:
        # Lazy-imported to avoid pulling numpy at module import time.
        import numpy as np  # type: ignore[import-not-found]

        embeddings = self._model.encode([a, b], convert_to_numpy=True)
        vec_a = embeddings[0]
        vec_b = embeddings[1]
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0 if norm_a == norm_b else 0.0
        cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Clamp into [0, 1]; cosine can be slightly negative or >1 numerically.
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


__all__ = [
    "EmbeddingSimilarity",
    "JaccardSimilarity",
    "LengthRatio",
    "Similarity",
]
