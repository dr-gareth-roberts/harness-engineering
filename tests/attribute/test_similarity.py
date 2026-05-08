"""Unit tests for the similarity metrics.

Covers spec test 4 (Jaccard identity / disjoint) for the cheap metrics that
ship in the base install, the LengthRatio behavior, and spec test 9: that
EmbeddingSimilarity raises a clear `ImportError` pointing the caller at the
`[attribute]` extra when `sentence-transformers` is not installed.
"""

from __future__ import annotations

import sys

import pytest

from harness.attribute.similarity import (
    EmbeddingSimilarity,
    JaccardSimilarity,
    LengthRatio,
)

# ---------------------------------------------------------------------------
# JaccardSimilarity (spec test 4)


def test_jaccard_identical_strings_score_one() -> None:
    sim = JaccardSimilarity()
    assert sim("hello world", "hello world") == 1.0


def test_jaccard_disjoint_vocab_scores_zero() -> None:
    sim = JaccardSimilarity()
    assert sim("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_jaccard_partial_overlap_scores_in_between() -> None:
    """Two tokens out of three shared → 2 / 4 = 0.5."""
    sim = JaccardSimilarity()
    score = sim("alpha beta gamma", "alpha beta delta")
    assert score == pytest.approx(0.5)


def test_jaccard_is_case_insensitive() -> None:
    sim = JaccardSimilarity()
    assert sim("Hello World", "hello world") == 1.0


def test_jaccard_both_empty_returns_one() -> None:
    """Empty union is the documented edge case — treat as identical."""
    sim = JaccardSimilarity()
    assert sim("", "") == 1.0


# ---------------------------------------------------------------------------
# LengthRatio


def test_length_ratio_identical_lengths_score_one() -> None:
    sim = LengthRatio()
    assert sim("abcd", "wxyz") == 1.0


def test_length_ratio_half_length_scores_half() -> None:
    sim = LengthRatio()
    assert sim("abcd", "ab") == 0.5


def test_length_ratio_both_empty_scores_one() -> None:
    sim = LengthRatio()
    assert sim("", "") == 1.0


# ---------------------------------------------------------------------------
# EmbeddingSimilarity — spec test 9


def test_embedding_similarity_raises_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec test 9: missing [attribute] extra raises a clear ImportError.

    `monkeypatch.setitem(sys.modules, "sentence_transformers", None)` makes a
    subsequent `import sentence_transformers` raise `ImportError` — Python's
    documented behavior for a None entry. This forces the lazy import in
    `EmbeddingSimilarity.__init__` to trip and re-raise with the
    install-instruction message.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError) as excinfo:
        EmbeddingSimilarity()

    message = str(excinfo.value)
    assert "[attribute] extra" in message
    assert "uv sync --extra attribute" in message


def test_embedding_similarity_works_when_installed() -> None:
    """If sentence-transformers happens to be installed, the class instantiates
    and produces a similarity score in [0, 1] for two identical strings.

    Skipped gracefully when the heavy [attribute] extra is not present — the
    base test suite must stay green without it.
    """
    pytest.importorskip("sentence_transformers")

    sim = EmbeddingSimilarity()
    score = sim("hello world", "hello world")
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.0, abs=1e-3)
