"""Tests for `harness.privacy.PresidioDetector` (Wave 13b #1).

The detector is a thin adapter around Presidio's `AnalyzerEngine`. We
test the adapter's contract — converting `RecognizerResult` objects
into `Detection`s, honoring `direction`/`action`/`score_threshold`,
filtering by `entities` — by passing in a fake analyzer rather than
installing the real `presidio-analyzer` package (which pulls spaCy +
a ~50MB model). The lazy-import path is verified by a separate test
that monkeypatches the import to fail.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import pytest

from harness.privacy.detectors import Detector


@dataclass
class _FakeRecognizerResult:
    """Mimics `presidio_analyzer.RecognizerResult` enough for the adapter
    to convert it into a `Detection`."""

    entity_type: str
    start: int
    end: int
    score: float


class _FakeAnalyzerEngine:
    """Stand-in for `presidio_analyzer.AnalyzerEngine`. The real engine
    loads spaCy and is slow to construct; this fake just records what
    it was asked and returns scripted results.
    """

    def __init__(self, scripted: list[_FakeRecognizerResult] | None = None) -> None:
        self.scripted = list(scripted or [])
        self.calls: list[dict[str, Any]] = []

    def analyze(
        self,
        *,
        text: str,
        entities: list[str] | None,
        language: str,
        score_threshold: float,
    ) -> list[_FakeRecognizerResult]:
        self.calls.append(
            {
                "text": text,
                "entities": entities,
                "language": language,
                "score_threshold": score_threshold,
            }
        )
        return list(self.scripted)


# ---------------------------------------------------------------------------
# Adapter behavior with a scripted fake analyzer


def test_presidio_detector_satisfies_detector_protocol() -> None:
    """Structural check: PresidioDetector matches the `Detector`
    Protocol signature without inheritance."""
    from harness.privacy.presidio import PresidioDetector

    detector: Detector = PresidioDetector(
        name="presidio_test",
        analyzer=_FakeAnalyzerEngine(),
    )
    assert detector.name == "presidio_test"


def test_presidio_detector_translates_recognizer_results_to_detections() -> None:
    from harness.privacy.presidio import PresidioDetector

    fake = _FakeAnalyzerEngine(
        scripted=[
            _FakeRecognizerResult(entity_type="PERSON", start=10, end=14, score=0.95),
            _FakeRecognizerResult(entity_type="EMAIL_ADDRESS", start=20, end=35, score=0.99),
        ]
    )
    detector = PresidioDetector(name="presidio_pii", analyzer=fake)
    detections = detector.scan("hello world ...", direction="outbound")

    assert len(detections) == 2
    # name = adapter.name + "." + entity_type
    assert detections[0].name == "presidio_pii.PERSON"
    assert detections[0].start == 10
    assert detections[0].end == 14
    assert detections[1].name == "presidio_pii.EMAIL_ADDRESS"


def test_presidio_detector_passes_score_threshold_through() -> None:
    """The configured threshold reaches Presidio's analyze() call."""
    from harness.privacy.presidio import PresidioDetector

    fake = _FakeAnalyzerEngine()
    detector = PresidioDetector(analyzer=fake, score_threshold=0.7)
    detector.scan("text", direction="outbound")

    assert fake.calls[0]["score_threshold"] == 0.7


def test_presidio_detector_passes_entity_filter_through() -> None:
    from harness.privacy.presidio import PresidioDetector

    fake = _FakeAnalyzerEngine()
    detector = PresidioDetector(
        analyzer=fake,
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
    )
    detector.scan("text", direction="outbound")
    assert fake.calls[0]["entities"] == ["EMAIL_ADDRESS", "PHONE_NUMBER"]


def test_presidio_detector_passes_language_through() -> None:
    from harness.privacy.presidio import PresidioDetector

    fake = _FakeAnalyzerEngine()
    detector = PresidioDetector(analyzer=fake, language="de")
    detector.scan("text", direction="outbound")
    assert fake.calls[0]["language"] == "de"


def test_presidio_detector_skips_when_direction_excludes_pass() -> None:
    """Outbound-only detector must early-return on inbound passes —
    same contract as the regex/entropy detectors."""
    from harness.privacy.presidio import PresidioDetector

    fake = _FakeAnalyzerEngine(
        scripted=[_FakeRecognizerResult(entity_type="PERSON", start=0, end=4, score=0.9)]
    )
    detector = PresidioDetector(analyzer=fake, direction="outbound")

    # Inbound pass: detector skips entirely (no analyze call).
    detections = detector.scan("text", direction="inbound")
    assert detections == []
    assert fake.calls == []

    # Outbound pass: detector runs.
    detections = detector.scan("text", direction="outbound")
    assert len(detections) == 1


def test_presidio_detector_action_propagates_to_detections() -> None:
    from harness.privacy.presidio import PresidioDetector

    fake = _FakeAnalyzerEngine(
        scripted=[_FakeRecognizerResult(entity_type="PERSON", start=0, end=4, score=0.9)]
    )
    detector = PresidioDetector(analyzer=fake, action="block")
    [detection] = detector.scan("text", direction="outbound")
    assert detection.action == "block"


def test_build_pii_pack_returns_one_outbound_detector() -> None:
    """The pre-built pack mirrors PII_PACK's posture (outbound by default)."""
    from harness.privacy.presidio import build_pii_pack

    fake = _FakeAnalyzerEngine()
    pack = build_pii_pack(analyzer=fake)
    assert len(pack) == 1
    assert pack[0].direction == "outbound"
    assert pack[0].action == "redact"


# ---------------------------------------------------------------------------
# Lazy import: missing extra raises a clear error


def test_constructor_raises_when_presidio_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting `sys.modules["presidio_analyzer"] = None` makes Python's
    import system treat the package as unavailable on the next import
    attempt. The lazy import inside `__init__` must raise with a
    message that points the user at `[privacy-ml]`.
    """
    cached = [
        name
        for name in sys.modules
        if name == "presidio_analyzer" or name.startswith("presidio_analyzer.")
    ]
    for name in cached:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)

    from harness.privacy.presidio import PresidioDetector

    with pytest.raises(ImportError, match=r"\[privacy-ml\]"):
        PresidioDetector()  # no `analyzer` injected → triggers lazy import
