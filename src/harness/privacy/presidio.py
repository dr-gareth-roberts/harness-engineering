"""Microsoft Presidio adapter for `harness.privacy` (Wave 13b #1).

`PresidioDetector` wraps Presidio's `AnalyzerEngine` behind the
existing `Detector` Protocol. The shipped regex + entropy detectors
(`harness.privacy.detectors`) catch common shapes (high-entropy
strings, structured PII like emails / SSNs); Presidio adds broader
recognizers — people's names, addresses, phone numbers in international
formats, dates of birth, IBAN numbers, etc. — backed by a small NLP
model (spacy by default).

When to pick Presidio over the regex pack:

- **Regex pack** is enough when your PII shapes are bounded (you know
  what you're scrubbing — SSN, AWS keys, GitHub tokens, etc.) and you
  need predictable, reproducible detections at zero marginal cost.
- **Presidio** is the right call when the input is free-form text
  carrying arbitrary user data and you can't enumerate the shapes
  ahead of time. The trade-off: ~50ms / scan, model download (~50MB)
  on first use, and looser "is this PII?" semantics.

Lazy imports `presidio_analyzer` from the constructor so importing
this module does not require the `[privacy-ml]` extra; only
constructing the detector does. Install with:
`uv sync --extra privacy-ml`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harness.privacy.detectors import (
    Action,
    Detection,
    Detector,
    DetectorDirection,
    Direction,
)

if TYPE_CHECKING:
    # Presidio is an opt-in extra; mypy may run without it installed.
    # Suppress the import-not-found here so the strict gate stays
    # green in environments that don't pull `[privacy-ml]`.
    from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]


class PresidioDetector:
    """Detector backed by Microsoft Presidio's `AnalyzerEngine`.

    Constructor parameters:

    - `name` — the detector's logical name, surfaced on each emitted
      `Detection`. Convention: `"presidio_pii"` or
      `"presidio_<entity>"` for narrowly-scoped instances.
    - `entities` — list of Presidio entity types to detect, e.g.
      `["EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER"]`. When `None`,
      Presidio's full default registry is used.
    - `score_threshold` — minimum confidence (0..1) for a detection
      to be reported. Default 0.5; tighten for fewer false positives,
      loosen for more recall.
    - `direction` / `action` — same as the regex/entropy detectors.
    - `analyzer` — optional pre-constructed `AnalyzerEngine` to share
      model loading across detector instances. When `None`, the
      detector lazy-loads its own.
    - `language` — `"en"` by default; pass `"de"` / `"es"` / etc. if
      you've configured Presidio with the matching spaCy model.
    """

    def __init__(
        self,
        name: str = "presidio_pii",
        *,
        entities: list[str] | None = None,
        score_threshold: float = 0.5,
        direction: DetectorDirection = "both",
        action: Action = "redact",
        analyzer: AnalyzerEngine | None = None,
        language: str = "en",
    ) -> None:
        self.name = name
        self.direction: DetectorDirection = direction
        self.action: Action = action
        self._entities = entities
        self._score_threshold = score_threshold
        self._language = language

        if analyzer is not None:
            self._analyzer: AnalyzerEngine = analyzer
        else:
            try:
                from presidio_analyzer import AnalyzerEngine as _AnalyzerEngine
            except ImportError as exc:
                raise ImportError(
                    "PresidioDetector requires the [privacy-ml] extra. "
                    "Install with: uv sync --extra privacy-ml"
                ) from exc
            self._analyzer = _AnalyzerEngine()

    def scan(self, text: str, *, direction: Direction) -> list[Detection]:
        # Skip detectors whose configured direction excludes the
        # current pass — same contract the regex/entropy detectors use.
        if self.direction != "both" and self.direction != direction:
            return []

        results = self._analyzer.analyze(
            text=text,
            entities=self._entities,
            language=self._language,
            score_threshold=self._score_threshold,
        )

        detections: list[Detection] = []
        for r in results:
            # Presidio's `RecognizerResult` exposes `start`, `end`,
            # `entity_type`, `score`. We surface entity_type as part
            # of the detection name for downstream filtering, but
            # keep the configured `self.name` as the prefix for
            # operator-recognizable grouping.
            entity_label = getattr(r, "entity_type", "PII")
            detections.append(
                Detection(
                    name=f"{self.name}.{entity_label}",
                    start=int(r.start),
                    end=int(r.end),
                    direction=direction,
                    action=self.action,
                )
            )
        return detections


# Static typing assist: PresidioDetector satisfies the structural
# Detector protocol. The check is a no-op at runtime; mypy verifies it.
_: type[Detector] = PresidioDetector


def build_pii_pack(
    *,
    score_threshold: float = 0.5,
    action: Action = "redact",
    direction: DetectorDirection = "outbound",
    analyzer: Any | None = None,
) -> list[PresidioDetector]:
    """Pre-built detector pack mirroring `harness.privacy.PII_PACK` in
    spirit — common PII shapes covered with Presidio's broader
    recognizers.

    Defaults to outbound-only (matches `PII_PACK`'s posture: don't
    leak real-user data to the model). Returns a list of one
    `PresidioDetector` configured for Presidio's standard PII
    entities; users can extend with additional narrowly-scoped
    instances if they want per-entity actions.
    """
    return [
        PresidioDetector(
            name="presidio_pii",
            entities=[
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "US_SSN",
                "US_DRIVER_LICENSE",
                "US_PASSPORT",
                "CREDIT_CARD",
                "IBAN_CODE",
                "DATE_TIME",
                "LOCATION",
                "IP_ADDRESS",
            ],
            score_threshold=score_threshold,
            direction=direction,
            action=action,
            analyzer=analyzer,
        )
    ]


__all__ = ["PresidioDetector", "build_pii_pack"]
