"""Detector unit tests.

Covers spec tests 1, 2, 9, 10 from `designs/standout.md` §6:

1. RegexDetector matches AKIA... AWS keys.
2. EntropyDetector behaviour (low- vs high-entropy strings).
9. SECRET_PACK round-trip catches AWS / GitHub / Stripe / Anthropic shapes.
10. PII_PACK round-trip catches SSN / phone / email shapes.

Plus a few targeted lower-level checks that lock in detector semantics
(direction filtering, action propagation).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from harness.privacy.detectors import (
    Detection,
    Detector,
    EntropyDetector,
    RegexDetector,
)
from harness.privacy.packs import PII_PACK, SECRET_PACK

# ---------------------------------------------------------------------------
# RegexDetector


def test_regex_detector_matches_aws_key_shape() -> None:
    """Spec test 1 — AWS-key-shaped string is detected."""
    detector = RegexDetector(
        "aws_access_key",
        r"\bAKIA[A-Z0-9]{16}\b",
        action="block",
    )
    detections = detector.scan(
        "leaked: AKIAIOSFODNN7EXAMPLE somewhere",
        direction="outbound",
    )
    assert len(detections) == 1
    assert detections[0].name == "aws_access_key"
    assert detections[0].action == "block"


def test_regex_detector_returns_empty_when_direction_excluded() -> None:
    detector = RegexDetector(
        "us_ssn",
        r"\b\d{3}-\d{2}-\d{4}\b",
        direction="outbound",
        action="redact",
    )
    assert detector.scan("123-45-6789", direction="outbound")
    assert detector.scan("123-45-6789", direction="inbound") == []


def test_regex_detector_finds_multiple_matches() -> None:
    detector = RegexDetector(
        "us_ssn",
        r"\b\d{3}-\d{2}-\d{4}\b",
        direction="both",
    )
    detections = detector.scan(
        "two: 111-22-3333 and 444-55-6666",
        direction="outbound",
    )
    assert len(detections) == 2
    starts = sorted(d.start for d in detections)
    assert starts == [5, 21]


def test_regex_detector_carries_action_into_detection() -> None:
    detector = RegexDetector("name", r"x", action="audit")
    [det] = detector.scan("x", direction="outbound")
    assert det.action == "audit"


# ---------------------------------------------------------------------------
# EntropyDetector


def test_entropy_detector_ignores_low_entropy_strings() -> None:
    """Spec test 2a — `"x" * 30` is long but low-entropy: no flag."""
    detector = EntropyDetector(min_entropy=4.5, min_length=24)
    assert detector.scan("x" * 30, direction="outbound") == []


def test_entropy_detector_flags_high_entropy_string() -> None:
    """Spec test 2b — a real high-entropy token is flagged.

    The synthetic token below has 64 hex-ish characters with very high
    Shannon entropy (>5 bits/char).
    """
    secret = "9aF3qZ7kP2vB8nC4xS6tR1dE5wY0uM7jL9hG3bN8oI4cV2pK6yA1zXrTeUq"
    detector = EntropyDetector(min_entropy=4.5, min_length=24)
    detections = detector.scan(f"key={secret} done", direction="outbound")
    assert len(detections) == 1
    assert detections[0].name == "high_entropy"


def test_entropy_detector_skips_short_tokens_under_min_length() -> None:
    detector = EntropyDetector(min_entropy=4.5, min_length=24)
    assert detector.scan("aB3cD4e", direction="outbound") == []


def test_entropy_detector_default_action_is_audit() -> None:
    detector = EntropyDetector()
    assert detector.action == "audit"


# ---------------------------------------------------------------------------
# Pre-built packs


def _scan_with_pack(
    pack: Sequence[Detector],
    text: str,
    direction: Literal["outbound", "inbound"] = "outbound",
) -> list[Detection]:
    """Run an ordered pack of detectors against `text` and merge detections."""
    flat: list[Detection] = []
    for det in pack:
        flat.extend(det.scan(text, direction=direction))
    return flat


def test_secret_pack_catches_aws_github_stripe_anthropic() -> None:
    """Spec test 9 — SECRET_PACK detects all four canonical secret shapes."""
    samples = {
        "aws_access_key": "credentials AKIAIOSFODNN7EXAMPLE here",
        "anthropic_api_key": "Bearer sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "github_token": "GH_TOKEN=ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
        "stripe_key": "STRIPE=sk_live_abcdefghijklmnopqrstuvwx",
    }
    for expected_name, payload in samples.items():
        detections = _scan_with_pack(SECRET_PACK, payload)
        names = {d.name for d in detections}
        assert expected_name in names, (
            f"SECRET_PACK missed {expected_name} in payload {payload!r}; got {names}"
        )


def test_pii_pack_catches_ssn_phone_email() -> None:
    """Spec test 10 — PII_PACK detects SSN, phone, and email shapes."""
    samples = {
        "us_ssn": "patient ssn 123-45-6789 on file",
        "us_phone": "call (415) 555-1234 today",
        "email": "send to user@example.com please",
    }
    for expected_name, payload in samples.items():
        detections = _scan_with_pack(PII_PACK, payload)
        names = {d.name for d in detections}
        assert expected_name in names, (
            f"PII_PACK missed {expected_name} in payload {payload!r}; got {names}"
        )
