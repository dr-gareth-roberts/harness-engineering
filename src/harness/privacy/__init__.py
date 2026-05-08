"""Privacy boundary layer.

Wrap any `Runner` with a pattern + entropy gate that scans every text
fragment crossing the prompt boundary. See `harness.privacy.boundary` for
architecture; `harness.privacy.detectors` for primitives;
`harness.privacy.packs` for pre-built detector lists.
"""

from harness.privacy.boundary import (
    AuditSink,
    PrivacyBoundary,
    PrivacyViolation,
)
from harness.privacy.detectors import (
    Action,
    Detection,
    Detector,
    DetectorDirection,
    Direction,
    EntropyDetector,
    RegexDetector,
)
from harness.privacy.events import DetectionEvent
from harness.privacy.packs import HIPAA_PACK, PII_PACK, SECRET_PACK

__all__ = [
    "HIPAA_PACK",
    "PII_PACK",
    "SECRET_PACK",
    "Action",
    "AuditSink",
    "Detection",
    "DetectionEvent",
    "Detector",
    "DetectorDirection",
    "Direction",
    "EntropyDetector",
    "PrivacyBoundary",
    "PrivacyViolation",
    "RegexDetector",
]
