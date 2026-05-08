"""Privacy detectors: pure-function-ish scanners over text fragments.

A `Detector` reports `Detection`s — *positions* of matches inside a text
fragment. The `PrivacyBoundary` is responsible for deciding what to *do*
about each detection (redact / block / audit) based on the detector's
configured `action` and the boundary's `on_detect` default.

Detectors do not know about messages, blocks, or directions in their inner
loop — they accept a `direction` so they can early-return when their own
configured direction excludes the current pass, but they otherwise stay a
pure `text -> list[Detection]` mapping. That makes them trivial to unit-test
and re-use outside of the boundary (e.g. for log-line scanning).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Direction = Literal["outbound", "inbound"]
DetectorDirection = Literal["outbound", "inbound", "both"]
Action = Literal["redact", "block", "audit"]


@dataclass(frozen=True)
class Detection:
    """A single match inside a text fragment.

    `location` is a structural hint such as
    ``"messages[2].content[0].text"`` — never the matched value. The boundary
    fills it in; raw `Detector.scan` calls receive an empty location which
    the boundary overwrites.
    """

    name: str
    start: int
    end: int
    direction: Direction
    action: Action
    location: str = ""

    @property
    def match_length(self) -> int:
        return self.end - self.start


@runtime_checkable
class Detector(Protocol):
    """Pure-function-ish scanner.

    Implementations return all detections in `text` for the current pass.
    `direction` is the *current pass* direction (set by the boundary);
    detectors that only run in one direction inspect it and early-return.
    """

    name: str
    direction: DetectorDirection
    action: Action

    def scan(self, text: str, *, direction: Direction) -> list[Detection]: ...


class RegexDetector:
    """Compiles a single regex and reports all non-overlapping matches.

    `direction` selects which boundary passes this detector participates in;
    `action` is the *per-detector* action and overrides the boundary's
    `on_detect` default at decision time.
    """

    def __init__(
        self,
        name: str,
        pattern: str,
        *,
        direction: DetectorDirection = "both",
        action: Action = "redact",
        flags: int = re.IGNORECASE,
    ) -> None:
        self.name = name
        self.direction: DetectorDirection = direction
        self.action: Action = action
        self._regex = re.compile(pattern, flags)

    def scan(self, text: str, *, direction: Direction) -> list[Detection]:
        if self.direction != "both" and self.direction != direction:
            return []
        return [
            Detection(
                name=self.name,
                start=m.start(),
                end=m.end(),
                direction=direction,
                action=self.action,
            )
            for m in self._regex.finditer(text)
        ]


# Tokens that look secret-like — long base64-ish, hex, JWTs, etc. Used by
# `EntropyDetector` to find candidate substrings before the (more expensive)
# Shannon-entropy check. Conservative on purpose: short tokens never get
# flagged, no matter their entropy.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_+/=\-]+")


def _shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character of `s`. Empty -> 0.0."""
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


class EntropyDetector:
    """Flags substrings whose Shannon entropy exceeds a threshold.

    Heuristic, not a guarantee. Defaults are tuned for "secret-shaped"
    strings: at least 24 characters of token-like alphabet with an
    entropy >= 4.5 bits/char. Tune `min_entropy` / `min_length` for
    your workload.

    Default action is `audit` — entropy is noisy enough that automatic
    redaction would corrupt legitimate content (UUIDs, hashes, etc.).
    Callers who want hard blocks should use `RegexDetector` patterns.
    """

    def __init__(
        self,
        *,
        name: str = "high_entropy",
        min_entropy: float = 4.5,
        min_length: int = 24,
        direction: DetectorDirection = "both",
        action: Action = "audit",
    ) -> None:
        self.name = name
        self.direction: DetectorDirection = direction
        self.action: Action = action
        self.min_entropy = min_entropy
        self.min_length = min_length

    def scan(self, text: str, *, direction: Direction) -> list[Detection]:
        if self.direction != "both" and self.direction != direction:
            return []
        out: list[Detection] = []
        for m in _TOKEN_RE.finditer(text):
            token = m.group(0)
            if len(token) < self.min_length:
                continue
            if _shannon_entropy(token) < self.min_entropy:
                continue
            out.append(
                Detection(
                    name=self.name,
                    start=m.start(),
                    end=m.end(),
                    direction=direction,
                    action=self.action,
                )
            )
        return out
