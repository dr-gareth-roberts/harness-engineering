"""Audit events emitted by `PrivacyBoundary` for every detection.

The load-bearing privacy guarantee: a `DetectionEvent` never carries the
matched value. Only the detector name, direction, action, structural
location, match length, and a UTC timestamp. Tests pin this contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DetectionEvent(BaseModel):
    """One privacy detection. Sinkable through any `Sink`-shaped target.

    Intentionally narrow: no `value`, no `text`, no `match` field. Sinks
    that interpolate this event into JSONL / OpenTelemetry / a database can
    safely persist it without an additional secret-scrub pass.
    """

    event_id: UUID = Field(default_factory=uuid4)
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    kind: Literal["privacy.detection"] = "privacy.detection"
    name: str
    direction: Literal["outbound", "inbound"]
    action: Literal["redact", "block", "audit"]
    location: str
    match_length: int
