from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def jsonify(value: Any) -> Any:
    """Round-trip `value` through JSON to coerce it to a serializable shape.

    Non-native types (Path, dataclasses, UUID, etc.) flow through `default=str`
    so a sink that calls `model_dump_json()` cannot raise on them.
    """
    return json.loads(json.dumps(value, default=str))


class TelemetryEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    kind: str


class ToolDispatched(TelemetryEvent):
    """Emitted by `Dispatcher.dispatch()` after each call completes.

    `arguments` is JSON-safe — coerced via `jsonify()` at construction time
    so any `Path` / dataclass / etc. survives a JSONL sink.
    """

    kind: Literal["tool.dispatched"] = "tool.dispatched"
    tool_name: str
    call_id: str | None = None
    arguments: dict[str, Any]
    is_error: bool
    duration_ms: float


class OrchestratorTurn(TelemetryEvent):
    """Emitted by `Orchestrator.run()` after each invocation, success or failure."""

    kind: Literal["orchestrator.turn"] = "orchestrator.turn"
    agent_name: str
    duration_ms: float
    error: str | None = None
