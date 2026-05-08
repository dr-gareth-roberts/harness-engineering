from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from harness.agents.definition import SubAgent
from harness.prompts.messages import Message


class SessionNotFound(KeyError):
    """Raised by `Session.restore` when no record exists for the given id."""


class SessionRecord(BaseModel):
    """A complete snapshot of a multi-turn conversation.

    Tool calls and decisions live inside `messages` (as `ContentBlock` entries
    of type `tool_use` / `tool_result`); we don't duplicate them at the record
    level.
    """

    session_id: str
    agent: SubAgent
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def touched(self) -> SessionRecord:
        """Return a copy with `updated_at` set to now; `created_at` preserved."""
        return self.model_copy(update={"updated_at": datetime.now(UTC)})
