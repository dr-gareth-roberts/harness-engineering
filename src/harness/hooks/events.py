from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from harness.tools.schema import ToolCall, ToolResult


class Event(BaseModel):
    pass


class SessionStart(Event):
    pass


class SessionEnd(Event):
    pass


class PromptSubmit(Event):
    prompt: str


class PreToolUse(Event):
    call: ToolCall


class PostToolUse(Event):
    call: ToolCall
    result: ToolResult


class Stop(Event):
    pass


class HookDecision(BaseModel):
    block: bool = False
    reason: str | None = None
    replacement: Any = None
