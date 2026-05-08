from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from harness.prompts.messages import Message
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


class PostAssistantMessage(Event):
    """Emitted by a `Runner` once per assistant message the model produces.

    Fires for *every* assistant message in a tool-use loop — not just the
    terminal one. An assistant message that contains both text and a
    `tool_use` block (a single-iteration "I'll look that up..." + call)
    triggers one `PostAssistantMessage` plus one `PreToolUse` per call.

    Hook handlers should treat this event as observational: by the time
    it fires, the message has already been produced by the model, so
    `HookDecision(block=True)` cannot un-emit it. Use `PreToolUse` for
    blocking; use `PostAssistantMessage` for inspection / contract
    enforcement / telemetry.
    """

    message: Message


class Stop(Event):
    pass


class HookDecision(BaseModel):
    block: bool = False
    reason: str | None = None
    replacement: Any = None
