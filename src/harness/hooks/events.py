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


class PauseTurn(Event):
    """Emitted when a vendor runner sees a `pause_turn` stop reason.

    Anthropic's `pause_turn` indicates the model wants to pause mid-turn
    (typically because a long-running tool reference exceeded the
    server's per-turn budget). The runner returns the partial assistant
    message; the caller can re-invoke with the same message appended to
    history to resume.

    Pre-Wave-10 the runner raised `RuntimeError` on `pause_turn`. The
    event surface lets callers handle pause-and-resume without catching
    a generic exception.
    """

    message: Message
    reason: str = "pause_turn"


class Refusal(Event):
    """Emitted when a vendor runner sees a `refusal` stop reason.

    The model refused to comply with the request. The runner returns a
    refusal-only assistant message; this event lets observers detect
    the refusal without inspecting the message blocks. The text
    accompanying the refusal (if any) is on `message`.
    """

    message: Message


class HookDecision(BaseModel):
    block: bool = False
    reason: str | None = None
    replacement: Any = None
