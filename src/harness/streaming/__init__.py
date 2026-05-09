"""Streaming output support for runners and the orchestrator.

Pre-Wave-13a, runners returned a single `Message` once the full
response arrived (and any tool-use loop completed). Wave 13a (#9 in
`docs/plan.md`) adds an event-stream alternative so callers can
observe partial output as the model generates it: text deltas as
they arrive, tool-call starts/ends as they happen, and a terminal
`MessageEnd` carrying the assembled final message.

The streaming surface is opt-in. The existing
`Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]`
protocol stays unchanged; runners that *also* support streaming
implement the `StreamingRunner` Protocol below by exposing a
`run_stream(...)` method that yields `StreamEvent` instances.

`Orchestrator.run_stream(...)` delegates to the runner's stream and
emits the same lifecycle hooks + telemetry as `Orchestrator.run()`.

Event types:

- `TextDelta` — incremental text from the model. Multiple per assistant
  message. Concatenating all `text` fields yields the final
  assistant text.
- `ToolUseStart` — emitted when the model has finished emitting a
  `tool_use` block (we know its name + arguments). Fires *before* the
  runner dispatches the call. Lets observers update progress UI / log
  the call before the handler runs.
- `ToolUseEnd` — emitted after the runner dispatches the call (the
  handler ran, or a hook short-circuited / replaced the result).
  Carries both the call and the result.
- `MessageEnd` — terminal event, exactly one per `run_stream()`
  invocation. Carries the final `Message` the runner would have
  returned from `__call__()` — useful for callers that want both
  streaming UX and the assembled message.

Hook + telemetry order: `Orchestrator.run_stream()` emits
`SessionStart` before the first stream event and `SessionEnd` /
`OrchestratorTurn` after `MessageEnd` (or after a raised exception),
mirroring `Orchestrator.run()`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from harness.agents.definition import SubAgent
from harness.prompts.messages import Message
from harness.tools.schema import ToolCall, ToolResult


class StreamEvent(BaseModel):
    """Base class for streaming events.

    Concrete subclasses use Pydantic for serializability so callers
    can persist the stream to a JSONL file, replay later, or pipe
    through MultiSink-style observers without bespoke encoding.
    """


class TextDelta(StreamEvent):
    """Incremental text from the model. Multiple per assistant message."""

    text: str


class ToolUseStart(StreamEvent):
    """Emitted when the model has finished emitting a tool_use block.

    Fires *before* the runner's hook + dispatch cycle for this call.
    `call.id` matches the model's tool_use id, so observers can
    correlate with the matching `ToolUseEnd`.
    """

    call: ToolCall


class ToolUseEnd(StreamEvent):
    """Emitted after the runner dispatches the tool call.

    Carries both the call and the result. The result reflects whatever
    happened: dispatcher output, `is_error=True` from a hook block,
    `replacement` from a hook decision, or a speculation hit.
    """

    call: ToolCall
    result: ToolResult


class MessageEnd(StreamEvent):
    """Terminal event — exactly one per `run_stream()` invocation.

    Carries the final assembled `Message` the runner would have
    returned from `__call__()`. Useful for callers that want both the
    streaming UX and a single object to persist / forward.
    """

    message: Message


@runtime_checkable
class StreamingRunner(Protocol):
    """Structural protocol for runners that expose a streaming surface.

    A runner satisfies this Protocol by exposing a `run_stream` method
    that returns an async iterator of `StreamEvent`s. The existing
    `Runner = Callable[..., Awaitable[Message]]` protocol stays the
    canonical entry point; `StreamingRunner` is an *additional*
    capability runners can opt into.

    `Orchestrator.run_stream(...)` checks `isinstance(runner,
    StreamingRunner)` (this protocol is `runtime_checkable`) and
    raises `TypeError` if the runner doesn't expose `run_stream`.

    Today: `AnthropicRunner` is the only runner that implements this.
    `OpenAICompatRunner` is queued for a follow-up — its non-streaming
    chat-completions API has the same delta-by-delta shape, but the
    integration is a separate piece of work.
    """

    def run_stream(
        self,
        agent: SubAgent,
        messages: list[Message],
    ) -> AsyncIterator[StreamEvent]: ...


__all__ = [
    "MessageEnd",
    "StreamEvent",
    "StreamingRunner",
    "TextDelta",
    "ToolUseEnd",
    "ToolUseStart",
]
