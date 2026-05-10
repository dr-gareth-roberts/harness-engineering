# `harness.streaming`

Pre-Wave-13a, runners returned a single `Message` once the full
response arrived. This module ships the event-stream alternative —
`Orchestrator.run_stream(...)` yields `TextDelta` per text chunk,
`ToolUseStart` when the model has emitted a tool_use block (before
dispatch), `ToolUseEnd` after dispatch, and exactly one terminal
`MessageEnd` carrying the assembled final message.

The streaming surface is opt-in. Existing
`Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]`
callers continue to work unchanged. Runners that *also* support
streaming implement the `StreamingRunner` Protocol by exposing a
`run_stream(...)` async generator. Today `AnthropicRunner` is the
only built-in that does; `OpenAICompatRunner` is queued for a
follow-up.

## When to reach for this

- You want a "typing indicator" UX — show text deltas as they
  arrive rather than waiting for the full assistant message.
- You want progress events around tool calls (start / end) for a
  live UI.
- You want the same `Orchestrator` to drive both streaming and
  non-streaming paths from the same `Runner` definition.

## Quick example

```python
import asyncio
from harness import Orchestrator, MessageEnd, TextDelta, ToolUseEnd, ToolUseStart

async def stream():
    async for event in orchestrator.run_stream(agent, messages):
        match event:
            case TextDelta(text=t):
                print(t, end="", flush=True)
            case ToolUseStart(call=c):
                print(f"\n[calling {c.name}({c.arguments})...]", end="")
            case ToolUseEnd(call=c, result=r):
                print(f" → {r.content}")
            case MessageEnd(message=m):
                print(f"\n--- final: {len(m.content)} blocks ---")

asyncio.run(stream())
```

## Gotchas

- **`Orchestrator.run_stream` raises `TypeError` immediately** if
  the runner doesn't satisfy `StreamingRunner`. Use `run()` for
  non-streaming runners.
- **`MessageEnd` is exactly once per `run_stream()` invocation** —
  even across multi-iteration tool-use loops. It's the terminal
  event.
- **`SessionStart` / `SessionEnd` hooks fire** around the entire
  stream (start before first event, end after `MessageEnd` or on
  exception). `OrchestratorTurn` telemetry fires once per stream.
- **`AnthropicRunner.run_stream` is path-B parallel to `__call__`** —
  the tool-use loop logic is duplicated rather than shared. Refactor
  to share is a future wave once both paths are proven.

## Related

- [`harness.agents`](agents.md) — `Orchestrator.run_stream` is the entry point.
- [`harness.runner`](runner.md) — `AnthropicRunner.run_stream`.
- Roadmap: `OpenAICompatRunner.run_stream` is on the deferred list.

## API reference

::: harness.streaming
