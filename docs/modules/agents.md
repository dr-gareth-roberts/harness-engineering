# `harness.agents`

`SubAgent` (a model + system prompt + tool allow-list) and the
`Orchestrator` that drives one through a runner, emitting
`SessionStart` / `SessionEnd` lifecycle hooks and optionally
`OrchestratorTurn` telemetry. `PromptSubmit` is emitted *one layer
up* by `harness.memory.Session.send` so a registered contract /
policy sees the user text before the orchestrator (and any runner)
does — see [`harness.memory`](memory.md) for the emission site.

## When to reach for this

- You're running a single agent (one tool-use loop). For multi-agent
  patterns see the [comparison](../comparison.md).
- You want lifecycle hooks fired around every run (audit, logging,
  cleanup).
- You want telemetry / trace_id correlation auto-propagated to the
  dispatcher.

## Quick example

<!-- reason: illustrative; AnthropicRunner needs the [anthropic] extra and uses placeholder `[...]` / `...` args -->
<!--pytest.mark.skip-->
```python
import asyncio
from harness import (
    AnthropicRunner, Dispatcher, HookRunner,
    Orchestrator, SubAgent, text,
)

orchestrator = Orchestrator(
    Dispatcher([...]),
    HookRunner(),
    AnthropicRunner(...),
)
agent = SubAgent(
    name="researcher",
    system_prompt="You are a careful researcher.",
    model="claude-opus-4-7",
    allowed_tools=["search", "summarize"],
)

reply = asyncio.run(orchestrator.run(agent, [text("user", "Survey LLM evaluation literature.")]))

# Streaming variant (Wave 13a):
async def stream():
    async for event in orchestrator.run_stream(agent, [text("user", "...")]):
        print(type(event).__name__, getattr(event, "text", ""))
```

## Gotchas

- **`Orchestrator.run_parallel`** is `asyncio.gather` over per-job
  `run` calls. Concurrent runs each get distinct `trace_id` /
  `span_id` (contextvars copy on `create_task`).
- **`run_stream` requires a `StreamingRunner`** — `AnthropicRunner`
  satisfies it; `OpenAICompatRunner` doesn't yet.
  `Orchestrator.run_stream` raises `TypeError` immediately if the
  configured runner doesn't.
- **`SessionStart` / `SessionEnd` fire even on exception** — they
  bracket the run via `try / finally`.
- **`PromptSubmit` is emitted by `Session.send`, not `Orchestrator.run`.**
  If you're driving the orchestrator directly (no `Session` wrapping
  it), no `PromptSubmit` fires — you emit one yourself via
  `hooks.emit(PromptSubmit(prompt=...))` before calling `run`, or use
  `Session` for the documented surface.

## Related

- [`harness.runner`](runner.md) — what `Orchestrator` calls.
- [`harness.streaming`](streaming.md) — the event types `run_stream` yields.
- [`examples/end_to_end.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/end_to_end.py)

## API reference

::: harness.agents
