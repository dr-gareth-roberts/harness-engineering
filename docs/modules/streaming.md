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
follow-up wave.

::: harness.streaming
