# `harness.runner`

Pluggable runners satisfying the `Runner` protocol. `EchoRunner`
and `CannedRunner` ship in the base install (no API key needed);
`AnthropicRunner` (`[anthropic]`) and `OpenAICompatRunner`
(`[openai-compat]`) drive real models.

## When to reach for this

- **`EchoRunner` / `CannedRunner`** — tests, demos, smoke checks,
  any path where you want determinism without an API key.
- **`AnthropicRunner`** — production with Claude (Anthropic Messages
  API). Implements `StreamingRunner` (Wave 13a).
- **`OpenAICompatRunner`** — production with OpenAI, or any
  OpenAI-compatible local server (vLLM, llama.cpp, Ollama, LM
  Studio, Together, Groq). Set `base_url=` to point at a local
  endpoint.

## Quick example

<!-- reason: illustrative; references undefined dispatcher / hooks and needs [anthropic] / [openai-compat] extras -->
<!--pytest.mark.skip-->
```python
from harness import AnthropicRunner, OpenAICompatRunner, EchoRunner

# No API key — for tests / demos.
runner = EchoRunner()

# Real Anthropic.
runner = AnthropicRunner(dispatcher, hooks)

# Real OpenAI.
runner = OpenAICompatRunner(dispatcher, hooks)

# Local Ollama.
runner = OpenAICompatRunner(
    dispatcher, hooks,
    base_url="http://localhost:11434/v1",
)
```

The vendor runners share kwargs you can plug optionally:

<!-- reason: illustrative; placeholder `...` args and undefined Speculator / PrefixWatcher constructors -->
<!--pytest.mark.skip-->
```python
AnthropicRunner(
    dispatcher, hooks,
    timeout_s=30.0,                              # Wave 10 #6
    speculator=Speculator(...),                  # Wave 3, 6
    prefix_watcher=PrefixWatcher(...),           # Wave 2
)
```

## Gotchas

- **`Runner` is a Callable, not a class.** Anything matching
  `Callable[[SubAgent, list[Message]], Awaitable[Message]]` works.
  Wrappers (`DebugRunner`, `PlanGuardedRunner`, `ReplayRunner`,
  `PrivacyBoundary.wrap(...)`) preserve this shape so they compose.
- **`HookDecision.replacement`** is honored on both `PreToolUse`
  (skip dispatch) and `PostToolUse` (rewrite result) since Wave 10
  #5. Pre-Wave-10, only `block` was honored.
- **`pause_turn` and `refusal` stop reasons** fire `PauseTurn` /
  `Refusal` events instead of raising (Wave 10 #4). The partial
  assistant message is returned; callers can re-invoke or inspect.
- **OpenAI's `content_filter` finish reason still raises**, awaiting
  the symmetric event treatment. Tracked as deferred.

## Related

- [`harness.streaming`](streaming.md) — `AnthropicRunner.run_stream`.
- [`harness.replay`](replay.md) — `ReplayRunner` for deterministic playback.
- [`harness.debug`](debug.md) — `DebugRunner` wraps any runner with breakpoints.
- [`examples/anthropic_runner.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/anthropic_runner.py)

## API reference

::: harness.runner
