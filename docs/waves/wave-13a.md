## Wave 13a — Streaming output

### Goal
Ship #9 (the heaviest single item in the 28-gap plan): callers can
observe partial output as the model generates rather than waiting for
the full assembled `Message`. Per the Wave 12 advisor split, this got
its own wave to avoid risking regressions in the 510 tests around
`AnthropicRunner.__call__`'s tool-use loop.

### Status
Shipped on `feature/wave-13a-streaming`. One gap cleared (#9).

### What landed

| | Surface | Location |
| --- | --- | --- |
| Event types | `TextDelta`, `ToolUseStart`, `ToolUseEnd`, `MessageEnd` (Pydantic models) | `src/harness/streaming/__init__.py` |
| Protocol | `StreamingRunner` (runtime_checkable, requires `run_stream(...)`) | `src/harness/streaming/__init__.py` |
| Runner method | `AnthropicRunner.run_stream()` — parallel async generator method | `src/harness/runner/anthropic.py` |
| Orchestrator method | `Orchestrator.run_stream()` with `SessionStart`/`SessionEnd` + telemetry session/span scopes | `src/harness/agents/orchestrator.py` |
| Top-level re-exports | `MessageEnd`, `StreamEvent`, `StreamingRunner`, `TextDelta`, `ToolUseEnd`, `ToolUseStart` from `harness` | `src/harness/__init__.py` |

**Path B duplication, per advisor**: `AnthropicRunner.__call__` is
~150 lines of intricate state management (tool-use loop, speculator
begin/end, hook emission order, cache-breakpoint counting, timeout
wrapping, replacement honoring, pause/refusal handling). Refactoring
to share the loop body between `__call__` and `run_stream` was
explicitly rejected — the win is deduplication, the cost is risking
all those tests. `run_stream` is a parallel method that mostly-
duplicates the logic with yield points; refactor-to-share is a
follow-up wave once both paths are proven.

Yield order, per iteration:
- `TextDelta(text=...)` per SDK text-delta event.
- `ToolUseStart(call=...)` per `content_block_stop` for tool_use,
  *after* `speculator.observe` but *before* the runner's
  hook + dispatch cycle.
- `ToolUseEnd(call=..., result=...)` after dispatch.

Terminal: exactly one `MessageEnd(message=...)` whose `message`
matches what `__call__` would have returned. Fires for `end_turn` /
`stop_sequence` / `pause_turn` / `refusal` stop reasons.

### Tests added

| File | Count | Coverage |
| --- | --- | --- |
| `tests/runner/test_streaming.py` | 11 | text-only stream yields TextDelta + MessageEnd; tool-use stream yields ToolUseStart + ToolUseEnd around dispatch; AnthropicRunner satisfies StreamingRunner; plain Callable runner does not; Orchestrator.run_stream raises TypeError for non-streaming runner; Orchestrator emits SessionStart before / SessionEnd after the runner stream; multi-event ordering pinned (TextDelta → ToolUseStart → ToolUseEnd → TextDelta → MessageEnd); **speculator-during-stream lifecycle** (the targeted test the advisor flagged before declaring done) — begin/observe/cancel_unobserved/try_resolve/end fire just as in `__call__`; speculator hit short-circuits dispatch but ToolUseEnd still fires; MessageEnd uniqueness across multi-iteration runs; non-streaming `__call__` regression sanity. |

11 new tests, **548 total** (was 537). Coverage **89%** (gate 85%).

### Verification gate

```
ruff check                       — clean
ruff format --check             — 176 files clean
mypy --strict src tests         — clean (161 source files)
pytest --cov=harness            — 548 passed, 1 skipped, 89% coverage
mkdocs build --strict           — clean (~1s)
uv build                         — wheel + sdist build cleanly
```

### Deferred from this wave

- **`OpenAICompatRunner.run_stream()`** — the chat-completions API
  has the delta-by-delta shape (chunks of text + tool_call deltas)
  but the integration is a separate piece of work. Queued for a
  follow-up wave; today only `AnthropicRunner` satisfies
  `StreamingRunner`.
- **CLI `--stream` mode** — `harness debug --stream` would print
  text deltas as they arrive. The wiring is mechanical given the
  primitives now ship; queued so the wave entry stays focused on
  the protocol surface.
- **Refactor `__call__` to share loop body with `run_stream`** —
  deliberate non-goal of this wave per the advisor recommendation.
  The duplication is intentional and bounded; refactor-to-share is
  a follow-up wave once both paths are proven.

### Commits

```
*  chore(progress): rotate Wave 12 to docs/waves/
*  feat(streaming): TextDelta / ToolUseStart / ToolUseEnd / MessageEnd + StreamingRunner Protocol
*  feat(runner): AnthropicRunner.run_stream() parallel method
*  feat(agents): Orchestrator.run_stream() delegates to StreamingRunner
*  docs: CHANGELOG + progress.md log of Wave 13a
```
