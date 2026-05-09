# Roadmap progress log

> Living document for the post-MVP roadmap work on `harness-engineering`.
> Each wave gets its own section with plan, decisions, and a per-step log.
> Older waves are archived under `docs/waves/` to keep this file focused
> on the current wave; the archive paths are linked in the status table.

## Status snapshot

| #     | Item                                              | Status  | Archive                                          |
| ----- | ------------------------------------------------- | ------- | ------------------------------------------------ |
| 0–6   | MVP scaffold + post-MVP items 1–6                 | shipped | [docs/waves/initial-scaffold.md](docs/waves/initial-scaffold.md) |
| Wave 1 | Counterfactual replay / contracts / fuzz / attribute / diff-eval | shipped | [docs/waves/wave-1.md](docs/waves/wave-1.md) |
| Wave 2 | Cache / privacy / plan / debug + post-Wave-2 integration fixes  | shipped | [docs/waves/wave-2.md](docs/waves/wave-2.md) |
| Wave 3 | Speculative tool execution (#5)                   | shipped | [docs/waves/wave-3.md](docs/waves/wave-3.md) |
| Wave 4 | OTel sink / plan inference / cross-session predictor / OpenAICompat speculator | shipped | [docs/waves/wave-4.md](docs/waves/wave-4.md) |
| Wave 5 | Runnable example per module                       | shipped | [docs/waves/wave-5.md](docs/waves/wave-5.md) |
| Wave 6 | Per-event speculator cancellation (`observe` / `cancel_unobserved`) | shipped | [docs/waves/wave-6.md](docs/waves/wave-6.md) |
| Wave 7 | DAP for debug REPL (`harness debug --dap`)        | shipped | [docs/waves/wave-7.md](docs/waves/wave-7.md) |
| Wave 8 | Polish + docs site + hardening                    | shipped | [docs/waves/wave-8.md](docs/waves/wave-8.md) |
| Wave 9 | CI/CD + governance + housekeeping                  | shipped | [docs/waves/wave-9.md](docs/waves/wave-9.md) |
| Wave 10 | Vendor runner parity + robustness                 | shipped | [docs/waves/wave-10.md](docs/waves/wave-10.md) |
| Wave 11 | Deeper observability + verification               | shipped | [docs/waves/wave-11.md](docs/waves/wave-11.md) |
| Wave 12 | Modality + Files API                              | shipped | [docs/waves/wave-12.md](docs/waves/wave-12.md) |
| Wave 13a | Streaming output (`Orchestrator.run_stream`)     | shipped | (current — see below)                            |

**Status: 10 of 10 standout features shipped, plus Waves 5–13a polish.**
The forward plan from `0.2.0` to `1.0` lives in
[`docs/plan.md`](docs/plan.md): six waves total. Waves 9–13a shipped
(20 of 28 gaps cleared; #1, #2, #15, #16, #17, #19 remain across
Wave 13b).

## Cross-cutting decisions

- **Optional extras over runtime deps.** Each module that pulls in a
  heavy dependency (Anthropic SDK, OpenAI SDK, Hypothesis, sentence-
  transformers, …) lands as `[extras]` so the base install stays at
  `pydantic` only. Imports at the top of submodules use guarded
  `try/except ImportError` with a clear error pointing at the extra.
- **Vendor-neutral primitives, vendor-specific glue.** Core types
  live in the base package; concrete integrations live in
  `harness.<module>.<vendor>` submodules (e.g.
  `harness.runner.anthropic`).
- **Structural protocols for runner extension.** Wave 2 + Wave 3
  added `prefix_watcher` and `speculator` kwargs on the runner
  constructors via `Protocol`s in `src/harness/runner/protocols.py`.
  Feature modules satisfy them structurally; the runner has no
  runtime dependency on any feature module.
- **Idempotency is a tool-author promise**, not enforced by the
  speculator. Marking `Tool.idempotent=True` allows speculative
  pre-execution; a tool that says it's idempotent but has side
  effects produces silent duplicate side effects on miss. The
  contract is documented loud in `Speculator`'s class docstring.
- **One PR, multiple waves**. Waves 1–8 all landed on
  `chore/initial-scaffold` (PR #1) as conceptually one delivery —
  "the post-MVP layer + standout features." Waves 9+ branch
  individually off `main` and land separately.

---


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
