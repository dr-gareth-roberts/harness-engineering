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
| Wave 10 | Vendor runner parity + robustness                 | shipped | (current — see below)                            |

**Status: 10 of 10 standout features shipped, plus Waves 5–10 polish.**
The forward plan from `0.2.0` to `1.0` lives in
[`docs/plan.md`](docs/plan.md): five waves (9 through 13), ~13–15
developer-days, every gap from the Wave 8 audit assigned to a wave.
Waves 9–10 shipped (13 of those gaps cleared; #1, #2, #7, #8, #9, #10,
#11, #15, #16, #17, #18, #19, #20 remain across Waves 11–13).

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


## Wave 10 — Vendor runner parity + robustness

### Goal
Bring `OpenAICompatRunner` to feature-parity with `AnthropicRunner` on
the speculator surface; make both runners safe to use under production
conditions (transient errors, slow upstream, hooks that want to rewrite
messages); enforce Anthropic's cache-breakpoint cap client-side.

### Status
Shipped on `feature/wave-10-runner-parity`. Five gaps cleared (#3, #4,
#5, #6, #12 from `docs/plan.md`).

### What landed

| # | Item | Implementation |
| --- | --- | --- |
| 12 | Cache-breakpoint cap | `_count_cache_breakpoints` walks the request before each iteration's SDK call; if the count exceeds 4, raises typed `CacheBreakpointLimitExceeded` (a `ValueError`). The check fires *per iteration* because tool_results we feed back may carry their own `cache_control` markers. |
| 6  | Per-iteration timeout | `timeout_s: float \| None = None` kwarg on both runners. AnthropicRunner wraps the entire stream context (`__aenter__`, every `__anext__`, `__aexit__`, `get_final_message`) via a small `_TimeoutStreamCtx` adapter. OpenAICompat wraps the chat-completions create call. Default `None` = no timeout (matches the SDK's own behavior). Retry/backoff is **deferred** — clean retry semantics across streaming + speculator state require more design than the wave budgets. Documented inline. |
| 5  | `HookDecision.replacement` | Both runners now honor `replacement` on `PreToolUse` (skip dispatch, use the supplied `ToolResult` with id patched) and on `PostToolUse` (rewrite the dispatched result before sending back to the model). Pre-Wave-10 only `block` was honored — `replacement` was silently ignored. |
| 4  | `pause_turn` / `refusal` events | Two new event types: `harness.hooks.PauseTurn` (carries the partial assistant message + reason) and `harness.hooks.Refusal` (carries the refusal-only message). AnthropicRunner emits these instead of raising on those stop_reasons; the partial assistant message is returned so callers can resume / inspect. |
| 3  | OpenAICompat speculator parity | `OpenAICompatRunner` now surfaces emitted `tool_call`s to `speculator.observe()` and calls `cancel_unobserved()` *before* the early-return for `stop`/`length` — text-only iterations still cancel any unmatched specs. Functional parity with `AnthropicRunner` Wave 6 cancellation timing. **Streaming refactor deferred**: the chat-completions API is invoked non-streaming today; switching to streaming with per-chunk observation is incremental on top of this and queued for a future wave. The bug surfaced by the parity test (`cancel_unobserved` was inside the tool_calls branch) is now fixed. |

### Tests added

| File | Count | Coverage |
| --- | --- | --- |
| `tests/runner/test_anthropic.py` | 11 | 4 cache-breakpoint cap (helper unit + zero-args helper + over-cap raises + boundary at 4 passes); 2 timeout (raises with delay + completes without); 2 replacement (PreToolUse short-circuits, PostToolUse rewrites); 2 pause_turn/refusal events; 1 unknown-stop-reason still raises (regression of Wave 4 behavior). |
| `tests/runner/test_openai_compat.py` | 4 | 2 timeout (raises / completes); 1 PreToolUse replacement parity; 2 #3 parity (observe per tool_call + cancel_unobserved fires per iteration even text-only). |

15 new tests, **510 total** (was 495).

### Verification gate

```
ruff check                       — clean
ruff format --check             — 171 files clean
mypy --strict src tests         — clean (156 source files)
pytest                           — 510 passed
mkdocs build --strict           — clean (~1s)
uv build                         — wheel + sdist build cleanly
```

### Deferred from this wave

- **Retry/backoff across streaming + speculator state** — clean
  semantics require resetting speculator state on retry, which adds
  complexity not warranted by the perf win. Tracking under #6 in the
  plan as "follow-up".
- **OpenAICompat streaming refactor for per-chunk speculator
  observation** — the non-streaming + post-response observe achieves
  the same cancellation timing as Wave 6's AnthropicRunner. The
  per-chunk variant would only matter for adapters that want to
  cancel mid-stream (Wave 13 #2 territory).
- **`content_filter` finish_reason as an event** on OpenAICompat —
  symmetric to the Anthropic `pause_turn`/`refusal` work, deferred
  for the next vendor-runner pass.

### Commits

```
*  feat(runner): cache-breakpoint cap enforcement + CacheBreakpointLimitExceeded
*  feat(runner): per-iteration timeout_s on both vendor runners
*  feat(runner): honor HookDecision.replacement (PreToolUse + PostToolUse)
*  feat(hooks): PauseTurn + Refusal events; AnthropicRunner emits instead of raising
*  feat(runner): OpenAICompat surfaces tool_calls to speculator.observe + cancel_unobserved
*  docs: progress.md log of Wave 10
```
