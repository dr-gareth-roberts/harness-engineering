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
