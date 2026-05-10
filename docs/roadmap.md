# Roadmap

The full per-wave history (with rationale, decisions, and commit SHAs)
lives in [`progress.md`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/progress.md)
in the repo root. This page summarizes the current state.

## Shipped

| Wave | Feature | Module(s) |
|---|---|---|
| MVP | Tools, prompts, hooks, policy, agents, runner, telemetry, memory, sandbox | core |
| 1 | Counterfactual replay, contracts, fuzz, attribute, diff-eval | replay, contracts, fuzz, attribute |
| 2 | Cache drift, privacy boundary, plan-as-contract, debug REPL | cache, privacy, plan, debug |
| 3 | Speculative tool execution | speculate |
| 4 | OTel sink, plan inference, cross-session predictor, OpenAICompat speculator | telemetry, plan, speculate, runner |
| 5 | One runnable example per module + smoke tests | examples/ |
| 6 | Per-event speculator cancellation (observe + cancel_unobserved) | speculate, runner |
| 7 | DAP for debug REPL (`harness debug --dap`) | debug |
| 8 | Polish + docs site + hardening | — |
| 9 | CI/CD + governance + housekeeping | .github/, docs/ |
| 10 | Vendor runner parity + robustness (cache cap, timeout, replacement, pause/refusal events, OpenAI speculator parity) | runner, hooks |
| 11 | Trace_id / span_id correlation + DAP CLI subprocess test + coverage gate | telemetry, debug, ci |
| 12 | Vision content blocks + Anthropic Files API integration | prompts, runner |
| 13a | Streaming output (`Orchestrator.run_stream` + `StreamingRunner` Protocol) | streaming, agents, runner |
| 13b | Presidio PII detector + DAP pause/step semantics + DAP evaluate opt-in + eager speculator cancel | privacy, debug, speculate |

10 of 10 standout features from the original `designs/standout.md`
are shipped. **All actionable Wave-8 audit gaps cleared.** Tests:
**565 passing, 89% coverage**; mypy strict clean across `src/` and
`tests/` (163 source files); `ruff` + `ruff format --check` clean;
`mkdocs build --strict` clean; `uv build` produces a clean wheel +
sdist; CI matrix runs Python 3.11 / 3.12 / 3.13 with PyPI publishing
via OIDC trusted publishing on tag push.

## Deferred

These are intentional gaps, not omissions:

- **Vendor cassette tests** — recorded Anthropic / OpenAI sessions
  replayed in CI to catch SDK shape drift. Needs a one-time
  recording step against the real APIs, gated on credentials.
- **Anthropic Files API upload helper** — `attach_file(file_id=...)`
  works today; the `upload_file(client, path) -> file_id`
  convenience helper is gated on the same credentials.
- **`OpenAICompatRunner.run_stream()`** — `AnthropicRunner.run_stream()`
  ships in Wave 13a; OpenAI's chat-completions streaming has a
  different delta-by-delta shape and is queued for a follow-up.
- **DAP `step_in` finer granularity** — currently treated as
  `step_over` because `DebugRunner` doesn't yet expose a one-shot
  pre-tool-use breakpoint surface.

## Archive

Wave-by-wave decision logs with rationale and commit SHAs:

- [Wave 1 — replay/contracts/fuzz/attribute/diff-eval](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-1.md)
- [Wave 2 — cache/privacy/plan/debug](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-2.md)
- [Wave 3 — speculative tool execution](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-3.md)
- [Wave 4 — OTel / plan inference / cross-session / OpenAI speculator](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-4.md)
- [Wave 5 — runnable example per module](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-5.md)
- [Wave 6 — per-event speculator cancellation](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-6.md)
- [Wave 7 — DAP for debug REPL](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-7.md)
- [Wave 8 — polish + docs site + hardening](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-8.md)
- [Wave 9 — CI/CD + governance + housekeeping](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-9.md)
- [Wave 10 — vendor runner parity + robustness](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-10.md)
- [Wave 11 — observability + verification](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-11.md)
- [Wave 12 — modality + Files API](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-12.md)
- [Wave 13a — streaming output](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-13a.md)
- Wave 13b lives inline in
  [`progress.md`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/progress.md)
  on the repo root.
