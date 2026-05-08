# Roadmap

The full per-wave history (with rationale, decisions, and commit SHAs)
lives in [`progress.md`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/progress.md)
in the repo root. This page summarizes the current state.

## Shipped

| Wave | Feature | Module(s) |
|---|---|---|
| MVP | Tools, prompts, hooks, policy, agents, runner, telemetry, memory, sandbox | core |
| 1 | Counterfactual replay, contracts, fuzz, attribute, diff-eval | replay, contracts, fuzz, attribute, replay |
| 2 | Cache drift, privacy boundary, plan-as-contract, debug REPL | cache, privacy, plan, debug |
| 3 | Speculative tool execution | speculate |
| 4 | OTel sink, plan inference, cross-session predictor, OpenAICompat speculator | telemetry, plan, speculate, runner |
| 5 | One runnable example per module + smoke tests | examples/ |
| 6 | Per-event speculator cancellation (observe + cancel_unobserved) | speculate, runner |
| 7 | DAP for debug REPL (`harness debug --dap`) | debug |
| 8 | Polish + docs site (this page) + hardening | — |

10 of 10 standout features from the original `designs/standout.md`
are shipped. Tests: 495+; mypy strict clean across 82 source files.

## Deferred

These are intentional gaps, not omissions:

- **ML-based privacy detection** — Microsoft Presidio / AWS Comprehend
  adapters under the existing `Detector` protocol. v1 is regex +
  entropy.
- **Eager per-block speculator cancellation** — today an unmatched
  speculation gets cancelled at stream-end (after the model finishes
  speaking). A future refinement could cancel mid-stream when a single
  emitted `tool_use` makes the speculation definitively a miss; the
  protocol shape (`observe()` per block) leaves room for it without
  breaking changes.

## Archive

Wave-by-wave decision logs with rationale and commit SHAs:

- [Wave 1 — replay/contracts/fuzz/attribute/diff-eval](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-1.md)
- [Wave 2 — cache/privacy/plan/debug](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/docs/waves/wave-2.md)
- Waves 3+ live inline in
  [`progress.md`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/progress.md)
  on the repo root.
