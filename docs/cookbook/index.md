# Cookbook

Concrete recipes for the most common workflows. Each recipe follows
the same shape: **problem**, **solution sketch**, **working code**,
**gotchas**, **related modules**.

If you haven't done the [Quickstart](../quickstart.md) yet, start
there — these recipes assume the basic agent flow (tool, dispatcher,
hooks, runner, orchestrator) is familiar.

## Recipes

| Recipe | What you'll learn |
|---|---|
| [Redact PII before sending to a model](redact-pii.md) | `PrivacyBoundary` wraps any runner; regex + entropy detectors out of the box; Presidio adapter for NLP-backed PII. |
| [Block prompt injection with a `PromptSubmit` contract](prompt-submit-contracts.md) | A `forbid` contract attached to the orchestrator's hooks raises `PromptBlocked` from `Session.send` *before* the runner is invoked — no round-trip cost when the prompt is rejected. |
| [Replay a session for evaluation](replay-evaluation.md) | `ReplayRunner` for deterministic playback; `run_eval` for batch evaluation; `diff_eval` for cross-provider matrices. |
| [Debug a bad trajectory](debug-a-trajectory.md) | `DebugRunner` wraps a runner; pause on a configurable predicate; inspect, mutate, fire ad-hoc tool calls; or drive from VS Code via DAP. |
| [Fuzz a tool with Hypothesis](fuzz-a-tool.md) | `fuzz_tool` drives Pydantic-typed inputs through `Dispatcher.dispatch`; surface failures as a structured report. |
| [Cache + speculate for latency wins](cache-and-speculate.md) | `PrefixWatcher` audits prompt-cache drift; `Speculator` pre-executes likely tool calls in parallel. |
| [Observability with OpenTelemetry](observability.md) | Pluggable `Sink`s; `OpenTelemetrySink` lights up Jaeger / Tempo / Honeycomb; trace_id / span_id correlation across orchestrator + dispatcher. |

## What's not here

These recipes target evaluating developers — concrete value
demonstrations. For deeper architectural / extension content:

- [**Architecture**](../architecture.md) — the protocol seams and
  composition pattern.
- [**Module reference**](../modules/tools.md) — per-module API.
- [**FAQ**](../faq.md) — common pitfalls + when-to-use-which-X.
