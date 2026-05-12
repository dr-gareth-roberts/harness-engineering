# `harness.plan`

`Plan` (Pydantic, JSON-serializable) describes an expected sequence
of `PlannedToolCall`s and carries its own `mode`.
`PlanGuardedRunner(real_runner, plan)` enforces it via the contracts
DFA — deviation raises `PlanViolation`. `derive_plan()` asks a live
planner agent to produce one; `infer_plan_from_records()` mines a
plan from past successful trajectories.

## When to reach for this

- You want the agent's tool-call sequence to follow a known plan
  (e.g., "search → summarize → respond"), not freelance.
- You want runtime enforcement *and* the same plan as a CI check.
- You want to mine a plan from successful past sessions and use it
  as a guardrail going forward.

## Quick example

```python
from harness import (
    Plan, PlannedToolCall, PlanGuardedRunner, AnthropicRunner,
    infer_plan_from_records,
)

# Hand-written plan. `mode` belongs on Plan, not on the runner.
plan = Plan(
    steps=[
        PlannedToolCall(tool_name="search"),
        PlannedToolCall(
            tool_name="summarize",
            arguments_regex={"max_words": r"^\d+$"},
        ),
    ],
    mode="superset",
)

# Or: mine one from past records (defaults to mode="superset").
plan = infer_plan_from_records(records)

# Wrap the runner. Deviations raise PlanViolation.
runner = PlanGuardedRunner(AnthropicRunner(dispatcher, hooks), plan)
```

Modes (on `Plan`):

- `"strict"` — every tool_use must match the corresponding step in
  order; no extras; plan must be exhausted at run end.
- `"superset"` — plan is a *minimum* sequence: every step must hit
  in order, extras are allowed, plan must still be exhausted.
- `"subset"` — plan is a *maximum* sequence: each tool_use must
  match some remaining step (skipping is fine); plan need not be
  exhausted; an offending call that matches no remaining step still
  fails.

## Gotchas

- **Plans are about tool-call structure, not content.** A
  `PlannedToolCall` checks `tool_name` (and optionally
  `arguments_match` for exact-match dict, `arguments_regex` for
  per-field regex); it doesn't inspect the model's text.
- **`infer_plan_from_records` defaults to `mode="superset"`** so
  deviations don't break sessions that legitimately add steps.
  Pass `mode="strict"` only when you've validated the corpus.
- **`derive_plan` requires a real model call.** Don't use it on a
  hot path; cache the derived plan.
- **`PlanViolation` is raised, not surfaced as an event.** Catch it
  upstream if you want graceful handling. It carries `expected`,
  `actual`, and `step_index` for diagnostics.

## Related

- [`harness.contracts`](contracts.md) — the DFA backbone.
- [`examples/plan.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/plan.py),
  [`examples/plan_inference.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/plan_inference.py)
- [`harness.memory`](memory.md) — `infer_plan_from_records` reads `SessionRecord`s.

## API reference

::: harness.plan
