# `harness.plan`

`Plan` (Pydantic, JSON-serializable) describes an expected sequence
of `PlannedToolCall`s. `PlanGuardedRunner(real_runner, plan, mode=...)`
enforces it via the contracts DFA — a deviation raises
`PlanViolation`. `derive_plan()` asks a live planner agent to
produce one; `infer_plan_from_records()` mines a plan from past
successful trajectories.

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

# Hand-written plan.
plan = Plan(steps=[
    PlannedToolCall(name="search", required_args={}),
    PlannedToolCall(name="summarize", required_args={}),
])

# Mine one from past records.
plan = infer_plan_from_records(records, mode="superset")

# Wrap the runner.
runner = PlanGuardedRunner(AnthropicRunner(...), plan, mode="superset")
# Deviations raise PlanViolation.
```

Modes:

- `"strict"` — exact match required.
- `"superset"` — extra calls allowed, but the plan's calls must
  appear in order.

## Gotchas

- **Plans are about tool-call structure, not content.** A
  `PlannedToolCall` checks `name` (and optionally `required_args`);
  it doesn't inspect the model's text.
- **`infer_plan_from_records` defaults to `mode="superset"`** so
  deviations don't break sessions that legitimately add steps.
  Use `"strict"` only when you've validated the corpus of past
  successful sessions.
- **`derive_plan` requires a real model call.** Don't use it on a
  hot path; cache the derived plan.
- **`PlanViolation` is raised, not surfaced as an event.** Catch it
  upstream if you want graceful handling.

## Related

- [`harness.contracts`](contracts.md) — the DFA backbone.
- [`examples/plan.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/plan.py),
  [`examples/plan_inference.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/plan_inference.py)
- [`harness.memory`](memory.md) — `infer_plan_from_records` reads `SessionRecord`s.

## API reference

::: harness.plan
