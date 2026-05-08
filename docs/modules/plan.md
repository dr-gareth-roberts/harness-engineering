# `harness.plan`

`Plan` (Pydantic, JSON-serializable) describes an expected sequence
of `PlannedToolCall`s. `PlanGuardedRunner(real_runner, plan, mode=...)`
enforces it via the contracts DFA — a deviation raises `PlanViolation`.
`derive_plan()` asks a live planner agent to produce one;
`infer_plan_from_records()` mines a plan from past successful
trajectories.

::: harness.plan
