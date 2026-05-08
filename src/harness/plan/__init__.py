"""Plan-as-contract: structured agent plans enforced as runtime guards.

A `Plan` is a serializable list of expected tool calls. `PlanGuardedRunner`
wraps any `Runner` and enforces the plan against the tool_use blocks the
inner runner emits — deviation raises `PlanViolation`.

The plan substrate composes with `harness.contracts`: each plan step
compiles to an `Always(HasToolUse(...) & ArgMatches(...))` contract, and
the same `compile_contract` / DFA machinery the contracts module uses is
what evaluates per-step matches inside the guard.
"""

from harness.plan.derive import derive_plan
from harness.plan.guard import PlanGuardedRunner
from harness.plan.plan import Plan, PlanMode, PlannedToolCall, PlanViolation

__all__ = [
    "Plan",
    "PlanGuardedRunner",
    "PlanMode",
    "PlanViolation",
    "PlannedToolCall",
    "derive_plan",
]
