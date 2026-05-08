"""`Plan` (Pydantic, JSON-serializable), `PlannedToolCall`, `PlanViolation`.

A plan is a serializable list of expected tool calls. Each `PlannedToolCall`
declares the tool name and (optionally) constraints on its arguments — either
as an exact-match dict, a per-field regex dict, or both. Both `None` means
"any arguments are acceptable for this step".

Crucially, `Plan.to_contracts()` compiles the plan into a list of
`harness.contracts.Contract` objects, one per step. Each contract is
`Always(HasToolUse(name=...) & ArgMatches(...))`, and the runtime guard
(`PlanGuardedRunner`) compiles each into a DFA via the canonical
`compile_contract` entry point. This is the load-bearing architectural
decision: plans share the contracts substrate — same DFA underneath.

`PlannedToolCall.arguments_match` carries *literal values* compared
field-by-field after `str(...)`-coercion (escape into a regex), while
`arguments_regex` carries Python `re` patterns matched via `re.search`. Both
match modes ride on the same `ArgMatches` predicate.

Callable matchers are intentionally *not* supported on the serializable
`Plan` model: closures don't round-trip through JSON. Callers wanting a
predicate-based check can construct a `Contract` directly and use the
contracts surface — `Plan` is the data model.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from harness.contracts import Always, ArgMatches, Contract, HasToolUse
from harness.contracts.predicates import Predicate
from harness.tools.schema import ToolCall

PlanMode = Literal["strict", "superset", "subset"]


class PlannedToolCall(BaseModel):
    """A single expected tool invocation in a plan.

    `arguments_match`: each key must be present and the stringified value
    must equal (after regex-escaping) the planned value. Use this for
    exact-match semantics.

    `arguments_regex`: each key must be present and the stringified value
    must match (`re.search`) the supplied regex. Use this when the model
    legitimately produces variable formatting (whitespace, case, ordering).

    Both `None`: any arguments accepted (the canonical "tool name only"
    plan step).

    Both set: both checks must pass. Useful when a few fields demand exact
    values and others tolerate regex matches.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    arguments_match: dict[str, Any] | None = None
    arguments_regex: dict[str, str] | None = None

    def to_predicate(self) -> Predicate:
        """Compile this step into a `Predicate` that matches an assistant
        message containing a tool_use block satisfying the constraints."""
        predicate: Predicate = HasToolUse(name=self.tool_name)
        if self.arguments_match is not None:
            predicate = predicate & ArgMatches(
                **{k: re.escape(str(v)) for k, v in self.arguments_match.items()}
            )
        if self.arguments_regex is not None:
            predicate = predicate & ArgMatches(**self.arguments_regex)
        return predicate

    def to_contract(self, name: str) -> Contract:
        """Compile this step into an `Always(HasToolUse(...) & ArgMatches(...))`
        contract. `Always` here is correct: the guard ticks each contract's
        DFA exactly once (with the synthesized one-block message), so the
        first-and-only tick decides pass/fail. Using `Always` keeps the same
        DFA semantics ("first miss is a violation") shared with the contracts
        runtime.
        """
        return Contract(name=name, pattern=Always(self.to_predicate()), action="forbid")


class Plan(BaseModel):
    """An ordered list of expected tool calls, plus a deviation mode.

    Modes:
      * `strict`   — every tool_use must match the corresponding plan step
                     in order; no extra calls; plan must be exhausted at run end.
      * `superset` — plan is a *minimum* sequence: every step must be hit in
                     order, extra calls beyond the plan are allowed, plan must
                     still be exhausted at run end.
      * `subset`   — plan is a *maximum* sequence: each tool_use must match
                     some remaining step (skipping intermediate steps allowed);
                     plan need not be exhausted; a wrong tool that matches no
                     remaining step still fails.
    """

    model_config = ConfigDict(frozen=True)

    steps: list[PlannedToolCall]
    mode: PlanMode = "strict"

    def to_contracts(self) -> list[Contract]:
        """Compile the plan to one Contract per step.

        Names are stable (`plan.step.<index>:<tool>`) so violations can be
        traced back to the originating step. Callers typically pass these to
        `compile_contract` to get DFAs for runtime evaluation.
        """
        return [
            step.to_contract(name=f"plan.step.{i}:{step.tool_name}")
            for i, step in enumerate(self.steps)
        ]


class PlanViolation(RuntimeError):
    """Raised by `PlanGuardedRunner` when the executor deviates from the plan.

    Carries the structured deviation:
      * `expected` — the plan step that was active when the violation fired
                     (or `None` if the run extended past the plan in strict
                     mode and the offender is an *extra* call).
      * `actual`   — the offending tool call observed (or `None` if the run
                     ended early in strict/superset and the violation is
                     "missing-required-call").
      * `step_index` — the index into `Plan.steps` of the step in question.
                       For end-of-run-unmet violations, this is the index of
                       the first unmet step. For extra-call violations, this
                       is `len(plan.steps)` (one past the end).
    """

    def __init__(
        self,
        *,
        expected: PlannedToolCall | None,
        actual: ToolCall | None,
        step_index: int,
        message: str | None = None,
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.step_index = step_index
        if message is None:
            message = self._default_message()
        super().__init__(message)

    def _default_message(self) -> str:
        if self.actual is None:
            return (
                f"plan violation at step {self.step_index}: "
                f"expected tool {self.expected!r} but run ended"
            )
        if self.expected is None:
            return (
                f"plan violation at step {self.step_index}: "
                f"unexpected extra tool call {self.actual.name!r} beyond plan"
            )
        return (
            f"plan violation at step {self.step_index}: "
            f"expected {self.expected.tool_name!r}, got {self.actual.name!r}"
        )
