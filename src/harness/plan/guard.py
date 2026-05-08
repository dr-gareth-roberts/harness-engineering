"""`PlanGuardedRunner` — wraps any `Runner` and enforces a `Plan`.

The guard implements Option 1 from the design: each plan step compiles to
a one-shot `Always(HasToolUse(...) & ArgMatches(...))` contract, and each
contract is compiled to a `DFA` lazily as its step becomes active. A
single tick of the DFA against a synthesized one-block assistant message
decides pass/fail for that step.

This means the load-bearing call site is `compile_contract(...)` in
`_compile_step_dfa` — that's where the plan crosses into the contracts
substrate. The guard never reimplements predicate matching or pattern
state machines; it composes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from harness.contracts import compile_contract
from harness.contracts.dfa import DFA
from harness.plan.plan import Plan, PlannedToolCall, PlanViolation
from harness.prompts.messages import ContentBlock, Message
from harness.tools.schema import ToolCall

if TYPE_CHECKING:
    from harness.agents.definition import SubAgent
    from harness.agents.orchestrator import Runner


class PlanGuardedRunner:
    """Wrap any `Runner` and enforce a `Plan` on the tool_use blocks it emits.

    Usage:
        guarded = PlanGuardedRunner(real_runner, plan)
        orch = Orchestrator(dispatcher, hooks, guarded)
        await orch.run(executor_agent, messages)
        guarded.finalize()  # raises PlanViolation if plan is unfinished

    The guard ticks one DFA per plan step. A fresh DFA is compiled the
    first time each step becomes active (DFAs are stateful, so we don't
    reuse them across runs).

    State persists across multiple `__call__` invocations: a multi-turn
    orchestrator session shares one step pointer, so a plan can describe
    a sequence that spans several turns.
    """

    def __init__(self, real_runner: Runner, plan: Plan) -> None:
        self._real_runner = real_runner
        self._plan = plan
        # Compile contracts up-front (cheap; pure data). DFAs are compiled
        # lazily per active step in `_compile_step_dfa`.
        self._contracts = plan.to_contracts()
        self._step_index = 0
        self._finalized = False

    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def step_index(self) -> int:
        """Current plan-step pointer. Tests can read this to assert progress."""
        return self._step_index

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        """Invoke the inner runner, then walk its tool_use blocks against the plan.

        Raises `PlanViolation` immediately on any deviation that the active
        mode forbids. If the message contains no tool_use blocks (e.g. a
        final text-only assistant turn), the guard is a passthrough.
        """
        if self._finalized:
            raise RuntimeError(
                "PlanGuardedRunner.finalize() was called; runner is no longer usable. "
                "Construct a new guard for a new run."
            )

        message = await self._real_runner(agent, messages)
        for tool_use in _iter_tool_uses(message):
            self._consume(tool_use)
        return message

    def finalize(self) -> None:
        """End-of-run check: in `strict` and `superset` modes, the plan must
        be fully consumed. `subset` allows leftover plan steps.

        Idempotent: safe to call after a `PlanViolation` already fired
        (the violation already surfaced; this just marks the guard as done).
        """
        if self._finalized:
            return
        self._finalized = True
        if self._plan.mode == "subset":
            return
        if self._step_index < len(self._plan.steps):
            unmet = self._plan.steps[self._step_index]
            raise PlanViolation(
                expected=unmet,
                actual=None,
                step_index=self._step_index,
            )

    # --- internals -----------------------------------------------------

    def _consume(self, tool_use: ToolCall) -> None:
        """Match one observed tool_use against the active plan state.

        Strict / superset: the next step at `_step_index` must accept this
        call. If it does, advance. If not:
          * strict   — `PlanViolation` (any deviation fails)
          * superset — allow the extra call (don't advance)

        Strict at end-of-plan with another tool_use → extra-call violation.

        Subset: scan forward in `plan.steps[step_index:]` for the first step
        whose DFA accepts this call. Found → advance past it. Not found →
        wrong-tool violation.
        """
        mode = self._plan.mode
        steps = self._plan.steps

        if self._step_index >= len(steps):
            # Plan exhausted; behaviour depends on mode.
            if mode == "superset":
                return
            # strict / subset: extra calls past the plan are violations.
            raise PlanViolation(
                expected=None,
                actual=tool_use,
                step_index=len(steps),
            )

        if mode == "subset":
            self._consume_subset(tool_use)
            return

        # strict / superset: try to match the current step.
        active_step = steps[self._step_index]
        if self._step_accepts(self._step_index, tool_use):
            self._step_index += 1
            return

        if mode == "superset":
            # Extra call that doesn't advance the plan; allowed in superset.
            return

        # strict: any mismatch fails.
        raise PlanViolation(
            expected=active_step,
            actual=tool_use,
            step_index=self._step_index,
        )

    def _consume_subset(self, tool_use: ToolCall) -> None:
        """Subset matching: skip past steps that don't match the call.

        Scan `plan.steps[step_index:]` for the first step that accepts the
        call. Found at offset `k` → advance pointer past it. Not found →
        wrong-tool violation referencing the current step.
        """
        steps = self._plan.steps
        for offset in range(len(steps) - self._step_index):
            absolute_index = self._step_index + offset
            if self._step_accepts(absolute_index, tool_use):
                self._step_index = absolute_index + 1
                return
        # No remaining step matched: the call doesn't fit anywhere in the
        # remaining plan. Use the current step as `expected` for clarity.
        raise PlanViolation(
            expected=steps[self._step_index],
            actual=tool_use,
            step_index=self._step_index,
        )

    def _step_accepts(self, step_index: int, tool_use: ToolCall) -> bool:
        """True iff the step at `step_index` accepts `tool_use`.

        Compiles a fresh DFA for the step (DFAs are stateful; we use each
        instance for exactly one decision) and ticks it with a synthesized
        single-block assistant message.

        This is the load-bearing call site: `compile_contract` is the
        canonical bridge from `Plan` semantics to the contracts substrate.
        """
        dfa = self._compile_step_dfa(step_index)
        message = _synthesize_tool_use_message(tool_use)
        violation = dfa.tick(message)
        return violation is None

    def _compile_step_dfa(self, step_index: int) -> DFA:
        """The single line that ties plans to the contracts DFA.

        See module docstring; this is what the brief asks us to "point at".
        """
        return compile_contract(self._contracts[step_index])


def _iter_tool_uses(message: Message) -> list[ToolCall]:
    """Extract every tool_use block from an assistant message, in order.

    Non-assistant messages yield nothing — only the model's outgoing tool
    calls are subject to plan enforcement. Models that emit multiple tool
    calls in one turn produce multiple blocks; we walk them sequentially.
    """
    if message.role != "assistant":
        return []
    out: list[ToolCall] = []
    for block in message.content:
        if block.type == "tool_use" and block.tool_use is not None:
            out.append(block.tool_use)
    return out


def _synthesize_tool_use_message(tool_use: ToolCall) -> Message:
    """Build a single-block assistant message carrying just this tool call.

    The contract predicates (`HasToolUse`, `ArgMatches`) match against
    assistant tool_use messages, so we feed the DFA a synthesized message
    of exactly that shape — one tool_use block, no text. This isolates
    each call so per-step decisions are independent.
    """
    return Message(
        role="assistant",
        content=[ContentBlock(type="tool_use", tool_use=tool_use)],
    )


# Re-export for callers who only import from .plan
__all__ = ["PlanGuardedRunner", "PlanViolation", "Plan", "PlannedToolCall"]
