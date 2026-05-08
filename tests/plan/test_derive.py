"""Tests for `derive_plan` — covers test 9 from the design doc.

  9. `derive_plan()` with a fake runner that returns a JSON-schema plan:
     returns a parsed `Plan`.

We use a small closure satisfying the `Runner` protocol that returns an
assistant message whose text is a JSON-encoded plan. `derive_plan` must
parse that into a `Plan` instance.
"""

from __future__ import annotations

import pytest

from harness.agents.definition import SubAgent
from harness.plan import Plan, PlannedToolCall, PlanViolation, derive_plan
from harness.prompts.messages import Message, text


def _planner_returning(json_str: str):  # type: ignore[no-untyped-def]
    """Build a runner that emits the given JSON string as assistant text."""

    async def run(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", json_str)

    return run


# --- Test 9 ---------------------------------------------------------------


async def test_derive_plan_parses_runner_json_into_plan() -> None:
    expected_plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search", arguments_match={"q": "agents"}),
            PlannedToolCall(tool_name="summarize"),
        ],
        mode="superset",
    )
    runner = _planner_returning(expected_plan.model_dump_json())
    planner = SubAgent(name="planner", system_prompt="emit JSON", model="test-model")

    derived = await derive_plan(planner, runner, messages=[text("user", "do things")])
    assert derived == expected_plan


async def test_derive_plan_returns_default_schema_when_caller_does_not_pass_one() -> None:
    """Smoke test: even if the caller passes no `plan_schema`, the helper
    successfully calls the runner and parses output. Keeps the API
    ergonomic for the common case."""
    plan = Plan(steps=[PlannedToolCall(tool_name="echo")])
    runner = _planner_returning(plan.model_dump_json())
    planner = SubAgent(name="p", system_prompt="", model="test-model")
    out = await derive_plan(planner, runner, messages=[])
    assert out == plan


async def test_derive_plan_raises_value_error_on_invalid_json() -> None:
    runner = _planner_returning("not actually json {{{")
    planner = SubAgent(name="p", system_prompt="", model="test-model")
    with pytest.raises(ValueError) as excinfo:
        await derive_plan(planner, runner, messages=[])
    assert "Plan JSON" in str(excinfo.value)


async def test_derive_plan_raises_value_error_on_empty_response() -> None:
    """A runner that returns an assistant message with no text content
    should produce a clear error rather than crash on an empty parse."""

    async def empty_runner(agent: SubAgent, messages: list[Message]) -> Message:
        # Assistant message with no text blocks.
        return Message(role="assistant", content=[])

    planner = SubAgent(name="p", system_prompt="", model="test-model")
    with pytest.raises(ValueError) as excinfo:
        await derive_plan(planner, empty_runner, messages=[])
    assert "no text content" in str(excinfo.value)


async def test_derive_plan_accepts_explicit_schema_argument() -> None:
    """The `plan_schema` kwarg is documented and may be supplied; the
    helper must accept it without errors. (It's the caller's job to do
    something useful with it in their system prompt.)"""
    plan = Plan(steps=[PlannedToolCall(tool_name="search")])
    runner = _planner_returning(plan.model_dump_json())
    planner = SubAgent(name="p", system_prompt="", model="test-model")
    out = await derive_plan(
        planner,
        runner,
        messages=[],
        plan_schema=Plan.model_json_schema(),
    )
    assert out == plan


# --- End-to-end (tests 10 & 11) ------------------------------------------


async def test_e2e_planner_then_executor_success_path() -> None:
    """Test 10: planner produces a plan, executor follows it end-to-end."""
    from harness.agents import Orchestrator
    from harness.hooks import HookRunner
    from harness.plan import PlanGuardedRunner
    from harness.prompts.messages import ContentBlock
    from harness.tools import Dispatcher
    from harness.tools.schema import ToolCall

    expected_plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search", arguments_match={"q": "rust"}),
            PlannedToolCall(tool_name="summarize"),
        ]
    )
    planner_runner = _planner_returning(expected_plan.model_dump_json())
    planner = SubAgent(name="planner", system_prompt="", model="test-model")
    plan = await derive_plan(planner, planner_runner, messages=[])

    # Now wire up the executor side: an orchestrator running a guarded runner.
    def _tool_use(name: str, arguments: dict[str, object], call_id: str) -> Message:
        return Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name=name, arguments=arguments, id=call_id),
                )
            ],
        )

    executor_replies = [
        _tool_use("search", {"q": "rust"}, "c1"),
        _tool_use("summarize", {}, "c2"),
    ]
    iter_replies = iter(executor_replies)

    async def executor_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return next(iter_replies)

    guarded = PlanGuardedRunner(executor_runner, plan)
    orch = Orchestrator(Dispatcher(), HookRunner(), guarded)
    executor = SubAgent(name="exec", system_prompt="", model="test-model")
    # Two turns following the plan.
    await orch.run(executor, [text("user", "kick off")])
    await orch.run(executor, [text("user", "next")])
    assert guarded.step_index == 2
    guarded.finalize()


async def test_e2e_planner_then_executor_deviation_raises() -> None:
    """Test 11: planner produces a plan, executor deviates -> PlanViolation."""
    from harness.agents import Orchestrator
    from harness.hooks import HookRunner
    from harness.plan import PlanGuardedRunner
    from harness.prompts.messages import ContentBlock
    from harness.tools import Dispatcher
    from harness.tools.schema import ToolCall

    expected_plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="summarize"),
        ]
    )
    planner_runner = _planner_returning(expected_plan.model_dump_json())
    planner = SubAgent(name="planner", system_prompt="", model="test-model")
    plan = await derive_plan(planner, planner_runner, messages=[])

    # Executor calls `delete` instead of `summarize` on the second turn.
    def _tool_use(name: str, arguments: dict[str, object], call_id: str) -> Message:
        return Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name=name, arguments=arguments, id=call_id),
                )
            ],
        )

    executor_replies = iter(
        [
            _tool_use("search", {}, "c1"),
            _tool_use("delete", {"id": 99}, "c2"),
        ]
    )

    async def executor_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return next(executor_replies)

    guarded = PlanGuardedRunner(executor_runner, plan)
    orch = Orchestrator(Dispatcher(), HookRunner(), guarded)
    executor = SubAgent(name="exec", system_prompt="", model="test-model")

    # First turn passes; second turn deviates.
    await orch.run(executor, [text("user", "kick off")])
    with pytest.raises(PlanViolation) as excinfo:
        await orch.run(executor, [text("user", "next")])
    err = excinfo.value
    assert err.step_index == 1
    assert err.expected is not None
    assert err.expected.tool_name == "summarize"
    assert err.actual is not None
    assert err.actual.name == "delete"
