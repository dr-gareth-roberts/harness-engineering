"""Tests for ``fuzz_tool`` and ``fuzz_agent``.

Each test imports Hypothesis through ``pytest.importorskip`` so the
suite degrades gracefully when the ``[fuzz]`` extra is not installed.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("hypothesis")

from harness.agents import Orchestrator, SubAgent  # noqa: E402
from harness.fuzz.runner import (  # noqa: E402
    FuzzReport,
    fuzz_agent,
    fuzz_tool,
)
from harness.hooks import HookRunner  # noqa: E402
from harness.prompts import Message, text  # noqa: E402
from harness.tools import Dispatcher, Tool, ToolCall  # noqa: E402
from harness.tools.schema import ToolResult  # noqa: E402


class _StringIn(BaseModel):
    raw: str


class _NumIn(BaseModel):
    value: int = Field(ge=-1000, le=1000)


def _crash_on_empty(args: _StringIn) -> str:
    if args.raw == "":
        raise ValueError("refuse to handle empty input")
    return args.raw.upper()


def _accepts_anything(args: _StringIn) -> str:
    return f"got {len(args.raw)} chars"


def _double(args: _NumIn) -> int:
    return args.value * 2


async def test_fuzz_tool_finds_failure_on_empty_string_handler() -> None:
    dispatcher = Dispatcher(
        [
            Tool(
                name="upper",
                description="Uppercase a string.",
                input_model=_StringIn,
                handler=_crash_on_empty,
            )
        ]
    )

    report = await fuzz_tool(dispatcher, "upper", n=100, seed=0)
    assert isinstance(report, FuzzReport)
    assert report.total > 0
    # The handler raises on `""`, which the dispatcher converts to an
    # error ToolResult. We expect at least one failure of that shape.
    assert report.failures, "expected fuzzing to discover the empty-string crash"
    empty_failures = [f for f in report.failures if f.input.get("raw") == ""]
    assert empty_failures, 'expected `""` to surface as a failure'
    failure = empty_failures[0]
    assert isinstance(failure.result, ToolResult)
    assert failure.result.is_error is True


async def test_fuzz_tool_clean_handler_yields_no_failures() -> None:
    dispatcher = Dispatcher(
        [
            Tool(
                name="upper",
                description="Length report.",
                input_model=_StringIn,
                handler=_accepts_anything,
            )
        ]
    )

    report = await fuzz_tool(dispatcher, "upper", n=50, seed=0)
    assert report.total == 50
    assert report.failures == []
    assert report.passed == 50
    assert not report  # FuzzReport is falsy when clean


async def test_fuzz_tool_is_deterministic_under_fixed_seed() -> None:
    dispatcher_a = Dispatcher(
        [
            Tool(
                name="dbl",
                description="Doubles.",
                input_model=_NumIn,
                handler=_double,
            )
        ]
    )
    dispatcher_b = Dispatcher(
        [
            Tool(
                name="dbl",
                description="Doubles.",
                input_model=_NumIn,
                handler=_double,
            )
        ]
    )
    a = await fuzz_tool(dispatcher_a, "dbl", n=20, seed=7)
    b = await fuzz_tool(dispatcher_b, "dbl", n=20, seed=7)
    assert a.total == b.total == 20
    assert a.failures == b.failures  # both empty, but same seed twice
    # Re-running with the same seed should hit the same handler results.
    # We can't capture inputs from FuzzReport when clean, so instead
    # verify by re-fuzzing a deliberately broken handler:
    dispatcher_c = Dispatcher(
        [
            Tool(
                name="upper",
                description="Crash on empty.",
                input_model=_StringIn,
                handler=_crash_on_empty,
            )
        ]
    )
    dispatcher_d = Dispatcher(
        [
            Tool(
                name="upper",
                description="Crash on empty.",
                input_model=_StringIn,
                handler=_crash_on_empty,
            )
        ]
    )
    c = await fuzz_tool(dispatcher_c, "upper", n=50, seed=42)
    d = await fuzz_tool(dispatcher_d, "upper", n=50, seed=42)
    assert [f.input for f in c.failures] == [f.input for f in d.failures]


async def test_fuzz_tool_unknown_tool_raises_keyerror() -> None:
    dispatcher = Dispatcher()
    with pytest.raises(KeyError, match="not registered"):
        await fuzz_tool(dispatcher, "missing", n=5)


async def test_fuzz_agent_invariant_passes_with_constant_runner() -> None:
    dispatcher = Dispatcher(
        [
            Tool(
                name="dbl",
                description="Doubles.",
                input_model=_NumIn,
                handler=_double,
            )
        ]
    )

    async def constant_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", "ok")

    orch = Orchestrator(dispatcher, HookRunner(), constant_runner)
    agent = SubAgent(name="bot", system_prompt="x", model="m", allowed_tools=["dbl"])

    def invariant(msg: Message) -> bool:
        return msg.role == "assistant"

    report = await fuzz_agent(orch, agent, "dbl", n=10, invariant=invariant, seed=0)
    assert report.total == 10
    assert not report.failures


async def test_fuzz_agent_records_runner_crashes() -> None:
    dispatcher = Dispatcher(
        [
            Tool(
                name="dbl",
                description="Doubles.",
                input_model=_NumIn,
                handler=_double,
            )
        ]
    )

    async def flaky_runner(agent: SubAgent, messages: list[Message]) -> Message:
        last_user = ""
        for msg in messages:
            for block in msg.content:
                if block.type == "text" and block.text:
                    last_user = block.text
        # Crash whenever the embedded value is positive.
        if "value': " in last_user and "value': -" not in last_user:
            raise RuntimeError("kaboom")
        return text("assistant", "ok")

    orch = Orchestrator(dispatcher, HookRunner(), flaky_runner)
    agent = SubAgent(name="bot", system_prompt="x", model="m", allowed_tools=["dbl"])

    def invariant(msg: Message) -> bool:
        return msg.role == "assistant"

    report = await fuzz_agent(orch, agent, "dbl", n=30, invariant=invariant, seed=0)
    assert report.total == 30
    assert report.failures, "expected at least one runner crash"
    assert any(isinstance(f.exception, RuntimeError) for f in report.failures)


async def test_fuzz_tool_dispatcher_propagates_via_toolcall() -> None:
    """The fuzzer drives inputs through ``Dispatcher.dispatch``.

    We confirm that the ToolCall path is exercised end-to-end by
    pre-dispatching a hand-built call with the same shape and asserting
    both produce a non-error result for a clean handler.
    """

    dispatcher = Dispatcher(
        [
            Tool(
                name="dbl",
                description="Doubles.",
                input_model=_NumIn,
                handler=_double,
            )
        ]
    )
    one_off = await dispatcher.dispatch(ToolCall(name="dbl", arguments={"value": 3}))
    assert one_off.is_error is False
    assert one_off.content == 6

    report = await fuzz_tool(dispatcher, "dbl", n=20, seed=0)
    assert report.failures == []
