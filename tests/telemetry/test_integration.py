from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.prompts import Message, text
from harness.telemetry import (
    JSONLSink,
    MemorySink,
    OrchestratorTurn,
    Telemetry,
    ToolDispatched,
)
from harness.tools import Dispatcher, Tool, ToolCall


class EchoIn(BaseModel):
    text: str


def echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echo it back.",
        input_model=EchoIn,
        handler=lambda a: a.text,
    )


# ---------------------------------------------------------------------------
# Dispatcher


async def test_dispatcher_emits_tool_dispatched() -> None:
    sink = MemorySink()
    dispatcher = Dispatcher([echo_tool()], telemetry=Telemetry(sink))

    result = await dispatcher.dispatch(ToolCall(name="echo", arguments={"text": "hi"}))
    assert result.is_error is False
    assert len(sink.events) == 1
    e = sink.events[0]
    assert isinstance(e, ToolDispatched)
    assert e.tool_name == "echo"
    assert e.is_error is False
    assert e.duration_ms >= 0
    assert e.arguments == {"text": "hi"}


async def test_dispatcher_emits_on_error() -> None:
    sink = MemorySink()
    dispatcher = Dispatcher([echo_tool()], telemetry=Telemetry(sink))

    await dispatcher.dispatch(ToolCall(name="missing", arguments={}))
    assert len(sink.events) == 1
    assert sink.events[0].is_error is True  # type: ignore[attr-defined]


async def test_dispatcher_default_emits_nothing() -> None:
    """Without telemetry= the dispatcher emits nothing — silent default."""
    sink = MemorySink()  # not wired to anything
    dispatcher = Dispatcher([echo_tool()])
    await dispatcher.dispatch(ToolCall(name="echo", arguments={"text": "hi"}))
    assert sink.events == []


# ---------------------------------------------------------------------------
# Orchestrator


async def test_orchestrator_emits_turn_on_success() -> None:
    sink = MemorySink()

    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", "ok")

    orch = Orchestrator(
        Dispatcher(),
        HookRunner(),
        fake_runner,
        telemetry=Telemetry(sink),
    )
    await orch.run(SubAgent(name="alpha", system_prompt=""), [text("user", "hi")])

    assert len(sink.events) == 1
    e = sink.events[0]
    assert isinstance(e, OrchestratorTurn)
    assert e.agent_name == "alpha"
    assert e.error is None
    assert e.duration_ms >= 0


async def test_orchestrator_emits_turn_on_failure() -> None:
    sink = MemorySink()

    async def boom(agent: SubAgent, messages: list[Message]) -> Message:
        raise RuntimeError("explode")

    orch = Orchestrator(
        Dispatcher(),
        HookRunner(),
        boom,
        telemetry=Telemetry(sink),
    )

    with pytest.raises(RuntimeError, match="explode"):
        await orch.run(SubAgent(name="bot", system_prompt=""), [])

    assert len(sink.events) == 1
    e = sink.events[0]
    assert isinstance(e, OrchestratorTurn)
    assert e.error is not None
    assert "RuntimeError" in e.error
    assert "explode" in e.error


# ---------------------------------------------------------------------------
# Concurrency


async def test_run_parallel_emits_one_event_per_turn() -> None:
    sink = MemorySink()

    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", agent.name)

    orch = Orchestrator(
        Dispatcher(),
        HookRunner(),
        fake_runner,
        telemetry=Telemetry(sink),
    )
    jobs = [(SubAgent(name=f"a{i}", system_prompt=""), []) for i in range(4)]
    results = await orch.run_parallel(jobs)

    assert [r.content[0].text for r in results] == ["a0", "a1", "a2", "a3"]
    assert len(sink.events) == 4
    names = {e.agent_name for e in sink.events}  # type: ignore[attr-defined]
    assert names == {"a0", "a1", "a2", "a3"}


async def test_jsonl_sink_under_run_parallel_writes_clean_lines(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    sink = JSONLSink(target)

    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", "ok")

    orch = Orchestrator(
        Dispatcher(),
        HookRunner(),
        fake_runner,
        telemetry=Telemetry(sink),
    )
    jobs = [(SubAgent(name=f"a{i}", system_prompt=""), []) for i in range(8)]
    await orch.run_parallel(jobs)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    parsed = [json.loads(line) for line in lines]      # all lines must be valid JSON
    assert {p["agent_name"] for p in parsed} == {f"a{i}" for i in range(8)}
