from __future__ import annotations

import pytest

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.memory import SessionRecord
from harness.prompts import text
from harness.replay import ReplayMismatch, ReplayRunner
from harness.tools import Dispatcher


async def test_returns_replies_in_order() -> None:
    runner = ReplayRunner([text("assistant", "first"), text("assistant", "second")])
    agent = SubAgent(name="x", system_prompt="", model="test-model")

    first = await runner(agent, [text("user", "hi")])
    second = await runner(agent, [text("user", "again")])

    assert first.content[0].text == "first"
    assert second.content[0].text == "second"


async def test_exhausted_raises_replay_mismatch() -> None:
    runner = ReplayRunner([text("assistant", "only")])
    agent = SubAgent(name="x", system_prompt="", model="test-model")
    await runner(agent, [text("user", "hi")])
    with pytest.raises(ReplayMismatch, match="exhausted"):
        await runner(agent, [text("user", "more")])


async def test_remaining_decrements() -> None:
    runner = ReplayRunner([text("assistant", "a"), text("assistant", "b")])
    agent = SubAgent(name="x", system_prompt="", model="test-model")
    assert runner.remaining == 2
    await runner(agent, [text("user", "x")])
    assert runner.remaining == 1
    await runner(agent, [text("user", "y")])
    assert runner.remaining == 0


def test_from_record_keeps_only_assistant_messages() -> None:
    record = SessionRecord(
        session_id="s1",
        agent=SubAgent(name="x", system_prompt="", model="test-model"),
        messages=[
            text("system", "be helpful"),
            text("user", "hi"),
            text("assistant", "hello"),
            text("user", "again"),
            text("assistant", "world"),
        ],
    )
    runner = ReplayRunner.from_record(record)
    assert runner.remaining == 2


async def test_drives_orchestrator_end_to_end() -> None:
    runner = ReplayRunner([text("assistant", "ack")])
    orch = Orchestrator(Dispatcher(), HookRunner(), runner)
    agent = SubAgent(name="bot", system_prompt="", model="test-model")

    result = await orch.run(agent, [text("user", "hi")])
    assert result.role == "assistant"
    assert result.content[0].text == "ack"
