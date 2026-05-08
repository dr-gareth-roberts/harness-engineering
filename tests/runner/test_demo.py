from __future__ import annotations

import pytest

from harness.agents import SubAgent
from harness.prompts import text
from harness.runner.demo import CannedRunner, EchoRunner


def make_agent() -> SubAgent:
    return SubAgent(name="t", system_prompt="", model="test-model")


# ---------------------------------------------------------------------------
# EchoRunner


async def test_echo_runner_returns_last_user_text() -> None:
    runner = EchoRunner()
    result = await runner(make_agent(), [text("user", "hello")])
    assert result.role == "assistant"
    assert result.content[0].text == "hello"


async def test_echo_runner_uses_last_user_message_only() -> None:
    runner = EchoRunner()
    result = await runner(
        make_agent(),
        [
            text("user", "first"),
            text("assistant", "ok"),
            text("user", "second"),
        ],
    )
    assert result.content[0].text == "second"


async def test_echo_runner_with_prefix() -> None:
    runner = EchoRunner(prefix="echo: ")
    result = await runner(make_agent(), [text("user", "world")])
    assert result.content[0].text == "echo: world"


async def test_echo_runner_returns_empty_when_no_user_text() -> None:
    runner = EchoRunner()
    result = await runner(make_agent(), [text("system", "be helpful")])
    assert result.content[0].text == ""


# ---------------------------------------------------------------------------
# CannedRunner


async def test_canned_runner_returns_replies_in_order() -> None:
    runner = CannedRunner(["one", "two"])
    a = await runner(make_agent(), [text("user", "x")])
    b = await runner(make_agent(), [text("user", "y")])
    assert a.content[0].text == "one"
    assert b.content[0].text == "two"


async def test_canned_runner_exhausted_raises() -> None:
    runner = CannedRunner(["only"])
    await runner(make_agent(), [text("user", "x")])
    with pytest.raises(RuntimeError, match="exhausted"):
        await runner(make_agent(), [text("user", "y")])
