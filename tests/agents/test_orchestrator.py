from __future__ import annotations

import asyncio
import contextlib
import time

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner, SessionEnd, SessionStart
from harness.prompts import Message, text
from harness.tools import Dispatcher


def make_orchestrator(runner) -> tuple[Orchestrator, list[type]]:  # type: ignore[no-untyped-def]
    seen: list[type] = []
    hooks = HookRunner()
    hooks.register(SessionStart, lambda e: seen.append(type(e)))
    hooks.register(SessionEnd, lambda e: seen.append(type(e)))
    return Orchestrator(Dispatcher(), hooks, runner), seen


async def test_run_emits_lifecycle_hooks_and_returns_runner_output() -> None:
    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", f"hi {agent.name}")

    orch, seen = make_orchestrator(fake_runner)
    agent = SubAgent(name="bot", system_prompt="be helpful", model="test-model")

    result = await orch.run(agent, [text("user", "hello")])
    assert result.role == "assistant"
    assert result.content[0].text == "hi bot"
    assert seen == [SessionStart, SessionEnd]


async def test_session_end_fires_even_when_runner_raises() -> None:
    async def boom(agent: SubAgent, messages: list[Message]) -> Message:
        raise RuntimeError("explode")

    orch, seen = make_orchestrator(boom)
    agent = SubAgent(name="bot", system_prompt="x", model="test-model")

    with contextlib.suppress(RuntimeError):
        await orch.run(agent, [])
    assert seen == [SessionStart, SessionEnd]


async def test_run_parallel_actually_runs_concurrently() -> None:
    delay = 0.1

    async def slow(agent: SubAgent, messages: list[Message]) -> Message:
        await asyncio.sleep(delay)
        return text("assistant", agent.name)

    orch, _ = make_orchestrator(slow)
    jobs: list[tuple[SubAgent, list[Message]]] = [
        (SubAgent(name=f"a{i}", system_prompt="", model="test-model"), []) for i in range(4)
    ]

    start = time.perf_counter()
    results = await orch.run_parallel(jobs)
    elapsed = time.perf_counter() - start

    assert [r.content[0].text for r in results] == ["a0", "a1", "a2", "a3"]
    # Sequential would take 4 * delay = 0.4s. Concurrent should be much closer to delay.
    assert elapsed < delay * 4 * 0.6, f"expected concurrent execution; took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Hardening: large history handling


async def test_orchestrator_handles_large_history_without_quadratic_blowup() -> None:
    """Pin that a long input history doesn't trigger a quadratic-time
    code path inside the orchestrator (e.g., per-turn copies that grow
    with `n`). 200 prior messages should run in milliseconds with a
    no-op runner — anything more than ~1s indicates an O(n²) regression
    somewhere on the hot path.
    """

    async def echo_runner(agent: SubAgent, messages: list[Message]) -> Message:
        # Touch every message so the runner is at least linear in input.
        return text("assistant", f"saw {len(messages)} messages")

    orch, _ = make_orchestrator(echo_runner)
    agent = SubAgent(name="t", system_prompt="", model="m")

    history: list[Message] = []
    for i in range(200):
        role = "user" if i % 2 == 0 else "assistant"
        history.append(text(role, f"msg {i}"))  # type: ignore[arg-type]

    start = time.perf_counter()
    result = await orch.run(agent, history)
    elapsed = time.perf_counter() - start

    assert result.content[0].text == "saw 200 messages"
    assert elapsed < 1.0, (
        f"orchestrator.run with 200-message history took {elapsed:.3f}s; "
        "suspect a quadratic-time code path"
    )
