from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from harness.agents.definition import SubAgent
from harness.hooks.events import SessionEnd, SessionStart
from harness.hooks.runner import HookRunner
from harness.prompts.messages import Message
from harness.tools.dispatcher import Dispatcher

if TYPE_CHECKING:
    from harness.telemetry.recorder import Telemetry

Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]


class Orchestrator:
    """Drives a sub-agent through a single turn, emitting lifecycle hooks.

    The orchestrator is deliberately model-agnostic: callers inject `runner`, the
    function that actually talks to a model. Tool dispatch inside `runner` is the
    caller's responsibility — `dispatcher` is exposed so the caller can use it.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        hooks: HookRunner,
        runner: Runner,
        *,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.hooks = hooks
        self._runner = runner
        self._telemetry = telemetry

    async def run(self, agent: SubAgent, messages: list[Message]) -> Message:
        start = time.perf_counter()
        err: str | None = None
        await self.hooks.emit(SessionStart())
        try:
            return await self._runner(agent, messages)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await self.hooks.emit(SessionEnd())
            if self._telemetry is not None:
                from harness.telemetry.events import OrchestratorTurn

                duration_ms = (time.perf_counter() - start) * 1000.0
                await self._telemetry.emit(
                    OrchestratorTurn(
                        agent_name=agent.name,
                        duration_ms=duration_ms,
                        error=err,
                    )
                )

    async def run_parallel(
        self,
        jobs: list[tuple[SubAgent, list[Message]]],
    ) -> list[Message]:
        return await asyncio.gather(*(self.run(agent, msgs) for agent, msgs in jobs))
