from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from harness.agents.definition import SubAgent
from harness.hooks.events import SessionEnd, SessionStart
from harness.hooks.runner import HookRunner
from harness.prompts.messages import Message
from harness.tools.dispatcher import Dispatcher

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
    ) -> None:
        self.dispatcher = dispatcher
        self.hooks = hooks
        self._runner = runner

    async def run(self, agent: SubAgent, messages: list[Message]) -> Message:
        await self.hooks.emit(SessionStart())
        try:
            return await self._runner(agent, messages)
        finally:
            await self.hooks.emit(SessionEnd())

    async def run_parallel(
        self,
        jobs: list[tuple[SubAgent, list[Message]]],
    ) -> list[Message]:
        return await asyncio.gather(*(self.run(agent, msgs) for agent, msgs in jobs))
