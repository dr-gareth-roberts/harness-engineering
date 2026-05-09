from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

from harness.agents.definition import SubAgent
from harness.hooks.events import SessionEnd, SessionStart
from harness.hooks.runner import HookRunner
from harness.prompts.messages import Message
from harness.streaming import StreamEvent, StreamingRunner
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

    @property
    def telemetry(self) -> Telemetry | None:
        """Public, read-only view of the configured telemetry sink (may be `None`).

        Lets wrappers like `harness.replay.counterfactual` re-instantiate an
        orchestrator with the same telemetry without poking at the private
        attribute.
        """
        return self._telemetry

    async def run(self, agent: SubAgent, messages: list[Message]) -> Message:
        start = time.perf_counter()
        err: str | None = None
        # Open a telemetry session_scope for the duration of the run if
        # a telemetry recorder is configured. Inside the scope, all
        # emitted events (this turn's OrchestratorTurn, downstream
        # ToolDispatched events from the dispatcher) inherit the same
        # `trace_id`. Each tool dispatch then opens its own
        # `span_scope`, so the tree is session → turn-span → tool-spans.
        if self._telemetry is not None:
            return await self._run_with_telemetry(agent, messages, start, err)
        return await self._run_without_telemetry(agent, messages)

    async def _run_with_telemetry(
        self,
        agent: SubAgent,
        messages: list[Message],
        start: float,
        err: str | None,
    ) -> Message:
        assert self._telemetry is not None
        async with self._telemetry.session_scope(), self._telemetry.span_scope():
            await self.hooks.emit(SessionStart())
            try:
                return await self._runner(agent, messages)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                await self.hooks.emit(SessionEnd())
                from harness.telemetry.events import OrchestratorTurn

                duration_ms = (time.perf_counter() - start) * 1000.0
                await self._telemetry.emit(
                    OrchestratorTurn(
                        agent_name=agent.name,
                        duration_ms=duration_ms,
                        error=err,
                    )
                )

    async def _run_without_telemetry(
        self,
        agent: SubAgent,
        messages: list[Message],
    ) -> Message:
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

    async def run_stream(
        self,
        agent: SubAgent,
        messages: list[Message],
    ) -> AsyncIterator[StreamEvent]:
        """Stream events from the runner instead of awaiting a final message.

        Wave 13a (#9 in `docs/plan.md`). The runner must implement the
        `StreamingRunner` protocol — i.e., expose a `run_stream(agent,
        messages) -> AsyncIterator[StreamEvent]` method. Today
        `AnthropicRunner` is the only one that does;
        `OpenAICompatRunner` is queued for a follow-up.

        Lifecycle and ordering match `run()` as closely as possible:

        - `SessionStart` hook fires before the first stream event
          (and before the runner's stream is touched).
        - `SessionEnd` hook + `OrchestratorTurn` telemetry fire after
          the stream ends (success or exception), exactly once.
        - Telemetry `session_scope` + `span_scope` wrap the entire
          run so downstream `Dispatcher.dispatch()` calls (which the
          runner does internally) inherit the trace_id. The runner
          itself is responsible for any tool-dispatch span_scope
          nesting it cares about.

        Raises `TypeError` immediately if the configured runner
        doesn't implement `StreamingRunner` — the call site decides
        whether to fall back to `run()` or surface the failure.
        """
        if not isinstance(self._runner, StreamingRunner):
            raise TypeError(
                "Orchestrator.run_stream requires a runner that implements "
                "StreamingRunner (a `run_stream` method); "
                f"got {type(self._runner).__name__}. Use `run()` for "
                "non-streaming runners or upgrade the runner."
            )

        if self._telemetry is None:
            async for event in self._stream_without_telemetry(agent, messages):
                yield event
            return

        async for event in self._stream_with_telemetry(agent, messages):
            yield event

    async def _stream_without_telemetry(
        self,
        agent: SubAgent,
        messages: list[Message],
    ) -> AsyncIterator[StreamEvent]:
        # `isinstance(self._runner, StreamingRunner)` was already
        # enforced by the public entry point; the assertion is here
        # only so mypy knows `run_stream` is callable.
        assert isinstance(self._runner, StreamingRunner)
        await self.hooks.emit(SessionStart())
        try:
            async for event in self._runner.run_stream(agent, messages):
                yield event
        finally:
            await self.hooks.emit(SessionEnd())

    async def _stream_with_telemetry(
        self,
        agent: SubAgent,
        messages: list[Message],
    ) -> AsyncIterator[StreamEvent]:
        assert self._telemetry is not None
        assert isinstance(self._runner, StreamingRunner)
        start = time.perf_counter()
        err: str | None = None
        async with self._telemetry.session_scope(), self._telemetry.span_scope():
            await self.hooks.emit(SessionStart())
            try:
                async for event in self._runner.run_stream(agent, messages):
                    yield event
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                await self.hooks.emit(SessionEnd())
                from harness.telemetry.events import OrchestratorTurn

                duration_ms = (time.perf_counter() - start) * 1000.0
                await self._telemetry.emit(
                    OrchestratorTurn(
                        agent_name=agent.name,
                        duration_ms=duration_ms,
                        error=err,
                    )
                )
