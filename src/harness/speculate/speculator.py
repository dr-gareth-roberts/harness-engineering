"""`Speculator` — pre-execute predicted tool calls in parallel with the model.

Implements `harness.runner.protocols.SpeculatorProtocol`. Construct once,
pass as `speculator=` to a runner that supports it (today: `AnthropicRunner`).

Lifecycle, per tool-use-loop iteration:

1. **`begin`** — runner calls at iteration start. We ask our `Predictor`
   for likely next calls, filter to the agent's `allowed_tools` ∩
   idempotent (when `only_idempotent=True`), and launch each as an
   `asyncio.Task` that runs the same `PreToolUse` / `dispatcher.dispatch`
   / `PostToolUse` cycle the runner would. Tasks run concurrently with
   the model's stream wait.
2. **`try_resolve`** — runner calls per `tool_use` block the model emits.
   If any pending speculation matches the call's `(name, arguments)`, we
   await its result, remove it from the pending list, and return the
   result with the model's `tool_use.id` patched in. The runner skips
   its own hook + dispatch cycle for that call. No match → return
   `None` and the runner takes over normally.
3. **`end`** — runner calls in `finally`. We cancel any unmatched
   pending tasks and clear state. Cancellation is best-effort: a tool
   handler that's already running may finish; its result gets discarded.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from harness.hooks.events import PostToolUse, PreToolUse
from harness.speculate.events import (
    SpeculationHit,
    SpeculationLaunched,
    SpeculationMiss,
)
from harness.speculate.predictor import Predictor
from harness.tools.schema import ToolCall, ToolResult

if TYPE_CHECKING:
    from harness.agents.definition import SubAgent
    from harness.hooks.runner import HookRunner
    from harness.prompts.messages import Message
    from harness.telemetry.recorder import Telemetry
    from harness.tools.dispatcher import Dispatcher


class Speculator:
    """Predict + pre-execute tool calls during model generation.

    .. warning::
        Idempotency is a *promise* by the tool author. The speculator
        runs `idempotent=True` tools whether the model would have called
        them or not. A tool that says it's idempotent but actually has
        side effects will produce silent duplicate side effects on miss.
        Mark a tool idempotent only if re-running it with the same args
        is observably equivalent to running it once.

    Parameters
    ----------
    predictor:
        The strategy used at `begin` time to pick which tools to fire.
        Ships: `LastCallPredictor`, `SequencePredictor`. Custom
        predictors satisfy the `Predictor` protocol structurally.
    max_speculations:
        Concurrency cap. Default 2 — speculations contend with the
        model's request for SDK / network resources, so a small cap
        keeps the wall-clock overhead bounded on miss.
    only_idempotent:
        When True (default), the speculator filters predictions to
        tools marked `Tool.idempotent=True`. When False, it speculates
        on any tool the agent is allowed to call — only set this if
        you've audited every tool's side-effect profile.
    telemetry:
        Optional sink. When set, fires `SpeculationLaunched` /
        `SpeculationHit` / `SpeculationMiss` events for hit-rate
        accounting.
    """

    def __init__(
        self,
        predictor: Predictor,
        *,
        max_speculations: int = 2,
        only_idempotent: bool = True,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._predictor = predictor
        self._max_speculations = max_speculations
        self._only_idempotent = only_idempotent
        self._telemetry = telemetry
        self._pending: list[tuple[ToolCall, asyncio.Task[ToolResult]]] = []

    async def begin(
        self,
        *,
        history: list[Message],
        agent: SubAgent,
        dispatcher: Dispatcher,
        hooks: HookRunner,
    ) -> None:
        # Build the eligible-tools dict: agent.allowed_tools intersect
        # registered tools, optionally filtered to idempotent only.
        registry = dispatcher.tools
        allowed_names = set(agent.allowed_tools)
        candidates = {
            name: tool
            for name, tool in registry.items()
            if name in allowed_names and (tool.idempotent or not self._only_idempotent)
        }
        if not candidates:
            return

        predictions = self._predictor.predict(
            history=history,
            idempotent_tools=candidates,
            max_predictions=self._max_speculations,
        )

        for call in predictions[: self._max_speculations]:
            if call.name not in candidates:
                # Predictor returned a non-eligible tool; ignore it.
                continue
            task = asyncio.create_task(self._dispatch_via_hooks(call, dispatcher, hooks))
            self._pending.append((call, task))
            if self._telemetry is not None:
                await self._telemetry.emit(SpeculationLaunched(tool_name=call.name))

    async def _dispatch_via_hooks(
        self,
        call: ToolCall,
        dispatcher: Dispatcher,
        hooks: HookRunner,
    ) -> ToolResult:
        """Run the same hook + dispatch cycle the runner uses, so a
        `BlockingPolicy` hook sees speculative calls too.

        Wrapped in a top-level `try/except`: a speculation that raises
        an exception (e.g. a `PreToolUse`/`PostToolUse` handler that
        threw) must NOT propagate out through `try_resolve` and crash
        the runner. The whole point of speculation is that wrong
        predictions are cheap; a buggy hook in the speculative path
        should be observably "miss-shaped" to the runner, not a
        whole-turn crash. We translate any exception into an
        is_error=True `ToolResult` so the speculation either hits
        cleanly or the caller's runner can fall back via try_resolve
        returning a recovered error result.

        Exceptions from `dispatcher.dispatch` itself are already
        caught inside the dispatcher and surfaced as
        `ToolResult(is_error=True)` — the dispatcher never raises
        from a handler bug. The remaining surface this `try/except`
        guards is hook-handler exceptions.
        """
        try:
            decisions = await hooks.emit(PreToolUse(call=call))
            blocked = next((d for d in decisions if d.block), None)
            if blocked is not None:
                result = ToolResult(
                    id=call.id,
                    content=blocked.reason or "blocked by hook (speculation)",
                    is_error=True,
                )
            else:
                result = await dispatcher.dispatch(call)
            await hooks.emit(PostToolUse(call=call, result=result))
            return result
        except Exception as exc:  # noqa: BLE001 - speculative path must not crash the runner
            return ToolResult(
                id=call.id,
                content=f"speculation error: {exc!r}",
                is_error=True,
            )

    async def try_resolve(self, call: ToolCall) -> ToolResult | None:
        match_idx: int | None = None
        for i, (spec_call, _spec_task) in enumerate(self._pending):
            if spec_call.name == call.name and spec_call.arguments == call.arguments:
                match_idx = i
                break

        if match_idx is None:
            if self._telemetry is not None:
                await self._telemetry.emit(SpeculationMiss(tool_name=call.name))
            return None

        # Hit: await the matched task, remove it from pending.
        _spec_call, spec_task = self._pending.pop(match_idx)
        result = await spec_task
        if self._telemetry is not None:
            await self._telemetry.emit(SpeculationHit(tool_name=call.name))
        # Patch the result id to match the model's tool_use id, so the
        # runner's tool_result block correlates correctly.
        if result.id != call.id:
            result = ToolResult(
                id=call.id,
                content=result.content,
                is_error=result.is_error,
            )
        return result

    async def end(self) -> None:
        await self._cancel_pending()

    async def _cancel_pending(self) -> None:
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        for _call, task in pending:
            if not task.done():
                task.cancel()
        # Drain the cancellations so resources are released before we
        # return. Each cancelled task raises CancelledError when awaited;
        # already-completed tasks just return their result, which we
        # discard.
        for _call, task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - cleanup phase, swallow tool errors
                # A speculative dispatch raising during cleanup is not
                # the runner's problem — the result was discarded
                # anyway. Don't let it surface from `end`.
                pass
