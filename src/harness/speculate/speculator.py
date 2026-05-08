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
2. **`observe`** *(Wave 6, optional)* — event-aware runners call this
   once per `tool_use` block as it arrives in the model's stream, i.e.
   at each `ContentBlockStopEvent` whose block type is `tool_use`. We
   mark a matching pending speculation as "observed" so it survives
   the next step. Skipping `observe` entirely is allowed and equivalent
   to "no speculation was observed" — `cancel_unobserved` then cancels
   the whole pending set.
3. **`cancel_unobserved`** *(Wave 6)* — runner calls once after the
   stream has fully arrived. We cancel any pending speculation that no
   `observe` claimed, freeing the handler runtime that would otherwise
   keep running between stream-end and `end`. Pending entries that *were*
   observed are kept for `try_resolve` to await.
4. **`try_resolve`** — runner calls per `tool_use` block the model emits.
   If any pending speculation matches the call's `(name, arguments)`, we
   await its result, remove it from the pending list, and return the
   result with the model's `tool_use.id` patched in. The runner skips
   its own hook + dispatch cycle for that call. No match → return
   `None` and the runner takes over normally.
5. **`end`** — runner calls in `finally`. We cancel any unmatched
   pending tasks and clear state. Cancellation is best-effort: a tool
   handler that's already running may finish; its result gets discarded.

Cancellation timing note: this implementation cancels at *stream-end*
(via `cancel_unobserved`), not eagerly per `ContentBlockStopEvent`.
Per-event eager cancellation would require a policy decision about
when a speculation is "definitively dead" given the predictions in
flight versus the calls observed so far — non-trivial when
`max_speculations > 1`. Stream-end cancellation captures the bulk of
the win (no handler runtime burned during the post-stream
dispatch phase) without the policy complexity.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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


@dataclass
class _Pending:
    """One in-flight speculation: the predicted call, its task, and whether
    the runner's `observe` has matched it against an emitted `tool_use`.

    `observed=True` means a `ContentBlockStopEvent` for `tool_use` matched
    this entry's `(name, arguments)`; `cancel_unobserved` will leave it
    alone. `observed=False` at `cancel_unobserved` time means the model
    didn't emit a matching call, and we cancel the task to free its
    handler runtime before the runner moves on to dispatch.
    """

    call: ToolCall
    task: asyncio.Task[ToolResult]
    observed: bool = False


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
        self._pending: list[_Pending] = []

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
            self._pending.append(_Pending(call=call, task=task))
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

    async def observe(self, call: ToolCall) -> None:
        """Mark a matching pending speculation as observed.

        Called by event-aware runners (post-Wave 6 `AnthropicRunner`)
        once per `tool_use` block as it arrives in the stream. Only the
        first unobserved match is claimed — duplicate observations of the
        same `(name, arguments)` shape walk down the pending list and
        each claim a separate task, which is the right behavior when the
        speculator launched multiple specs for the same call (rare, but
        permitted).

        No telemetry is emitted here; an `observe` is bookkeeping, not a
        hit. The hit/miss event still fires from `try_resolve`.
        """
        for entry in self._pending:
            if (
                not entry.observed
                and entry.call.name == call.name
                and entry.call.arguments == call.arguments
            ):
                entry.observed = True
                return

    async def cancel_unobserved(self) -> None:
        """Cancel pending speculations that no `observe` claimed.

        Called once after the model's stream has fully arrived but
        before the runner starts dispatching the model's emitted
        `tool_use` blocks. Frees the handler runtime that would
        otherwise keep running between stream-end and `end`.

        Observed entries stay in the pending list for `try_resolve` to
        consume. Calling this when `observe` was never called is
        equivalent to "no observations made, cancel everything" — which
        is fine: the runner will then go to dispatch the model's calls
        normally without speculation.
        """
        unobserved = [entry for entry in self._pending if not entry.observed]
        if not unobserved:
            return
        # Drop the unobserved entries from pending so try_resolve can't
        # later try to await a cancelled task.
        self._pending = [entry for entry in self._pending if entry.observed]
        await self._cancel_entries(unobserved)

    async def try_resolve(self, call: ToolCall) -> ToolResult | None:
        match_idx: int | None = None
        for i, entry in enumerate(self._pending):
            if entry.call.name == call.name and entry.call.arguments == call.arguments:
                match_idx = i
                break

        if match_idx is None:
            if self._telemetry is not None:
                await self._telemetry.emit(SpeculationMiss(tool_name=call.name))
            return None

        # Hit: await the matched task, remove it from pending.
        entry = self._pending.pop(match_idx)
        result = await entry.task
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
        await self._cancel_entries(pending)

    async def _cancel_entries(self, entries: list[_Pending]) -> None:
        for entry in entries:
            if not entry.task.done():
                entry.task.cancel()
        # Drain the cancellations so resources are released before we
        # return. Each cancelled task raises CancelledError when awaited;
        # already-completed tasks just return their result, which we
        # discard.
        for entry in entries:
            try:
                await entry.task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - cleanup phase, swallow tool errors
                # A speculative dispatch raising during cleanup is not
                # the runner's problem — the result was discarded
                # anyway. Don't let it surface from `end`.
                pass
