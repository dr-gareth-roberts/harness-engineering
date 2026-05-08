"""Structural protocols runners accept as constructor kwargs.

These live here (not inside the vendor-specific runner files) so each
vendor runner can import them without having to redeclare the same shape,
and so feature modules (`harness.cache`, `harness.speculate`) can satisfy
the protocol without taking a runtime dependency on any one vendor SDK.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from harness.agents.definition import SubAgent
    from harness.hooks.runner import HookRunner
    from harness.prompts.messages import Message
    from harness.tools.dispatcher import Dispatcher
    from harness.tools.schema import ToolCall, ToolResult


class PrefixWatcherProtocol(Protocol):
    """Anything callable as `await watcher.fingerprint(request)`.

    The `harness.cache.PrefixWatcher` (Wave-2 feature #3) implements this.
    Runners call `await prefix_watcher.fingerprint(request)` once per
    iteration of their tool-use loop, immediately before the model call.

    The `request` dict is whatever the runner is about to send to its SDK
    — its exact shape is vendor-specific, but the watcher only needs it
    to compute a stable byte-level fingerprint, so it treats the value
    as opaque JSON-serializable data.
    """

    async def fingerprint(self, request: dict[str, Any]) -> None: ...


class SpeculatorProtocol(Protocol):
    """Pre-execute likely tool calls while the model is still generating.

    The `harness.speculate.Speculator` (Wave-3 feature #5) implements this.
    Runners that accept a `speculator=` kwarg call:

        await speculator.begin(history=..., agent=..., dispatcher=..., hooks=...)

    once at the start of each tool-use-loop iteration — before the model
    call. The speculator predicts likely tool calls from `history`, filters
    to idempotent tools (per the speculator's own config), and launches
    them as `asyncio` tasks that run concurrently with the model's
    generation. Each speculation goes through the same `PreToolUse` /
    `dispatcher.dispatch` / `PostToolUse` flow the runner would use, so
    `BlockingPolicy` hooks see speculative calls too.

        await speculator.observe(call)

    is called by event-aware runners (`AnthropicRunner` post-Wave 6) once
    per `tool_use` block as it arrives in the model's stream — i.e. at
    each `ContentBlockStopEvent` whose block type is `tool_use`. The
    speculator marks any matching pending speculation as "observed" so
    `cancel_unobserved` won't cancel it. This is purely a hint: a runner
    that doesn't iterate stream events MAY skip `observe` entirely, in
    which case `cancel_unobserved` becomes a no-op and `try_resolve`
    still works the same way. Implementations MUST tolerate `observe`
    being skipped.

        await speculator.cancel_unobserved()

    fires once after the model's stream has fully arrived — i.e. after
    the runner has observed every `tool_use` block the model emitted but
    before the runner starts dispatching them. The speculator cancels
    any pending speculations that no `observe` matched. This frees the
    handler runtime that would otherwise be wasted between stream-end
    and `end`. Skipping it is safe: `end` still cancels everything in a
    `finally` block.

        result = await speculator.try_resolve(call)

    is called *once per tool_use block* the model emits — before the
    runner falls back to its normal dispatch path. A non-`None` return
    means the speculation hit: the result has already been produced
    through the full hook flow, and the runner should NOT fire
    `PreToolUse` / `PostToolUse` or call the dispatcher itself for this
    call. A `None` return means the speculator wants the runner to
    handle this one normally.

        await speculator.end()

    fires at the end of each iteration (in a `finally` block, so it runs
    on errors too). It cancels any leftover speculations that didn't get
    matched and resets internal state so the next iteration's `begin`
    starts fresh.

    Idempotency contract: the speculator only runs tools the author has
    marked `idempotent=True` on the `Tool` definition. That mark is a
    *promise* — speculative execution will run a tool whether the model
    would have called it or not, so a tool that says it's idempotent but
    has side effects will produce silent duplicates on miss. The
    speculator does not enforce this; it trusts the flag.
    """

    async def begin(
        self,
        *,
        history: list[Message],
        agent: SubAgent,
        dispatcher: Dispatcher,
        hooks: HookRunner,
    ) -> None: ...

    async def observe(self, call: ToolCall) -> None: ...

    async def cancel_unobserved(self) -> None: ...

    async def try_resolve(self, call: ToolCall) -> ToolResult | None: ...

    async def end(self) -> None: ...
