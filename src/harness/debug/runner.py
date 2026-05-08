from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TextIO

from harness.agents.definition import SubAgent
from harness.agents.orchestrator import Runner
from harness.debug.context import DebugContext
from harness.prompts.messages import Message
from harness.tools.dispatcher import Dispatcher
from harness.tools.schema import ToolCall

if TYPE_CHECKING:
    from harness.debug.repl import DebugRepl  # noqa: F401  (only imported lazily)


class DebugAborted(RuntimeError):
    """Raised when a debug session is terminated via `DebugContext.abort()`."""


BreakPredicate = Callable[[DebugContext], bool]
BreakpointCallback = Callable[[DebugContext], Awaitable[None] | None]


class DebugRunner:
    """A `Runner` wrapper that pauses on configurable breakpoints.

    Wraps any callable matching the `Runner` protocol — including vendor
    runners, `EchoRunner`, `CannedRunner`, or `ReplayRunner` — and adds
    `pdb`-style breakpoints. When `break_on(ctx)` returns True, control
    is handed to either:

      - `breakpoint_callback(ctx)` — a programmatic hook (sync or async),
      - or an interactive REPL on stdin/stdout (when `interactive=True`).

    After the breakpoint exits cleanly, the runner returns the mutated
    next-turn message if `ctx.mutate(...)` was called, otherwise it
    delegates to the wrapped runner. If the breakpoint calls
    `ctx.abort()`, the runner raises `DebugAborted`.

    Example:
        debug = DebugRunner(
            real_runner,
            break_on=lambda c: c.turn_index == 2,
            interactive=True,
            dispatcher=dispatcher,
        )
        orchestrator = Orchestrator(dispatcher, hooks, debug)
    """

    def __init__(
        self,
        real_runner: Runner,
        *,
        break_on: BreakPredicate | None = None,
        breakpoint_callback: BreakpointCallback | None = None,
        interactive: bool = False,
        dispatcher: Dispatcher | None = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        if break_on is not None and breakpoint_callback is None and not interactive:
            raise ValueError(
                "DebugRunner: break_on was set but neither breakpoint_callback "
                "nor interactive=True was provided — the runner would have "
                "no way to handle a breakpoint hit."
            )
        if breakpoint_callback is not None and interactive:
            raise ValueError(
                "DebugRunner: pass exactly one of breakpoint_callback "
                "or interactive=True, not both."
            )

        self._real_runner = real_runner
        self._break_on: BreakPredicate = break_on if break_on is not None else _never
        self._callback = breakpoint_callback
        self._interactive = interactive
        self._dispatcher = dispatcher
        self._stdin = stdin
        self._stdout = stdout

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        ctx = self._build_context(messages)

        if not self._break_on(ctx):
            return await self._real_runner(agent, messages)

        await self._handle_breakpoint(ctx)

        if ctx.aborted:
            raise DebugAborted(f"debug session aborted at turn {ctx.turn_index}")

        if ctx.pending_mutation is not None:
            return ctx.pending_mutation

        return await self._real_runner(agent, messages)

    # ------------------------------------------------------------------ helpers

    def _build_context(self, messages: list[Message]) -> DebugContext:
        last_call = _find_last_tool_use(messages)
        turn_index = sum(1 for m in messages if m.role == "assistant")
        return DebugContext(
            messages,
            last_call=last_call,
            turn_index=turn_index,
            dispatcher=self._dispatcher,
        )

    async def _handle_breakpoint(self, ctx: DebugContext) -> None:
        if self._callback is not None:
            outcome = self._callback(ctx)
            if inspect.isawaitable(outcome):
                await outcome
            return

        if self._interactive:
            from harness.debug.repl import DebugRepl

            repl = DebugRepl(ctx, stdin=self._stdin, stdout=self._stdout)
            await repl.run()
            return

        # Defensive — _validate_config above should have caught this combination.
        raise RuntimeError("DebugRunner: breakpoint hit but no handler is configured")


def _never(_ctx: DebugContext) -> bool:
    return False


def _find_last_tool_use(messages: list[Message]) -> ToolCall | None:
    """Return the most recent `tool_use` ToolCall in the conversation, if any."""
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        for block in reversed(msg.content):
            if block.type == "tool_use" and block.tool_use is not None:
                return block.tool_use
    return None
