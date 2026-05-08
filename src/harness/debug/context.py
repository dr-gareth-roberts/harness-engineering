from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harness.prompts.messages import Message
from harness.tools.dispatcher import Dispatcher
from harness.tools.schema import ToolCall, ToolResult


class DebugContext:
    """A read/write window into a paused trajectory.

    Readable:
        .messages: list[Message] — full conversation history at the
            breakpoint (a copy; mutations go through .mutate).
        .last_call: ToolCall | None — most recent tool_use block, if any.
        .turn_index: int — count of assistant messages already produced
            (so a break_on `c.turn_index == 5` pauses just before the
            6th assistant turn would be generated).

    Writable (queued; flushed when `DebugRunner` resumes):
        .mutate(replacement: Message) — replaces the next assistant turn.
        .fire(tool_name, args) -> ToolResult — async; runs ad-hoc through
            the dispatcher; doesn't advance the conversation.
        .inspect(callable) — runs a one-shot inspection function over
            this context and returns its result.
        .resume() — exits the breakpoint; the runner will return the
            (possibly mutated) next assistant turn.
        .abort() — marks the context for termination; the runner will
            raise `DebugAborted`.

    The class is intentionally small. New surface should be justified;
    most workflows want a way to inspect, mutate, fire, and choose.
    """

    def __init__(
        self,
        messages: list[Message],
        *,
        last_call: ToolCall | None = None,
        turn_index: int = 0,
        dispatcher: Dispatcher | None = None,
    ) -> None:
        # Defensive copy — caller mutations to the source list shouldn't
        # bleed into the context, and our `.messages` should be a stable
        # snapshot of the conversation at the moment we paused.
        self._messages: list[Message] = list(messages)
        self._last_call = last_call
        self._turn_index = turn_index
        self._dispatcher = dispatcher

        # Queued state — flushed by the runner after resume().
        self._pending_mutation: Message | None = None
        self._resumed = False
        self._aborted = False

    # ------------------------------------------------------------------ readable

    @property
    def messages(self) -> list[Message]:
        """Snapshot of conversation history at the breakpoint."""
        return list(self._messages)

    @property
    def last_call(self) -> ToolCall | None:
        """Most recent assistant `tool_use` block, if any."""
        return self._last_call

    @property
    def turn_index(self) -> int:
        """Count of assistant messages already produced before this pause."""
        return self._turn_index

    @property
    def pending_mutation(self) -> Message | None:
        """The mutation queued for the next turn, if any. Read-only."""
        return self._pending_mutation

    @property
    def resumed(self) -> bool:
        return self._resumed

    @property
    def aborted(self) -> bool:
        return self._aborted

    # ------------------------------------------------------------------ writable

    def mutate(self, replacement: Message) -> None:
        """Queue `replacement` to be returned in place of the next assistant turn.

        Only the most recent call wins — mutate is idempotent for the same
        turn. Pass a fully formed `Message` (typically built with
        `harness.prompts.messages.text(...)`).
        """
        if not isinstance(replacement, Message):
            raise TypeError(f"mutate() expected a Message, got {type(replacement).__name__}")
        self._pending_mutation = replacement

    async def fire(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Dispatch an ad-hoc tool call without advancing the conversation.

        Requires a dispatcher to have been wired into `DebugRunner`. Returns
        the `ToolResult` directly so the caller can inspect it. The call
        does not appear in `.messages` — it's a side channel for the
        debugger to probe state.
        """
        if self._dispatcher is None:
            raise RuntimeError("fire() requires a Dispatcher; pass `dispatcher=` to DebugRunner")
        call = ToolCall(name=tool_name, arguments=dict(args))
        return await self._dispatcher.dispatch(call)

    def inspect(self, fn: Callable[[DebugContext], Any]) -> Any:
        """Run a one-shot inspection function against this context."""
        return fn(self)

    def resume(self) -> None:
        """Exit the breakpoint; the runner will continue with any queued mutation."""
        if self._aborted:
            raise RuntimeError("cannot resume after abort()")
        self._resumed = True

    def abort(self) -> None:
        """Mark the context for termination; the runner will raise DebugAborted."""
        self._aborted = True
