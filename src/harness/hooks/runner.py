from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TypeVar

from harness.hooks.events import Event, HookDecision

E = TypeVar("E", bound=Event)
HookHandler = Callable[[E], Awaitable[HookDecision | None] | HookDecision | None]


class HookRunner:
    """Registers and dispatches event handlers in registration order.

    The first decision with `block=True` short-circuits further handlers; the runner
    returns the list of decisions collected so far (including the blocker as the last
    element). The runner does not enforce policy — callers inspect the decisions and
    decide what to do.

    Exception discipline: a handler that raises propagates the exception up through
    `emit` and aborts the turn. This is intentionally asymmetric with the
    `Dispatcher`, which converts handler exceptions to `ToolResult(is_error=True)`.
    See `docs/contracts/user-code-execution.md` for how exceptions from hook
    handlers vs tool handlers vs sink emit are handled.
    """

    def __init__(self) -> None:
        self._handlers: list[tuple[type[Event], HookHandler[Event]]] = []

    def register(
        self,
        event_type: type[E],
        handler: HookHandler[E],
    ) -> None:
        # The cast is safe in practice — emit() only invokes handlers whose
        # registered type is an instance of the event being dispatched.
        self._handlers.append((event_type, handler))  # type: ignore[arg-type]

    async def emit(self, event: Event) -> list[HookDecision]:
        decisions: list[HookDecision] = []
        for event_type, handler in self._handlers:
            if not isinstance(event, event_type):
                continue
            outcome = handler(event)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if outcome is None:
                continue
            decisions.append(outcome)
            if outcome.block:
                break
        return decisions
