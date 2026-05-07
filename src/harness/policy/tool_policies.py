from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from harness.hooks.events import HookDecision, PreToolUse
from harness.hooks.runner import HookRunner

Policy = Callable[[PreToolUse], HookDecision | None]


@dataclass(frozen=True)
class AllowList:
    """Block any tool whose name is not in `allowed`."""

    allowed: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def of(cls, names: Iterable[str]) -> AllowList:
        return cls(frozenset(names))

    def __call__(self, event: PreToolUse) -> HookDecision | None:
        if event.call.name in self.allowed:
            return None
        return HookDecision(
            block=True,
            reason=f"tool {event.call.name!r} not in allow-list",
        )


@dataclass(frozen=True)
class DenyList:
    """Block any tool whose name is in `denied`."""

    denied: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def of(cls, names: Iterable[str]) -> DenyList:
        return cls(frozenset(names))

    def __call__(self, event: PreToolUse) -> HookDecision | None:
        if event.call.name not in self.denied:
            return None
        return HookDecision(
            block=True,
            reason=f"tool {event.call.name!r} is on the deny-list",
        )


@dataclass(frozen=True)
class ArgumentMatcher:
    """Block calls to `tool_name` when `predicate(arguments)` is true.

    Useful for guarding specific argument shapes — e.g. "block shell calls
    whose command contains `rm -rf`" — without writing a custom policy class.
    """

    tool_name: str
    predicate: Callable[[dict[str, Any]], bool]
    reason: str | None = None

    def __call__(self, event: PreToolUse) -> HookDecision | None:
        if event.call.name != self.tool_name:
            return None
        if not self.predicate(event.call.arguments):
            return None
        return HookDecision(
            block=True,
            reason=self.reason or f"argument matcher blocked {self.tool_name!r}",
        )


def attach_pre_tool_policies(runner: HookRunner, *policies: Policy) -> None:
    """Register each policy as a `PreToolUse` handler on `runner`.

    Order matters: the first policy that returns a `block=True` decision
    short-circuits the rest, per `HookRunner.emit` semantics.
    """
    for policy in policies:
        runner.register(PreToolUse, policy)
