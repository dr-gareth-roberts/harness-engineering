"""Per-message predicates: data-carrying classes that compose with `&` / `|`.

Predicates are deliberately *not* lambdas — closures don't serialize, and we
want `Contract` definitions to be inspectable / hashable / loggable. Composition
goes through `_And` / `_Or` instances that recursively evaluate their children.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from harness.prompts.messages import Message, Role


class Predicate:
    """Base class for per-message predicates.

    Subclasses implement `matches(message: Message) -> bool`. Composition with
    `&` / `|` returns `_And` / `_Or` — also predicates — so arbitrary boolean
    expressions remain pure data.
    """

    def matches(self, message: Message) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def __and__(self, other: Predicate) -> _And:
        return _And(left=self, right=other)

    def __or__(self, other: Predicate) -> _Or:
        return _Or(left=self, right=other)


@dataclass(frozen=True)
class _And(Predicate):
    left: Predicate
    right: Predicate

    def matches(self, message: Message) -> bool:
        return self.left.matches(message) and self.right.matches(message)


@dataclass(frozen=True)
class _Or(Predicate):
    left: Predicate
    right: Predicate

    def matches(self, message: Message) -> bool:
        return self.left.matches(message) or self.right.matches(message)


@dataclass(frozen=True)
class HasToolUse(Predicate):
    """Matches an *assistant* message containing a `tool_use` block.

    With `name=None` matches any tool_use block; otherwise matches only blocks
    whose `tool_use.name` equals `name`.
    """

    name: str | None = None

    def matches(self, message: Message) -> bool:
        if message.role != "assistant":
            return False
        for block in message.content:
            if block.type != "tool_use" or block.tool_use is None:
                continue
            if self.name is None or block.tool_use.name == self.name:
                return True
        return False


@dataclass(frozen=True)
class TextMatches(Predicate):
    """Matches a message whose concatenated text content matches the regex.

    The regex uses `re.search`, so callers can anchor explicitly with `^` / `$`.
    """

    regex: str

    def matches(self, message: Message) -> bool:
        text = "".join(block.text or "" for block in message.content if block.type == "text")
        if not text:
            return False
        return re.search(self.regex, text) is not None


@dataclass(frozen=True)
class RoleIs(Predicate):
    """Strict role match against `message.role`."""

    role: Role

    def matches(self, message: Message) -> bool:
        return message.role == self.role


@dataclass(frozen=True)
class ArgMatches(Predicate):
    """For assistant messages with a `tool_use` block, each kwarg's regex must
    match the stringified value of the corresponding argument.

    A message matches if *any* tool_use block in it satisfies all the regexes.
    Missing arguments do not match (the contract author opted into that name).
    """

    # `frozenset[tuple[str, str]]` keeps the dataclass hashable / frozen even
    # though logically this is a `dict[str, str]`. Use the `field_regexes`
    # property to access as a dict.
    _items: frozenset[tuple[str, str]]

    def __init__(self, **field_regexes: str) -> None:
        # Bypass frozen-dataclass guards via object.__setattr__.
        object.__setattr__(self, "_items", frozenset(field_regexes.items()))

    @property
    def field_regexes(self) -> dict[str, str]:
        return dict(self._items)

    def matches(self, message: Message) -> bool:
        if message.role != "assistant":
            return False
        regexes = self.field_regexes
        if not regexes:
            # Vacuous: any tool_use matches.
            return any(block.type == "tool_use" for block in message.content)
        for block in message.content:
            if block.type != "tool_use" or block.tool_use is None:
                continue
            args = block.tool_use.arguments
            if all(
                field_name in args and re.search(pattern, str(args[field_name])) is not None
                for field_name, pattern in regexes.items()
            ):
                return True
        return False
