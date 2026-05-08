"""Temporal patterns over message streams.

Each pattern compiles to a tiny state machine (`PatternState`) ticked once per
message. The state's `tick` returns a `_TickOutcome` that the surrounding DFA
turns into a `Violation`. Patterns themselves are pure data — no closures —
so a `Contract` is fully serializable / hashable / loggable.

The four patterns:

* `Always(p)`     — every message must satisfy `p`. First miss is a violation.
                    For composite inner patterns (e.g. `Always(Earlier(...).when(...))`)
                    the inner pattern decides relevance, not `Always`.
* `Eventually(p)` — at least one message must satisfy `p`. Violated only at end-of-session.
* `Earlier(req).when(trigger)` — every trigger message must be preceded by
                    some earlier req-matching message. Violated on the first
                    trigger that fires before req has matched.
* `Never(p)`      — `p` must never match. Violated as soon as it does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from harness.contracts.predicates import Predicate
from harness.prompts.messages import Message

TickKind = Literal["forbid_match", "require_unmet", "warn_match"]


@dataclass(frozen=True)
class _TickOutcome:
    """Outcome of one `PatternState.tick` call.

    ``violated`` means the pattern just transitioned to a violated state on
    this exact message. The DFA labels it with the contract action's
    appropriate `kind` (forbid_match / warn_match for immediate violations,
    require_unmet for finalize-time ones).
    """

    violated: bool


_OK = _TickOutcome(violated=False)
_BAD = _TickOutcome(violated=True)


class PatternState:
    """Compiled, mutable per-contract state. One instance per active contract."""

    def tick(self, message: Message) -> _TickOutcome:  # pragma: no cover - abstract
        raise NotImplementedError

    def finalize(self) -> _TickOutcome:
        """Called at end-of-session. Default: no terminal violation."""
        return _OK

    def reset(self) -> None:  # pragma: no cover - default no-op
        """Reset to initial state; called between `Orchestrator.run` lifecycles."""
        return None


class Pattern:
    """Pure-data pattern definition. ``compile()`` produces a fresh `PatternState`."""

    def compile(self) -> PatternState:  # pragma: no cover - abstract
        raise NotImplementedError


# --- Never -----------------------------------------------------------------


@dataclass(frozen=True)
class Never(Pattern):
    predicate: Predicate

    def compile(self) -> PatternState:
        return _NeverState(predicate=self.predicate)


class _NeverState(PatternState):
    def __init__(self, predicate: Predicate) -> None:
        self._predicate = predicate
        self._violated = False

    def tick(self, message: Message) -> _TickOutcome:
        if self._violated:
            # Already triggered once; subsequent ticks stay quiet so the DFA
            # only emits a violation on the first hit.
            return _OK
        if self._predicate.matches(message):
            self._violated = True
            return _BAD
        return _OK

    def reset(self) -> None:
        self._violated = False


# --- Eventually ------------------------------------------------------------


@dataclass(frozen=True)
class Eventually(Pattern):
    predicate: Predicate

    def compile(self) -> PatternState:
        return _EventuallyState(predicate=self.predicate)


class _EventuallyState(PatternState):
    def __init__(self, predicate: Predicate) -> None:
        self._predicate = predicate
        self._satisfied = False

    def tick(self, message: Message) -> _TickOutcome:
        if not self._satisfied and self._predicate.matches(message):
            self._satisfied = True
        return _OK

    def finalize(self) -> _TickOutcome:
        return _OK if self._satisfied else _BAD

    def reset(self) -> None:
        self._satisfied = False


# --- Always ----------------------------------------------------------------


@dataclass(frozen=True)
class Always(Pattern):
    """Every message tick must succeed.

    If `inner` is a `Pattern` (e.g. `Earlier(...).when(...)`), `Always` simply
    propagates the inner state's tick — relevance gating happens inside the
    inner pattern. If `inner` is a bare `Predicate`, every message must match.
    """

    inner: Pattern | Predicate

    def compile(self) -> PatternState:
        if isinstance(self.inner, Pattern):
            return _AlwaysPatternState(inner=self.inner.compile())
        return _AlwaysPredicateState(predicate=self.inner)


class _AlwaysPredicateState(PatternState):
    def __init__(self, predicate: Predicate) -> None:
        self._predicate = predicate
        self._violated = False

    def tick(self, message: Message) -> _TickOutcome:
        if self._violated:
            return _OK
        if not self._predicate.matches(message):
            self._violated = True
            return _BAD
        return _OK

    def reset(self) -> None:
        self._violated = False


class _AlwaysPatternState(PatternState):
    def __init__(self, inner: PatternState) -> None:
        self._inner = inner
        self._violated = False

    def tick(self, message: Message) -> _TickOutcome:
        if self._violated:
            return _OK
        outcome = self._inner.tick(message)
        if outcome.violated:
            self._violated = True
        return outcome

    def finalize(self) -> _TickOutcome:
        if self._violated:
            return _OK
        return self._inner.finalize()

    def reset(self) -> None:
        self._violated = False
        self._inner.reset()


# --- Earlier(req).when(trigger) -------------------------------------------


@dataclass(frozen=True)
class EarlierBuilder:
    """Builder returned by `Earlier(req)`. Call `.when(trigger)` to finish."""

    requirement: Predicate

    def when(self, trigger: Predicate) -> _EarlierWhen:
        return _EarlierWhen(requirement=self.requirement, trigger=trigger)


def Earlier(requirement: Predicate) -> EarlierBuilder:  # noqa: N802 - matches public API name
    return EarlierBuilder(requirement=requirement)


@dataclass(frozen=True)
class _EarlierWhen(Pattern):
    """A trigger fires; some earlier message must have matched the requirement."""

    requirement: Predicate
    trigger: Predicate

    def compile(self) -> PatternState:
        return _EarlierWhenState(requirement=self.requirement, trigger=self.trigger)


class _EarlierWhenState(PatternState):
    def __init__(self, requirement: Predicate, trigger: Predicate) -> None:
        self._requirement = requirement
        self._trigger = trigger
        self._req_seen = False
        self._violated = False

    def tick(self, message: Message) -> _TickOutcome:
        if self._violated:
            return _OK
        # Update history *before* the trigger check: a single message that
        # matches both req and trigger would otherwise spuriously fail.
        # However the contract reads as "some earlier message" — so a message
        # that matches both is *not* sufficient, the req must have matched
        # strictly before. We therefore check trigger first, then update.
        is_trigger = self._trigger.matches(message)
        if is_trigger and not self._req_seen:
            self._violated = True
            return _BAD
        if self._requirement.matches(message):
            self._req_seen = True
        return _OK

    def reset(self) -> None:
        self._req_seen = False
        self._violated = False
