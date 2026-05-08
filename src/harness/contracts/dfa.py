"""DFA: the single message-stream evaluator that runtime and offline share.

A DFA wraps a `Contract` and its compiled `PatternState`. Its `tick` and
`finalize` are the only two entry points the rest of the system uses, which
guarantees test 19 holds — both runtime and offline reach a verdict via the
same code path.
"""

from __future__ import annotations

from harness.contracts.contract import Contract, Violation, ViolationKind
from harness.contracts.patterns import PatternState
from harness.prompts.messages import Message


class DFA:
    """One DFA per `Contract`. Stateful within a single session lifetime."""

    def __init__(self, contract: Contract) -> None:
        self.contract = contract
        self._state: PatternState = contract.pattern.compile()
        self._index = -1

    @property
    def index(self) -> int:
        """Index of the most recently ticked message (-1 if none yet)."""
        return self._index

    def tick(self, message: Message) -> Violation | None:
        """Advance the DFA by one message; return a violation if one fires now."""
        self._index += 1
        outcome = self._state.tick(message)
        if not outcome.violated:
            return None
        kind = self._tick_kind()
        return Violation(
            contract=self.contract.name,
            message_index=self._index,
            kind=kind,
        )

    def finalize(self) -> Violation | None:
        """Called at end-of-session. Only `require` semantics produce here."""
        outcome = self._state.finalize()
        if not outcome.violated:
            return None
        # If we finalize as violated, the index records *where* the session
        # ended — useful for users scanning a record. We use the count of
        # observed messages, which equals `_index + 1` (starts at -1).
        end_index = max(self._index, 0)
        return Violation(
            contract=self.contract.name,
            message_index=end_index,
            kind="require_unmet",
        )

    def reset(self) -> None:
        """Reset to initial state. Called between `Orchestrator.run` lifecycles."""
        self._state.reset()
        self._index = -1

    # --- internals -----------------------------------------------------

    def _tick_kind(self) -> ViolationKind:
        """Map the contract's action to the right kind for an immediate hit."""
        action = self.contract.action
        if action == "forbid":
            return "forbid_match"
        if action == "warn":
            return "warn_match"
        # `require` patterns fire immediate violations only when their pattern
        # has a hard-fail mid-stream (e.g. `Always(...)`); otherwise they fire
        # in `finalize`. For the immediate path, treat as a forbid_match —
        # callers can still differentiate by `Contract.action`.
        return "forbid_match"


def compile_contract(contract: Contract) -> DFA:
    """Public alias for `DFA(contract)` — the canonical "compile" entry point."""
    return DFA(contract)
