"""`Contract` (pure data) and `Violation` / `ContractViolation`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from harness.contracts.patterns import Pattern

ContractAction = Literal["forbid", "warn", "require"]
ViolationKind = Literal["forbid_match", "require_unmet", "warn_match"]


class Violation(BaseModel):
    """A single contract violation, carrying just the data needed to act on it.

    `message_index` is the offset into the message sequence the DFA observed.
    Offline (`check`) it's the index into `record.messages`. Runtime, it's a
    per-`Orchestrator.run` counter incremented on each synthesized message.
    """

    contract: str
    message_index: int
    kind: ViolationKind
    reason: str | None = None


class Contract(BaseModel):
    """A behavioral contract: name, pattern, action.

    Embedding `Pattern` directly requires `arbitrary_types_allowed=True`
    because `Pattern` subclasses are dataclasses, not Pydantic models. The
    "pure data, no closures" rule lives in the *predicates* and *pattern*
    layers — `Contract` is just the labeled assembly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    pattern: Pattern
    action: ContractAction = "forbid"


class ContractViolation(Exception):
    """Raised at runtime when a `forbid` contract matches or a `require`
    contract is unsatisfied at session end.

    Carries the `Violation` so callers can inspect / log structurally.
    """

    def __init__(self, violation: Violation) -> None:
        self.violation = violation
        super().__init__(
            f"contract {violation.contract!r} violated "
            f"({violation.kind} at message_index={violation.message_index})"
        )
