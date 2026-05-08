"""Behavioral contracts: declarative invariants over agent trajectories.

Same `Contract` definition runs as a live runtime guardrail (via `HookRunner`)
and as an offline check against a recorded `SessionRecord`. The compiled DFA
is shared between both surfaces so verdicts are identical.
"""

from harness.contracts.check import check
from harness.contracts.contract import Contract, ContractViolation, Violation
from harness.contracts.patterns import (
    Always,
    Earlier,
    EarlierBuilder,
    Eventually,
    Never,
    Pattern,
)
from harness.contracts.predicates import (
    ArgMatches,
    HasToolUse,
    Predicate,
    RoleIs,
    TextMatches,
)
from harness.contracts.runtime import attach_contracts

__all__ = [
    "Always",
    "ArgMatches",
    "Contract",
    "ContractViolation",
    "Earlier",
    "EarlierBuilder",
    "Eventually",
    "HasToolUse",
    "Never",
    "Pattern",
    "Predicate",
    "RoleIs",
    "TextMatches",
    "Violation",
    "attach_contracts",
    "check",
]
