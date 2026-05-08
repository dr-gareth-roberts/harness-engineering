"""Offline contract evaluation against a `SessionRecord`."""

from __future__ import annotations

from harness.contracts.contract import Contract, Violation
from harness.contracts.dfa import compile_contract
from harness.memory.record import SessionRecord


def check(record: SessionRecord, contracts: list[Contract]) -> list[Violation]:
    """Evaluate `contracts` against the messages in `record`.

    Builds one DFA per contract, ticks each through every message in order,
    then finalizes. Returns the aggregated violations. The same DFA implementation
    is used at runtime — see `runtime.attach_contracts` — so a contract that
    would have blocked at runtime also fails offline, and vice versa.
    """
    dfas = [compile_contract(c) for c in contracts]
    violations: list[Violation] = []
    for message in record.messages:
        for dfa in dfas:
            v = dfa.tick(message)
            if v is not None:
                violations.append(v)
    for dfa in dfas:
        v = dfa.finalize()
        if v is not None:
            violations.append(v)
    return violations
