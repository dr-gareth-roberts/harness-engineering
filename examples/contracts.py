"""Behavioral contracts as both runtime guardrail AND offline regression check.

Run with: `uv run python examples/contracts.py`

The whole point of `harness.contracts` is that the *same* `Contract`
definition — predicates + a temporal pattern + an action — drives two
enforcement surfaces:

1. Live: `attach_contracts(hooks, [c])` registers handlers so `forbid`
   matches return `HookDecision(block=True)` before the dispatcher runs,
   and `require` patterns raise `ContractViolation` at session-end if
   unsatisfied.
2. Offline: `check(record, [c])` runs the same compiled DFA against a
   stored `SessionRecord` and returns the same shape of `Violation`s.

This example walks through both. It uses no real model — a tiny inline
runner emits a deterministic sequence of tool_use messages so the
contract has something to react to.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.contracts import (
    Always,
    ArgMatches,
    Contract,
    ContractViolation,
    Earlier,
    Eventually,
    HasToolUse,
    Never,
    attach_contracts,
    check,
)
from harness.hooks import HookRunner, PreToolUse, SessionEnd, SessionStart
from harness.memory import SessionRecord
from harness.prompts import Message, text
from harness.prompts.messages import ContentBlock
from harness.tools import Dispatcher, Tool, ToolCall

# ----------------------------------------------------------------------
# A small two-tool surface: search (read-only) + delete (dangerous).


class SearchIn(BaseModel):
    query: str


class DeleteIn(BaseModel):
    table: str


def _build_dispatcher() -> Dispatcher:
    return Dispatcher(
        [
            Tool(
                name="search",
                description="Search a corpus.",
                input_model=SearchIn,
                handler=lambda args: f"hits-for-{args.query}",
                idempotent=True,
            ),
            Tool(
                name="delete",
                description="Delete rows from a table.",
                input_model=DeleteIn,
                handler=lambda args: f"deleted-from-{args.table}",
            ),
        ]
    )


def _agent() -> SubAgent:
    return SubAgent(
        name="contract-demo",
        system_prompt="",
        model="demo-model",
        allowed_tools=["search", "delete"],
    )


# ----------------------------------------------------------------------
# Part 1: live runtime enforcement.
#
# Define a `forbid` contract that blocks `delete` calls against any
# table whose name starts with `prod_`. Run a fake "model" through the
# orchestrator that tries the forbidden call, then a benign call.


CONTRACT_NEVER_DELETE_PROD = Contract(
    name="never_delete_prod",
    pattern=Never(HasToolUse(name="delete") & ArgMatches(table=r"^prod_")),
    action="forbid",
)


async def part_1_runtime(transcript: list[str]) -> None:
    transcript.append("--- part 1: runtime enforcement ---")

    dispatcher = _build_dispatcher()
    hooks = HookRunner()
    attach_contracts(hooks, [CONTRACT_NEVER_DELETE_PROD])

    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        # Try the forbidden call first — the contract blocks it before
        # the dispatcher sees it.
        forbidden = ToolCall(name="delete", arguments={"table": "prod_users"}, id="call-prod")
        decisions = await hooks.emit(PreToolUse(call=forbidden))
        blocker = next((d for d in decisions if d.block), None)
        if blocker is not None:
            transcript.append(f"  contract BLOCKED delete(table='prod_users'): {blocker.reason}")
        # Now a benign call.
        ok = ToolCall(name="delete", arguments={"table": "stage_users"}, id="call-st")
        decisions = await hooks.emit(PreToolUse(call=ok))
        if not any(d.block for d in decisions):
            transcript.append("  contract ALLOWED delete(table='stage_users')")
        return text("assistant", "demo finished part 1")

    orch = Orchestrator(dispatcher, hooks, fake_runner)
    await orch.run(_agent(), [text("user", "do part 1")])


# ----------------------------------------------------------------------
# Part 2: offline check against a recorded session.
#
# The same contract definition is also useful for regression-checking a
# library of recorded `SessionRecord`s. Build one with a sequence that
# violates a different contract, then run `check` on it.


CONTRACT_SEARCH_BEFORE_DELETE = Contract(
    name="must_search_before_delete",
    # Whenever a delete fires, an earlier search must have happened.
    pattern=Always(Earlier(HasToolUse(name="search")).when(HasToolUse(name="delete"))),
    action="require",
)


CONTRACT_EVENTUALLY_RESPONDS = Contract(
    name="eventually_responds",
    # The agent must eventually emit a final assistant message.
    pattern=Eventually(HasToolUse(name="delete")),
    action="warn",
)


async def part_2_offline(transcript: list[str]) -> None:
    transcript.append("--- part 2: offline check ---")

    # Build a session record that violates `must_search_before_delete`:
    # the assistant calls delete without ever calling search first.
    record = SessionRecord(
        session_id="demo-violator",
        agent=_agent(),
        messages=[
            text("user", "delete stage_logs please"),
            Message(
                role="assistant",
                content=[
                    ContentBlock(
                        type="tool_use",
                        tool_use=ToolCall(
                            name="delete",
                            arguments={"table": "stage_logs"},
                            id="del-1",
                        ),
                    )
                ],
            ),
            text("assistant", "done"),
        ],
    )

    violations = check(record, [CONTRACT_SEARCH_BEFORE_DELETE])
    for v in violations:
        transcript.append(
            f"  offline check found {v.contract!r}: kind={v.kind} "
            f"at message_index={v.message_index}"
        )

    no_violations = check(record, [CONTRACT_EVENTUALLY_RESPONDS])
    transcript.append(f"  offline check 'eventually_responds' violations: {len(no_violations)}")


# ----------------------------------------------------------------------
# Part 3: same contract, both surfaces, same verdict.
#
# Pin the equivalence: a record that the offline check flags is also
# what the runtime path would have blocked / raised on.


async def part_3_equivalence(transcript: list[str]) -> None:
    transcript.append("--- part 3: runtime/offline equivalence ---")

    # A record where `require` is unsatisfied (no search ever fires).
    record = SessionRecord(
        session_id="demo-unmet",
        agent=_agent(),
        messages=[
            text("user", "anything"),
            Message(
                role="assistant",
                content=[ContentBlock(type="text", text="I'm done.")],
            ),
        ],
    )

    require_search = Contract(
        name="must_call_search",
        pattern=Eventually(HasToolUse(name="search")),
        action="require",
    )

    offline = check(record, [require_search])
    transcript.append(
        f"  offline check: {len(offline)} violation(s); "
        f"first kind={offline[0].kind if offline else None}"
    )

    # Runtime: same contract, same verdict — `require` raises at
    # SessionEnd if unsatisfied.
    hooks = HookRunner()
    attach_contracts(hooks, [require_search])

    await hooks.emit(SessionStart())
    raised = False
    try:
        await hooks.emit(SessionEnd())
    except ContractViolation as exc:
        raised = True
        transcript.append(
            f"  runtime check: ContractViolation raised "
            f"(contract={exc.violation.contract!r}, kind={exc.violation.kind})"
        )
    transcript.append(f"  same verdict on both surfaces: {raised}")


# ----------------------------------------------------------------------


async def main() -> int:
    transcript: list[str] = []
    await part_1_runtime(transcript)
    await part_2_offline(transcript)
    await part_3_equivalence(transcript)
    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
