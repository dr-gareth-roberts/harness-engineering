from __future__ import annotations

import json
from pathlib import Path

from harness.agents import SubAgent
from harness.contracts import (
    Always,
    ArgMatches,
    Contract,
    Earlier,
    Eventually,
    HasToolUse,
    Never,
    RoleIs,
    TextMatches,
    attach_contracts,
    check,
)
from harness.hooks import HookRunner, PreToolUse, SessionEnd, SessionStart
from harness.memory import SessionRecord
from harness.prompts import assistant_tool_use, text
from harness.tools import ToolCall


def _agent() -> SubAgent:
    return SubAgent(name="t", system_prompt="x", model="m")


def _delete_prod() -> ToolCall:
    return ToolCall(name="delete", arguments={"table": "prod_users"}, id="c1")


def _delete_stage() -> ToolCall:
    return ToolCall(name="delete", arguments={"table": "stage_users"}, id="c2")


def _search_q() -> ToolCall:
    return ToolCall(name="search", arguments={"q": "x"}, id="cs")


def test_check_returns_empty_for_clean_record() -> None:
    record = SessionRecord(
        session_id="s1",
        agent=_agent(),
        messages=[
            text("user", "hello"),
            assistant_tool_use(_search_q()),
            text("assistant", "Answer: hi"),
        ],
    )
    contract = Contract(
        name="search_then_answer",
        pattern=Always(
            Earlier(HasToolUse(name="search")).when(RoleIs("assistant") & TextMatches(r"^Answer:"))
        ),
        action="forbid",
    )
    assert check(record, [contract]) == []


def test_check_reports_forbid_match_with_message_index() -> None:
    record = SessionRecord(
        session_id="s2",
        agent=_agent(),
        messages=[
            text("user", "delete prod"),
            assistant_tool_use(_delete_stage()),  # idx 1: ok
            assistant_tool_use(_delete_prod()),  # idx 2: violation
        ],
    )
    contract = Contract(
        name="never_delete_prod",
        pattern=Never(HasToolUse(name="delete") & ArgMatches(table=r"^prod_")),
        action="forbid",
    )
    violations = check(record, [contract])
    assert len(violations) == 1
    v = violations[0]
    assert v.contract == "never_delete_prod"
    assert v.message_index == 2
    assert v.kind == "forbid_match"


def test_check_reports_require_unmet_at_end_of_record() -> None:
    record = SessionRecord(
        session_id="s3",
        agent=_agent(),
        messages=[
            text("user", "no search here"),
            text("assistant", "I have no idea."),
        ],
    )
    contract = Contract(
        name="must_call_search",
        pattern=Eventually(HasToolUse(name="search")),
        action="require",
    )
    violations = check(record, [contract])
    assert len(violations) == 1
    v = violations[0]
    assert v.contract == "must_call_search"
    assert v.kind == "require_unmet"


async def test_offline_check_matches_runtime_verdict_for_same_contract() -> None:
    """Same Contract definition: runtime block iff offline reports a violation."""
    contract = Contract(
        name="never_delete_prod",
        pattern=Never(HasToolUse(name="delete") & ArgMatches(table=r"^prod_")),
        action="forbid",
    )

    # ---- Runtime path: a forbidden call should produce a HookDecision(block=True).
    hooks = HookRunner()
    attach_contracts(hooks, [contract])
    await hooks.emit(SessionStart())
    decisions = await hooks.emit(PreToolUse(call=_delete_prod_call_with_id()))
    runtime_blocked = any(d.block for d in decisions)
    runtime_violation_name = decisions[-1].reason if decisions else ""
    await hooks.emit(SessionEnd())

    # ---- Offline path: build the equivalent record from the same call shape.
    record = SessionRecord(
        session_id="cmp",
        agent=_agent(),
        messages=[assistant_tool_use(_delete_prod_call_with_id())],
    )
    offline_violations = check(record, [contract])

    assert runtime_blocked is True
    assert "never_delete_prod" in (runtime_violation_name or "")
    assert len(offline_violations) == 1
    assert offline_violations[0].contract == "never_delete_prod"
    assert offline_violations[0].kind == "forbid_match"


def _delete_prod_call_with_id() -> ToolCall:
    return ToolCall(name="delete", arguments={"table": "prod_orders"}, id="cx")


def test_check_loads_session_jsonl_and_finds_violations(tmp_path: Path) -> None:
    """Integration: round-trip a SessionRecord through JSON and run check."""
    record = SessionRecord(
        session_id="s4",
        agent=_agent(),
        messages=[
            text("user", "drop prod"),
            assistant_tool_use(_delete_prod()),
        ],
    )
    jsonl_path = tmp_path / "session.jsonl"
    jsonl_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    # Parse back.
    line = jsonl_path.read_text(encoding="utf-8").strip().splitlines()[0]
    revived = SessionRecord.model_validate_json(line)
    # Sanity check: parser preserved the tool_use payload.
    assert json.loads(line)["messages"][1]["content"][0]["type"] == "tool_use"

    contract = Contract(
        name="never_delete_prod",
        pattern=Never(HasToolUse(name="delete") & ArgMatches(table=r"^prod_")),
        action="forbid",
    )
    violations = check(revived, [contract])
    assert [v.contract for v in violations] == ["never_delete_prod"]
    assert violations[0].message_index == 1
