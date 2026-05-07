from __future__ import annotations

from harness.hooks import HookRunner, PreToolUse
from harness.policy import (
    AllowList,
    ArgumentMatcher,
    DenyList,
    attach_pre_tool_policies,
)
from harness.tools import ToolCall


def event(name: str, **arguments: object) -> PreToolUse:
    return PreToolUse(call=ToolCall(name=name, arguments=dict(arguments)))


def test_allow_list_passes_through_for_listed_tool() -> None:
    policy = AllowList.of({"echo", "add"})
    assert policy(event("echo")) is None
    assert policy(event("add")) is None


def test_allow_list_blocks_unlisted_tool() -> None:
    policy = AllowList.of({"echo"})
    decision = policy(event("delete"))
    assert decision is not None
    assert decision.block is True
    assert "'delete'" in (decision.reason or "")


def test_deny_list_blocks_listed_tool() -> None:
    policy = DenyList.of({"shell"})
    decision = policy(event("shell"))
    assert decision is not None
    assert decision.block is True


def test_deny_list_passes_through_for_other_tool() -> None:
    policy = DenyList.of({"shell"})
    assert policy(event("echo")) is None


def test_argument_matcher_blocks_only_target_tool() -> None:
    policy = ArgumentMatcher(
        tool_name="shell",
        predicate=lambda args: "rm -rf" in str(args.get("command", "")),
        reason="dangerous shell",
    )
    blocked = policy(event("shell", command="rm -rf /"))
    assert blocked is not None
    assert blocked.block is True
    assert blocked.reason == "dangerous shell"

    # Different tool name → ignored even if args would match.
    assert policy(event("echo", command="rm -rf /")) is None
    # Same tool but args don't match → no decision.
    assert policy(event("shell", command="ls")) is None


def test_argument_matcher_default_reason() -> None:
    policy = ArgumentMatcher(tool_name="x", predicate=lambda _: True)
    decision = policy(event("x"))
    assert decision is not None
    assert "'x'" in (decision.reason or "")


async def test_attach_helper_wires_policies_in_order() -> None:
    runner = HookRunner()
    attach_pre_tool_policies(
        runner,
        AllowList.of({"echo", "shell"}),
        ArgumentMatcher(
            tool_name="shell",
            predicate=lambda args: "rm -rf" in str(args.get("command", "")),
        ),
    )

    # AllowList passes "echo", ArgumentMatcher ignores it → no decisions.
    assert await runner.emit(event("echo")) == []

    # AllowList rejects "delete" first → only one decision (the block).
    decisions = await runner.emit(event("delete"))
    assert len(decisions) == 1
    assert decisions[0].block is True
    assert "delete" in (decisions[0].reason or "")

    # AllowList passes "shell", ArgumentMatcher blocks it → one decision.
    decisions = await runner.emit(event("shell", command="rm -rf /"))
    assert len(decisions) == 1
    assert decisions[0].block is True

    # AllowList passes "shell", ArgumentMatcher ignores benign args → no decisions.
    assert await runner.emit(event("shell", command="ls")) == []
