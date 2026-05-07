from __future__ import annotations

from harness.hooks import (
    HookDecision,
    HookRunner,
    PreToolUse,
    PromptSubmit,
    SessionStart,
)
from harness.tools import ToolCall


async def test_handlers_run_in_registration_order() -> None:
    runner = HookRunner()
    seen: list[str] = []

    def first(event: SessionStart) -> None:
        seen.append("first")

    def second(event: SessionStart) -> None:
        seen.append("second")

    runner.register(SessionStart, first)
    runner.register(SessionStart, second)
    await runner.emit(SessionStart())
    assert seen == ["first", "second"]


async def test_block_short_circuits_subsequent_handlers() -> None:
    runner = HookRunner()
    seen: list[str] = []

    def allow(event: PromptSubmit) -> HookDecision:
        seen.append("allow")
        return HookDecision(block=False)

    def deny(event: PromptSubmit) -> HookDecision:
        seen.append("deny")
        return HookDecision(block=True, reason="nope")

    def never(event: PromptSubmit) -> HookDecision:
        seen.append("never")
        return HookDecision(block=False)

    runner.register(PromptSubmit, allow)
    runner.register(PromptSubmit, deny)
    runner.register(PromptSubmit, never)
    decisions = await runner.emit(PromptSubmit(prompt="hi"))

    assert seen == ["allow", "deny"]
    assert len(decisions) == 2
    assert decisions[-1].block is True
    assert decisions[-1].reason == "nope"


async def test_async_handler_awaited() -> None:
    runner = HookRunner()
    seen: list[str] = []

    async def handler(event: SessionStart) -> HookDecision:
        seen.append("ran")
        return HookDecision(block=False, reason="async")

    runner.register(SessionStart, handler)
    decisions = await runner.emit(SessionStart())
    assert seen == ["ran"]
    assert decisions[0].reason == "async"


async def test_only_matching_event_type_dispatched() -> None:
    runner = HookRunner()
    seen: list[str] = []

    def on_session(event: SessionStart) -> None:
        seen.append("session")

    def on_tool(event: PreToolUse) -> None:
        seen.append("tool")

    runner.register(SessionStart, on_session)
    runner.register(PreToolUse, on_tool)
    await runner.emit(SessionStart())
    assert seen == ["session"]

    await runner.emit(PreToolUse(call=ToolCall(name="x", arguments={})))
    assert seen == ["session", "tool"]


async def test_handler_returning_none_contributes_no_decision() -> None:
    runner = HookRunner()

    def silent(event: SessionStart) -> None:
        return None

    runner.register(SessionStart, silent)
    decisions = await runner.emit(SessionStart())
    assert decisions == []
