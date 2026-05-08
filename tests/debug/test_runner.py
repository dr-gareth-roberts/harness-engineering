from __future__ import annotations

import contextlib

import pytest
from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.debug.context import DebugContext
from harness.debug.runner import DebugAborted, DebugRunner
from harness.hooks import HookRunner
from harness.prompts.messages import ContentBlock, Message, text
from harness.runner import CannedRunner
from harness.tools import Dispatcher
from harness.tools.schema import Tool, ToolCall

# ---------- helpers


def _agent(name: str = "bot") -> SubAgent:
    return SubAgent(name=name, system_prompt="x", model="test-model")


class _NoArgs(BaseModel):
    pass


def _fake_tool(name: str = "noop") -> Tool:
    def handler(_: _NoArgs) -> str:
        return "ok"

    return Tool(name=name, description="", input_model=_NoArgs, handler=handler)


# ---------- spec test #4: breakpoint predicate only triggers when condition holds


async def test_breakpoint_does_not_fire_when_predicate_false() -> None:
    inner = CannedRunner(["unchanged"])
    seen: list[DebugContext] = []

    def cb(ctx: DebugContext) -> None:
        seen.append(ctx)
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: c.turn_index == 999,
        breakpoint_callback=cb,
    )
    out = await runner(_agent(), [text("user", "hi")])

    assert out.content[0].text == "unchanged"
    assert seen == []


async def test_breakpoint_fires_when_predicate_true() -> None:
    inner = CannedRunner(["from-runner"])
    seen: list[int] = []

    def cb(ctx: DebugContext) -> None:
        seen.append(ctx.turn_index)
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: c.turn_index == 0,
        breakpoint_callback=cb,
    )
    out = await runner(_agent(), [text("user", "hi")])

    assert seen == [0]
    assert out.content[0].text == "from-runner"


# ---------- spec test #5: programmatic resume() vs abort()


async def test_resume_continues_to_inner_runner() -> None:
    inner = CannedRunner(["onward"])

    def cb(ctx: DebugContext) -> None:
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: True,
        breakpoint_callback=cb,
    )
    out = await runner(_agent(), [text("user", "hi")])
    assert out.content[0].text == "onward"


async def test_abort_raises_debug_aborted() -> None:
    inner = CannedRunner(["never-reached"])

    def cb(ctx: DebugContext) -> None:
        ctx.abort()

    runner = DebugRunner(
        inner,
        break_on=lambda c: True,
        breakpoint_callback=cb,
    )
    with pytest.raises(DebugAborted):
        await runner(_agent(), [text("user", "hi")])


# ---------- spec test #6: DebugRunner satisfies the Runner protocol end-to-end


async def test_debug_runner_works_inside_orchestrator() -> None:
    """Orchestrator only knows about the `Runner` protocol; DebugRunner must
    plug in transparently."""
    inner = CannedRunner(["wrapped"])
    # No breakpoint configured at all — pure pass-through.
    runner = DebugRunner(inner)
    orch = Orchestrator(Dispatcher(), HookRunner(), runner)

    out = await orch.run(_agent(), [text("user", "hello")])
    assert out.content[0].text == "wrapped"


async def test_debug_runner_with_orchestrator_handles_breakpoint_and_resumes() -> None:
    inner = CannedRunner(["resumed-result"])
    hits: list[int] = []

    def cb(ctx: DebugContext) -> None:
        hits.append(ctx.turn_index)
        ctx.resume()

    runner = DebugRunner(inner, break_on=lambda c: True, breakpoint_callback=cb)
    orch = Orchestrator(Dispatcher(), HookRunner(), runner)

    out = await orch.run(_agent(), [text("user", "hi")])
    assert out.content[0].text == "resumed-result"
    assert hits == [0]


# ---------- spec test #8: programmatic mode end-to-end without TTY


async def test_programmatic_mode_runs_end_to_end_without_tty() -> None:
    inner = CannedRunner(["a", "b", "c"])
    bp_hits: list[int] = []

    def cb(ctx: DebugContext) -> None:
        bp_hits.append(ctx.turn_index)
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: c.turn_index in (0, 2),
        breakpoint_callback=cb,
    )

    history: list[Message] = []
    for prompt in ("p1", "p2", "p3"):
        history.append(text("user", prompt))
        reply = await runner(_agent(), history)
        history.append(reply)

    # Two breakpoints fired (turn_index 0 and 2), the middle call passed through.
    assert bp_hits == [0, 2]
    assert [m.content[-1].text for m in history if m.role == "assistant"] == ["a", "b", "c"]


# ---------- spec test #9: mutation persists into the returned Message


async def test_mutation_persists_into_returned_message() -> None:
    """When ctx.mutate(replacement) is called, DebugRunner must return that
    replacement instead of delegating to the inner runner."""
    inner = CannedRunner(["original"])
    swap = text("assistant", "swapped")

    def cb(ctx: DebugContext) -> None:
        ctx.mutate(swap)
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: True,
        breakpoint_callback=cb,
    )
    out = await runner(_agent(), [text("user", "hi")])

    assert out is swap
    assert out.content[0].text == "swapped"


async def test_no_mutation_falls_through_to_inner_runner() -> None:
    inner = CannedRunner(["from-inner"])

    def cb(ctx: DebugContext) -> None:
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: True,
        breakpoint_callback=cb,
    )
    out = await runner(_agent(), [text("user", "hi")])
    assert out.content[0].text == "from-inner"


# ---------- spec test #10: multiple breakpoints in sequence


async def test_multiple_breakpoints_in_sequence() -> None:
    inner = CannedRunner(["r0", "r1", "r2", "r3"])
    seen: list[int] = []

    def cb(ctx: DebugContext) -> None:
        seen.append(ctx.turn_index)
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: True,  # break every turn
        breakpoint_callback=cb,
    )

    history: list[Message] = []
    for i in range(4):
        history.append(text("user", f"p{i}"))
        history.append(await runner(_agent(), history))

    assert seen == [0, 1, 2, 3]


async def test_breakpoint_predicate_can_inspect_last_call() -> None:
    """The break_on predicate can pick out specific tool calls."""
    call = ToolCall(name="delete", arguments={"path": "/x"}, id="tu_1")
    msgs = [
        text("user", "go"),
        Message(role="assistant", content=[ContentBlock(type="tool_use", tool_use=call)]),
    ]
    inner = CannedRunner(["after-delete"])
    fired: list[str] = []

    def cb(ctx: DebugContext) -> None:
        assert ctx.last_call is not None
        fired.append(ctx.last_call.name)
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: c.last_call is not None and c.last_call.name == "delete",
        breakpoint_callback=cb,
    )
    out = await runner(_agent(), msgs)

    assert fired == ["delete"]
    assert out.content[0].text == "after-delete"


# ---------- config validation


def test_break_on_without_handler_raises_at_construction() -> None:
    inner = CannedRunner([])
    with pytest.raises(ValueError, match="no way to handle"):
        DebugRunner(inner, break_on=lambda c: True)


def test_callback_and_interactive_both_set_raises() -> None:
    inner = CannedRunner([])
    with pytest.raises(ValueError, match="exactly one"):
        DebugRunner(
            inner,
            break_on=lambda c: True,
            breakpoint_callback=lambda c: None,
            interactive=True,
        )


# ---------- ad-hoc fire integration


async def test_fire_in_callback_dispatches_through_provided_dispatcher() -> None:
    dispatcher = Dispatcher([_fake_tool()])
    inner = CannedRunner(["after-fire"])
    captured: list[str] = []

    async def cb(ctx: DebugContext) -> None:
        result = await ctx.fire("noop", {})
        captured.append(str(result.content))
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: True,
        breakpoint_callback=cb,
        dispatcher=dispatcher,
    )
    out = await runner(_agent(), [text("user", "hi")])

    assert captured == ["ok"]
    assert out.content[0].text == "after-fire"


# ---------- async callback support


async def test_async_callback_is_awaited() -> None:
    inner = CannedRunner(["done"])
    side_effect: list[str] = []

    async def cb(ctx: DebugContext) -> None:
        side_effect.append("ran")
        ctx.resume()

    runner = DebugRunner(
        inner,
        break_on=lambda c: True,
        breakpoint_callback=cb,
    )
    out = await runner(_agent(), [text("user", "hi")])
    assert side_effect == ["ran"]
    assert out.content[0].text == "done"


# ---------- abort propagation through Orchestrator


async def test_abort_propagates_through_orchestrator() -> None:
    inner = CannedRunner(["never"])

    def cb(ctx: DebugContext) -> None:
        ctx.abort()

    runner = DebugRunner(inner, break_on=lambda c: True, breakpoint_callback=cb)
    orch = Orchestrator(Dispatcher(), HookRunner(), runner)

    with contextlib.suppress(Exception):
        await orch.run(_agent(), [text("user", "hi")])

    # Confirm it was a DebugAborted, not something else.
    with pytest.raises(DebugAborted):
        await orch.run(_agent(), [text("user", "hi")])
