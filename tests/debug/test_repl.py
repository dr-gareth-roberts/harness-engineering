from __future__ import annotations

import io

import pytest
from pydantic import BaseModel

from harness.debug.context import DebugContext
from harness.debug.repl import DebugRepl
from harness.debug.runner import DebugRunner
from harness.prompts.messages import ContentBlock, Message, text
from harness.runner import CannedRunner
from harness.tools.dispatcher import Dispatcher
from harness.tools.schema import Tool, ToolCall


class _NoArgs(BaseModel):
    pass


def _ok_tool(name: str = "ping") -> Tool:
    def handler(_: _NoArgs) -> str:
        return "pong"

    return Tool(name=name, description="", input_model=_NoArgs, handler=handler)


# ---------- basic line dispatch


async def test_repl_resume_exits_loop() -> None:
    ctx = DebugContext([text("user", "hi")])
    stdin = io.StringIO("resume\n")
    stdout = io.StringIO()

    repl = DebugRepl(ctx, stdin=stdin, stdout=stdout)
    await repl.run()

    assert ctx.resumed is True
    assert "resuming" in stdout.getvalue()


async def test_repl_abort_exits_loop_and_marks_aborted() -> None:
    ctx = DebugContext([text("user", "hi")])
    stdin = io.StringIO("abort\n")
    stdout = io.StringIO()
    repl = DebugRepl(ctx, stdin=stdin, stdout=stdout)

    await repl.run()

    assert ctx.aborted is True
    assert "aborting" in stdout.getvalue()


async def test_repl_eof_acts_like_resume() -> None:
    ctx = DebugContext([text("user", "hi")])
    stdin = io.StringIO("")  # immediate EOF
    stdout = io.StringIO()
    repl = DebugRepl(ctx, stdin=stdin, stdout=stdout)

    await repl.run()

    assert ctx.resumed is True
    assert "EOF" in stdout.getvalue()


# ---------- read commands


async def test_messages_command_lists_each_turn() -> None:
    msgs = [text("user", "first"), text("assistant", "second")]
    ctx = DebugContext(msgs)
    stdin = io.StringIO("messages\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()
    out = stdout.getvalue()

    assert "[0] user: first" in out
    assert "[1] assistant: second" in out


async def test_messages_command_renders_tool_use_block() -> None:
    call = ToolCall(name="delete", arguments={"path": "/x"}, id="tu_1")
    msgs = [Message(role="assistant", content=[ContentBlock(type="tool_use", tool_use=call)])]
    ctx = DebugContext(msgs)
    stdin = io.StringIO("messages\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()
    out = stdout.getvalue()

    assert "tool_use(delete" in out
    assert "/x" in out


async def test_last_call_command_when_present() -> None:
    call = ToolCall(name="search", arguments={"q": "x"}, id="tu_2")
    ctx = DebugContext([], last_call=call)
    stdin = io.StringIO("last_call\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    assert "tool_use(search" in stdout.getvalue()


async def test_last_call_command_when_absent() -> None:
    ctx = DebugContext([])
    stdin = io.StringIO("last_call\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    assert "no tool calls" in stdout.getvalue()


async def test_turn_index_command() -> None:
    ctx = DebugContext([], turn_index=7)
    stdin = io.StringIO("turn_index\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    out = stdout.getvalue()
    # The bare value should appear, distinguishable from any '7' embedded in
    # the surrounding chrome.
    assert "> 7\n" in out


# ---------- mutate


async def test_mutate_command_queues_message() -> None:
    ctx = DebugContext([text("user", "hi")])
    stdin = io.StringIO('mutate user "wait, cancel that"\nresume\n')
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    pending = ctx.pending_mutation
    assert pending is not None
    assert pending.role == "user"
    assert pending.content[0].text == "wait, cancel that"


async def test_mutate_invalid_role_is_rejected() -> None:
    ctx = DebugContext([text("user", "hi")])
    stdin = io.StringIO("mutate spy text-here\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    assert ctx.pending_mutation is None
    assert "invalid role" in stdout.getvalue()


# ---------- fire


async def test_fire_command_dispatches_through_dispatcher() -> None:
    dispatcher = Dispatcher([_ok_tool()])
    ctx = DebugContext([], dispatcher=dispatcher)
    stdin = io.StringIO("fire ping {}\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    out = stdout.getvalue()
    assert "is_error=False" in out
    assert "pong" in out


async def test_fire_without_dispatcher_reports_error() -> None:
    ctx = DebugContext([])
    stdin = io.StringIO("fire ping {}\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    assert "fire failed" in stdout.getvalue()


async def test_fire_with_bad_json_reports_error() -> None:
    ctx = DebugContext([], dispatcher=Dispatcher([_ok_tool()]))
    stdin = io.StringIO("fire ping not-json\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    assert "arg parse error" in stdout.getvalue()


# ---------- inspect


async def test_inspect_evaluates_expression_against_ctx() -> None:
    ctx = DebugContext([text("user", "a")], turn_index=99)
    stdin = io.StringIO("inspect 7 + ctx.turn_index\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    # Computed result distinguishes the inspect output from any chrome.
    assert "> 106\n" in stdout.getvalue()


async def test_inspect_error_does_not_crash_repl() -> None:
    ctx = DebugContext([])
    stdin = io.StringIO("inspect 1/0\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()
    assert ctx.resumed is True  # REPL still terminated cleanly
    assert "inspect error" in stdout.getvalue()


# ---------- help / unknown


async def test_help_command_lists_commands() -> None:
    ctx = DebugContext([])
    stdin = io.StringIO("help\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()
    out = stdout.getvalue()

    for verb in ("messages", "mutate", "fire", "inspect", "resume", "abort"):
        assert verb in out


async def test_unknown_command_warns_but_keeps_running() -> None:
    ctx = DebugContext([])
    stdin = io.StringIO("frobnicate\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    out = stdout.getvalue()
    assert "unknown command" in out
    assert ctx.resumed is True


async def test_blank_line_is_skipped() -> None:
    ctx = DebugContext([])
    stdin = io.StringIO("\n\nresume\n")
    stdout = io.StringIO()

    await DebugRepl(ctx, stdin=stdin, stdout=stdout).run()

    assert ctx.resumed is True


# ---------- end-to-end: REPL via DebugRunner with monkeypatched stdin


async def test_debug_runner_drives_interactive_repl_through_streams() -> None:
    """End-to-end: DebugRunner with interactive=True consumes stdin/stdout
    we pass in, and produces a working session where the REPL controls
    resume/abort decisions."""
    inner = CannedRunner(["after-repl"])
    stdin = io.StringIO("messages\nresume\n")
    stdout = io.StringIO()

    runner = DebugRunner(
        inner,
        break_on=lambda c: c.turn_index == 0,
        interactive=True,
        stdin=stdin,
        stdout=stdout,
    )

    from harness.agents import SubAgent

    out = await runner(
        SubAgent(name="x", system_prompt="", model="m"),
        [text("user", "hello")],
    )

    assert out.content[0].text == "after-repl"
    assert "user: hello" in stdout.getvalue()


async def test_debug_runner_repl_abort_raises_debug_aborted() -> None:
    from harness.agents import SubAgent
    from harness.debug.runner import DebugAborted

    inner = CannedRunner(["never"])
    stdin = io.StringIO("abort\n")
    stdout = io.StringIO()

    runner = DebugRunner(
        inner,
        break_on=lambda c: True,
        interactive=True,
        stdin=stdin,
        stdout=stdout,
    )

    with pytest.raises(DebugAborted):
        await runner(
            SubAgent(name="x", system_prompt="", model="m"),
            [text("user", "hi")],
        )
