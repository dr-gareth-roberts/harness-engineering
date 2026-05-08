"""End-to-end tests for `harness.debug.dap.DapAdapter`.

The tests build an in-memory pipe pair so the adapter and the simulated
editor share a real `asyncio.StreamReader/StreamWriter` surface — same
code path as the stdio transport, no special-cased test seam.

The load-bearing test is `test_inspect_requests_pump_during_breakpoint_hold`:
it pins the concurrency property that the DAP read-loop and the
orchestrator session share the event loop. A regression there would
deadlock the editor on every inspect request.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.debug.dap import DapAdapter
from harness.debug.dap_protocol import read_message, write_message
from harness.debug.runner import DebugRunner
from harness.hooks import HookRunner
from harness.prompts import text
from harness.runner import CannedRunner
from harness.tools import Dispatcher, Tool

# ---------------------------------------------------------------------------
# In-memory pipe helpers


class _PipeWriter:
    """Tiny `StreamWriter` shim — feeds writes into a paired StreamReader.

    The real `asyncio.StreamWriter` carries a transport; tests don't need
    one. We mirror the surface `read_message` + `write_message` actually
    consume (`write` + `drain`).
    """

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader

    def write(self, data: bytes) -> None:
        self._reader.feed_data(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self._reader.feed_eof()


def make_pipe() -> tuple[asyncio.StreamReader, _PipeWriter]:
    """Build a (reader, writer) pair where writes are immediately readable."""
    reader = asyncio.StreamReader()
    writer = _PipeWriter(reader)
    return reader, writer


# ---------------------------------------------------------------------------
# Adapter scaffolding


class _NoArgs(BaseModel):
    pass


def _build_dispatcher() -> Dispatcher:
    async def noop(_args: _NoArgs) -> str:
        return "ok"

    return Dispatcher(
        [Tool(name="noop", description="", input_model=_NoArgs, handler=noop, idempotent=True)]
    )


def _agent() -> SubAgent:
    return SubAgent(
        name="t",
        system_prompt="",
        model="demo",
        allowed_tools=["noop"],
    )


def _build_adapter(
    *,
    canned_replies: list[str] | None = None,
    break_at_turns: list[int] | None = None,
    user_messages: int = 1,
    synthesize_lines: list[str] | None = None,
) -> tuple[DapAdapter, Dispatcher]:
    """Wire a DapAdapter to an in-memory CannedRunner-driven session.

    The session is `user_messages` user turns, each producing one
    canned assistant reply via `CannedRunner`. Breakpoints fire at
    each `turn_index` in `break_at_turns` (default: just turn 0).
    """
    replies = canned_replies if canned_replies is not None else ["reply"]
    breaks = break_at_turns if break_at_turns is not None else [0]
    lines = (
        synthesize_lines
        if synthesize_lines is not None
        else [f"turn {i}: assistant reply" for i in range(len(replies))]
    )

    dispatcher = _build_dispatcher()
    adapter = DapAdapter()
    # Pre-load the breakpoint set so we don't depend on setBreakpoints
    # for tests that aren't about that command. Real editors will
    # always send setBreakpoints, but unit tests need flexibility.
    adapter._breakpoint_turns = set(breaks)
    adapter.synthesize_source = lambda: list(lines)

    inner = CannedRunner(replies=replies)
    debug = DebugRunner(
        inner,
        break_on=adapter.break_on_predicate,
        breakpoint_callback=adapter.breakpoint_callback,
        dispatcher=dispatcher,
    )
    orchestrator = Orchestrator(dispatcher, HookRunner(), debug)

    async def _run() -> None:
        history: list[Any] = []
        for i in range(user_messages):
            history.append(text("user", f"hi {i}"))
            reply = await orchestrator.run(_agent(), history)
            history.append(reply)

    adapter.run_session = _run
    return adapter, dispatcher


async def _serve(
    adapter: DapAdapter,
) -> tuple[
    asyncio.Task[None],
    Callable[[dict[str, Any]], Awaitable[None]],
    Callable[[], Awaitable[dict[str, Any]]],
]:
    """Start `adapter.serve` on in-memory pipes; return the task plus
    `send` and `recv` helpers for driving the editor side."""
    in_reader, in_writer = make_pipe()
    out_reader, out_writer = make_pipe()

    serve_task = asyncio.create_task(adapter.serve(in_reader, out_writer))  # type: ignore[arg-type]

    async def send(msg: dict[str, Any]) -> None:
        await write_message(in_writer, msg)  # type: ignore[arg-type]

    async def recv() -> dict[str, Any]:
        return await read_message(out_reader)

    return serve_task, send, recv


async def _wait_event(
    recv: Callable[[], Awaitable[dict[str, Any]]],
    name: str,
    *,
    timeout: float = 1.0,
    drained: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Drain messages until an event with `event == name` arrives.

    Buffers any non-matching messages into `drained` (if provided) so the
    test can still inspect them.
    """
    deadline_left = timeout

    async def _next() -> dict[str, Any]:
        return await asyncio.wait_for(recv(), timeout=deadline_left)

    while True:
        msg = await _next()
        if drained is not None:
            drained.append(msg)
        if msg.get("type") == "event" and msg.get("event") == name:
            return msg


# ---------------------------------------------------------------------------
# Initialize


async def test_initialize_returns_capabilities_and_emits_initialized() -> None:
    adapter, _ = _build_adapter()
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})

        response = await recv()
        assert response["type"] == "response"
        assert response["command"] == "initialize"
        assert response["success"] is True
        assert response["request_seq"] == 1
        # Capabilities surface — our subset
        body = response["body"]
        assert body["supportsConfigurationDoneRequest"] is True
        assert body["supportsTerminateRequest"] is True

        evt = await recv()
        assert evt["type"] == "event"
        assert evt["event"] == "initialized"

        # Adapter assigns sequential `seq` values to every outbound message.
        assert evt["seq"] == response["seq"] + 1
    finally:
        serve_task.cancel()


# ---------------------------------------------------------------------------
# setBreakpoints validation


async def test_set_breakpoints_validates_against_synthesized_source_length() -> None:
    """Lines outside `1..len(synthesize_source())` come back unverified
    with a helpful message; in-range lines come back verified."""
    adapter, _ = _build_adapter(
        canned_replies=["a", "b", "c"],
        break_at_turns=[],
        synthesize_lines=["turn 0", "turn 1", "turn 2"],
    )
    serve_task, send, recv = await _serve(adapter)
    try:
        await send(
            {
                "seq": 1,
                "type": "request",
                "command": "setBreakpoints",
                "arguments": {"breakpoints": [{"line": 1}, {"line": 99}, {"line": 3}]},
            }
        )
        response = await recv()
        assert response["success"] is True
        bps = response["body"]["breakpoints"]
        assert [b["verified"] for b in bps] == [True, False, True]
        assert "out of trajectory range" in bps[1]["message"]
        # Adapter recorded the in-range turns (DAP line N → turn_index N - 1).
        assert adapter._breakpoint_turns == {0, 2}
    finally:
        serve_task.cancel()


# ---------------------------------------------------------------------------
# Full launch → break → continue → terminated flow


async def test_launch_break_continue_terminated_flow() -> None:
    """The end-to-end happy path: editor initializes, sets a breakpoint,
    launches, sees `stopped`, continues, sees `terminated`."""
    adapter, _ = _build_adapter(canned_replies=["reply 0"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        # initialize
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()  # response
        await recv()  # initialized event

        # configurationDone (no setBreakpoints — pre-loaded)
        await send({"seq": 2, "type": "request", "command": "configurationDone"})
        cd_resp = await recv()
        assert cd_resp["command"] == "configurationDone"

        # launch — kicks off the orchestrator
        await send({"seq": 3, "type": "request", "command": "launch"})
        launch_resp = await recv()
        assert launch_resp["command"] == "launch"
        assert launch_resp["success"] is True

        # Wait for `stopped` — the breakpoint fired at turn 0.
        stopped = await _wait_event(recv, "stopped", timeout=1.0)
        assert stopped["body"]["reason"] == "breakpoint"
        assert stopped["body"]["threadId"] == DapAdapter.THREAD_ID

        # continue — releases the breakpoint
        await send({"seq": 4, "type": "request", "command": "continue"})
        cont_resp = await recv()
        assert cont_resp["command"] == "continue"
        # `continued` event accompanies the response.
        cont_evt = await _wait_event(recv, "continued", timeout=1.0)
        assert cont_evt["body"]["threadId"] == DapAdapter.THREAD_ID

        # Session finishes; adapter emits terminated + exited.
        terminated = await _wait_event(recv, "terminated", timeout=1.0)
        assert terminated["event"] == "terminated"
        exited = await _wait_event(recv, "exited", timeout=1.0)
        assert exited["body"]["exitCode"] == 0
    finally:
        serve_task.cancel()


# ---------------------------------------------------------------------------
# Concurrency — the load-bearing test


async def test_inspect_requests_pump_during_breakpoint_hold() -> None:
    """Pin the concurrency invariant: while the breakpoint callback is
    parked on `_continue_event`, the DAP read-loop must keep pulling
    requests off the wire and answering them from the held context.

    A regression where the message loop blocks on the breakpoint
    callback would make these requests hang — this test would time out.
    """
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()
        await recv()
        await send({"seq": 2, "type": "request", "command": "launch"})
        await recv()

        await _wait_event(recv, "stopped", timeout=1.0)

        # While stopped, fire several inspect requests interleaved.
        # Each MUST get a response before we send `continue`.
        await send({"seq": 10, "type": "request", "command": "threads"})
        threads_resp = await asyncio.wait_for(recv(), timeout=0.5)
        assert threads_resp["body"]["threads"][0]["name"] == "trajectory"

        await send({"seq": 11, "type": "request", "command": "stackTrace"})
        stack_resp = await asyncio.wait_for(recv(), timeout=0.5)
        frames = stack_resp["body"]["stackFrames"]
        assert len(frames) == 1
        assert frames[0]["line"] == 1  # turn 0 + 1

        await send({"seq": 12, "type": "request", "command": "scopes"})
        scopes_resp = await asyncio.wait_for(recv(), timeout=0.5)
        assert scopes_resp["body"]["scopes"][0]["name"] == "context"

        await send(
            {
                "seq": 13,
                "type": "request",
                "command": "variables",
                "arguments": {"variablesReference": DapAdapter.SCOPE_REFERENCE},
            }
        )
        vars_resp = await asyncio.wait_for(recv(), timeout=0.5)
        names = [v["name"] for v in vars_resp["body"]["variables"]]
        assert "turn_index" in names
        assert "message_count" in names

        # Now release the breakpoint.
        await send({"seq": 99, "type": "request", "command": "continue"})
        await recv()  # response
        await _wait_event(recv, "continued", timeout=1.0)
        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


# ---------------------------------------------------------------------------
# evaluate (limited)


async def test_evaluate_supported_name_returns_value() -> None:
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()
        await recv()
        await send({"seq": 2, "type": "request", "command": "launch"})
        await recv()
        await _wait_event(recv, "stopped", timeout=1.0)

        await send(
            {
                "seq": 10,
                "type": "request",
                "command": "evaluate",
                "arguments": {"expression": "turn_index"},
            }
        )
        resp = await asyncio.wait_for(recv(), timeout=0.5)
        assert resp["success"] is True
        assert resp["body"]["result"] == "0"
        assert resp["body"]["type"] == "int"

        await send({"seq": 99, "type": "request", "command": "continue"})
        await recv()
        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


async def test_evaluate_unsupported_expression_returns_error() -> None:
    """Arbitrary expressions are out of scope — the adapter responds with
    an error message that lists what *is* supported."""
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()
        await recv()
        await send({"seq": 2, "type": "request", "command": "launch"})
        await recv()
        await _wait_event(recv, "stopped", timeout=1.0)

        await send(
            {
                "seq": 10,
                "type": "request",
                "command": "evaluate",
                "arguments": {"expression": "1 + 1"},
            }
        )
        resp = await asyncio.wait_for(recv(), timeout=0.5)
        assert resp["success"] is False
        assert "unsupported expression" in resp["message"]
        assert "turn_index" in resp["message"]  # listed as supported

        await send({"seq": 99, "type": "request", "command": "continue"})
        await recv()
        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


# ---------------------------------------------------------------------------
# Source request


async def test_source_request_returns_synthesized_trajectory() -> None:
    adapter, _ = _build_adapter(
        canned_replies=["x", "y"],
        break_at_turns=[],
        synthesize_lines=["line one", "line two"],
    )
    serve_task, send, recv = await _serve(adapter)
    try:
        await send(
            {
                "seq": 1,
                "type": "request",
                "command": "source",
                "arguments": {"sourceReference": DapAdapter.SOURCE_REFERENCE},
            }
        )
        resp = await recv()
        assert resp["success"] is True
        body = resp["body"]
        assert body["mimeType"] == "text/plain"
        assert "line one" in body["content"]
        assert "line two" in body["content"]
    finally:
        serve_task.cancel()


async def test_source_request_with_unknown_reference_fails() -> None:
    adapter, _ = _build_adapter()
    serve_task, send, recv = await _serve(adapter)
    try:
        await send(
            {
                "seq": 1,
                "type": "request",
                "command": "source",
                "arguments": {"sourceReference": 9999},
            }
        )
        resp = await recv()
        assert resp["success"] is False
        assert "unknown sourceReference" in resp["message"]
    finally:
        serve_task.cancel()


# ---------------------------------------------------------------------------
# Disconnect mid-breakpoint


async def test_disconnect_during_breakpoint_aborts_session() -> None:
    """Editor disconnects while we're parked on a breakpoint. The adapter
    must abort the DebugContext and let the session task wind down so the
    process doesn't leak.
    """
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()
        await recv()
        await send({"seq": 2, "type": "request", "command": "launch"})
        await recv()
        await _wait_event(recv, "stopped", timeout=1.0)

        await send({"seq": 99, "type": "request", "command": "disconnect"})
        # Adapter responds, session task sees the abort, terminated event fires.
        # Drain until we see terminated.
        seen: list[dict[str, Any]] = []
        await _wait_event(recv, "terminated", timeout=1.0, drained=seen)
        # The serve loop exits after disconnect.
        await asyncio.wait_for(serve_task, timeout=1.0)
    finally:
        if not serve_task.done():
            serve_task.cancel()


# ---------------------------------------------------------------------------
# Error paths


async def test_unknown_command_returns_error_response() -> None:
    adapter, _ = _build_adapter()
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "no_such_command"})
        resp = await recv()
        assert resp["success"] is False
        assert "unknown command" in resp["message"]
        assert resp["request_seq"] == 1
    finally:
        serve_task.cancel()


async def test_launch_without_run_session_fails() -> None:
    adapter = DapAdapter()  # no run_session set
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "launch"})
        resp = await recv()
        assert resp["success"] is False
        assert "no run_session" in resp["message"]
    finally:
        serve_task.cancel()


# ---------------------------------------------------------------------------
# Aborted session surfaces cleanly


async def test_session_aborted_via_disconnect_does_not_propagate_to_serve() -> None:
    """`DebugAborted` raised by the orchestrator (because we aborted via
    disconnect) must not crash the serve loop — it's an expected outcome
    of an editor-driven kill."""
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()
        await recv()
        await send({"seq": 2, "type": "request", "command": "launch"})
        await recv()
        await _wait_event(recv, "stopped", timeout=1.0)

        await send({"seq": 99, "type": "request", "command": "disconnect"})
        await asyncio.wait_for(serve_task, timeout=1.0)
        # serve_task completed without raising
        assert serve_task.exception() is None
    finally:
        if not serve_task.done():
            serve_task.cancel()


# ---------------------------------------------------------------------------
# Sanity: types check at import time


def test_module_imports_clean() -> None:
    """Type-only smoke: imports succeed without forward-reference errors."""
    from harness.debug.dap import DapAdapter as _DA  # noqa: F401
    from harness.debug.dap_messages import (  # noqa: F401
        Breakpoint,
        Capabilities,
        Scope,
        Source,
        StackFrame,
        Variable,
    )
