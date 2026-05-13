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


async def _drain_pending(
    recv: Callable[[], Awaitable[dict[str, Any]]],
    *,
    quiet_for: float = 0.05,
) -> list[dict[str, Any]]:
    """Pull every message currently sitting in the writer's buffer.

    Used by lifecycle tests that need to verify the adapter does **not**
    duplicate `terminated` / `exited` events: read until no message
    arrives within `quiet_for`, then count. The in-memory pipe writes
    are synchronous (`feed_data`) so once the producer has returned,
    the bytes are already readable.
    """
    out: list[dict[str, Any]] = []
    while True:
        try:
            msg = await asyncio.wait_for(recv(), timeout=quiet_for)
        except (TimeoutError, EOFError):
            return out
        out.append(msg)


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
    must abort the DebugContext, let the session task wind down so the
    process doesn't leak, and emit the lifecycle pair `terminated`+`exited`
    **exactly once** — not the legacy `terminated, exited, terminated`
    sequence that confused editor state machines (M1.8).
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
        # Wait for the serve loop to exit; by then every lifecycle event
        # the adapter intends to emit is already in the pipe buffer (the
        # in-memory writes complete synchronously before
        # `_shutdown_session` returns).
        await asyncio.wait_for(serve_task, timeout=1.0)

        # Drain *everything* the writer queued: the pre-fix bug emitted
        # `terminated, exited, terminated`; the post-fix path collapses
        # that to one of each.
        seen = await _drain_pending(recv)
        events = [m for m in seen if m.get("type") == "event"]
        terminated_count = sum(1 for m in events if m.get("event") == "terminated")
        exited_count = sum(1 for m in events if m.get("event") == "exited")
        assert terminated_count == 1, (
            f"expected exactly one 'terminated' event, got {terminated_count}; "
            f"events={[m.get('event') for m in events]}"
        )
        assert exited_count == 1, (
            f"expected exactly one 'exited' event, got {exited_count}; "
            f"events={[m.get('event') for m in events]}"
        )
    finally:
        if not serve_task.done():
            serve_task.cancel()


async def test_terminate_then_disconnect_emits_lifecycle_pair_exactly_once() -> None:
    """Symmetric M1.8 case: `terminate` followed by `disconnect` (the
    DAP-spec-compliant editor sequence) must still produce exactly one
    `terminated` + one `exited` event across the whole exchange. The
    duplicate-emit guard applies uniformly to both lifecycle paths so
    editors don't see a spurious second `terminated` from the post-task
    `_shutdown_session(reason="disconnect")` step.
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

        # `terminate` winds the session down but leaves the serve loop
        # running so the editor can follow up with `disconnect`.
        await send({"seq": 90, "type": "request", "command": "terminate"})
        await recv()  # terminate response

        await send({"seq": 99, "type": "request", "command": "disconnect"})
        await asyncio.wait_for(serve_task, timeout=1.0)

        seen = await _drain_pending(recv)
        events = [m for m in seen if m.get("type") == "event"]
        terminated_count = sum(1 for m in events if m.get("event") == "terminated")
        exited_count = sum(1 for m in events if m.get("event") == "exited")
        assert terminated_count == 1, (
            f"expected exactly one 'terminated' event across terminate+disconnect, "
            f"got {terminated_count}; events={[m.get('event') for m in events]}"
        )
        assert exited_count == 1, (
            f"expected exactly one 'exited' event across terminate+disconnect, "
            f"got {exited_count}; events={[m.get('event') for m in events]}"
        )
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


# ---------------------------------------------------------------------------
# Wave 13b #16: pause request


async def test_pause_sets_flag_so_next_break_on_check_fires() -> None:
    """The DAP `pause` request flips a flag the runner's `break_on`
    consults; the next runner invocation pauses unconditionally.
    Editor's pause button now works."""
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[])
    serve_task, send, recv = await _serve(adapter)
    try:
        # Send a pause request without launching — it just sets a flag.
        await send({"seq": 1, "type": "request", "command": "pause"})
        resp = await recv()
        assert resp["success"] is True

        # Predicate now fires unconditionally on the next consult.
        # Use a trivial DebugContext to drive the predicate directly.
        from harness.debug.context import DebugContext
        from harness.prompts import text as text_msg

        ctx = DebugContext([text_msg("user", "hi")])
        # First check → True (consumes the flag).
        assert adapter.break_on_predicate(ctx) is True
        # Second check → False (flag was consumed; no scheduled bps).
        assert adapter.break_on_predicate(ctx) is False
    finally:
        serve_task.cancel()


# ---------------------------------------------------------------------------
# Wave 13b #15: step semantics


async def test_next_sets_step_over_flag() -> None:
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()
        await recv()
        await send({"seq": 2, "type": "request", "command": "launch"})
        await recv()
        await _wait_event(recv, "stopped", timeout=1.0)

        await send({"seq": 10, "type": "request", "command": "next"})
        await recv()
        # `next` set step_over and resumed; verify the flag is set.
        assert adapter._step_mode == "step_over"

        # Drain remaining events.
        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


async def test_step_over_predicate_fires_then_clears() -> None:
    """After `next` is sent, the next break_on check fires; subsequent
    checks fall back to the configured per-turn breakpoints."""
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[])
    adapter._step_mode = "step_over"

    from harness.debug.context import DebugContext
    from harness.prompts import text as text_msg

    ctx = DebugContext([text_msg("user", "hi")])
    assert adapter.break_on_predicate(ctx) is True
    # Flag consumed.
    assert adapter._step_mode is None
    # Next check is governed by the per-turn breakpoints (empty → False).
    assert adapter.break_on_predicate(ctx) is False


# ---------------------------------------------------------------------------
# Wave 13b #17: evaluate parity opt-in


async def test_evaluate_default_only_returns_supported_names() -> None:
    """Without launch arg `allowEvaluate: true`, evaluate returns the
    same restricted name set as before (Wave 7 behavior)."""
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()
        await recv()
        await send({"seq": 2, "type": "request", "command": "launch"})
        await recv()
        await _wait_event(recv, "stopped", timeout=1.0)

        # Arbitrary expression → error (no allowEvaluate).
        await send(
            {
                "seq": 10,
                "type": "request",
                "command": "evaluate",
                "arguments": {"expression": "1 + 1"},
            }
        )
        resp = await recv()
        assert resp["success"] is False
        assert "unsupported expression" in resp["message"]

        # Continue + drain.
        await send({"seq": 99, "type": "request", "command": "continue"})
        await recv()
        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


async def test_evaluate_with_allow_evaluate_runs_arbitrary_python() -> None:
    """Editor passes `allowEvaluate: true` in launch args; evaluate
    routes through the REPL's `evaluate_in_context` helper."""
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()
        await recv()
        await send(
            {
                "seq": 2,
                "type": "request",
                "command": "launch",
                "arguments": {"allowEvaluate": True},
            }
        )
        await recv()
        await _wait_event(recv, "stopped", timeout=1.0)

        # Arbitrary Python expression — succeeds with allowEvaluate.
        await send(
            {
                "seq": 10,
                "type": "request",
                "command": "evaluate",
                "arguments": {"expression": "1 + 1"},
            }
        )
        resp = await recv()
        assert resp["success"] is True
        assert resp["body"]["result"] == "2"

        # ctx is bound — can introspect.
        await send(
            {
                "seq": 11,
                "type": "request",
                "command": "evaluate",
                "arguments": {"expression": "ctx.turn_index"},
            }
        )
        resp = await recv()
        assert resp["success"] is True
        assert resp["body"]["result"] == "0"

        await send({"seq": 99, "type": "request", "command": "continue"})
        await recv()
        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


async def test_evaluate_with_allow_evaluate_surface_errors_as_failed_response() -> None:
    """A bad expression surfaces as a non-success response with the
    error message, not a crashing adapter."""
    adapter, _ = _build_adapter(canned_replies=["x"], break_at_turns=[0])
    serve_task, send, recv = await _serve(adapter)
    try:
        await send({"seq": 1, "type": "request", "command": "initialize"})
        await recv()
        await recv()
        await send(
            {
                "seq": 2,
                "type": "request",
                "command": "launch",
                "arguments": {"allowEvaluate": True},
            }
        )
        await recv()
        await _wait_event(recv, "stopped", timeout=1.0)

        # Syntax error.
        await send(
            {
                "seq": 10,
                "type": "request",
                "command": "evaluate",
                "arguments": {"expression": "not valid python @@"},
            }
        )
        resp = await recv()
        assert resp["success"] is False
        assert "SyntaxError" in resp["message"]

        await send({"seq": 99, "type": "request", "command": "continue"})
        await recv()
        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


# ---------------------------------------------------------------------------
# M3.6 — frame-aware step semantics
#
# Pre-1.1.0, `next`, `stepIn`, and `stepOut` all aliased to step_over
# (per-turn). The M3.6 fix splits them apart and adds a hook-listener
# path that lets the adapter pause inside a tool frame. The tests
# below pin: (a) stepIn fires on the next PreToolUse, (b) repeated
# stepIns walk through tool dispatches one at a time, (c) stepOut
# from a tool frame returns to the next event, (d) stepOut from the
# orchestrator falls back to step_over, and (e) frame tracking and
# fallback semantics in `break_on_predicate`.


async def _fire_pre_tool_use(adapter: DapAdapter, name: str) -> None:
    """Simulate a `PreToolUse` event landing on the adapter's hook
    listener. Drives the same code path the real runner does when it
    emits the event through the `HookRunner` we attached.
    """
    from harness.hooks.events import PreToolUse
    from harness.tools.schema import ToolCall

    await adapter._on_pre_tool_use(PreToolUse(call=ToolCall(name=name, arguments={})))


async def _fire_post_tool_use(adapter: DapAdapter, name: str) -> None:
    """Simulate a `PostToolUse` event landing on the adapter."""
    from harness.hooks.events import PostToolUse
    from harness.tools.schema import ToolCall, ToolResult

    await adapter._on_post_tool_use(
        PostToolUse(call=ToolCall(name=name, arguments={}), result=ToolResult(content="ok"))
    )


async def test_attach_hooks_registers_pre_and_post_listeners() -> None:
    """`attach_hooks` should register listeners on both `PreToolUse` and
    `PostToolUse` so the adapter sees both ends of each tool dispatch.
    """
    from harness.hooks.events import PostToolUse, PreToolUse

    adapter = DapAdapter()
    hooks = HookRunner()
    adapter.attach_hooks(hooks)

    registered_types = {event_type for event_type, _ in hooks._handlers}
    assert PreToolUse in registered_types
    assert PostToolUse in registered_types


async def test_pre_tool_use_sets_current_frame_to_tool() -> None:
    """Observing a `PreToolUse` event flips the adapter's frame state
    to 'tool' so subsequent stepOut requests know where they are.
    """
    adapter = DapAdapter()
    assert adapter._current_frame is None
    await _fire_pre_tool_use(adapter, "search")
    assert adapter._current_frame == "tool"
    assert adapter._active_tool_call is not None
    assert adapter._active_tool_call.name == "search"


async def test_post_tool_use_returns_frame_to_orchestrator() -> None:
    """`PostToolUse` closes the tool frame and the adapter snaps back
    to 'orchestrator' so a follow-up stepOut request degrades to
    step_over instead of stalling.
    """
    adapter = DapAdapter()
    await _fire_pre_tool_use(adapter, "search")
    assert adapter._current_frame == "tool"
    await _fire_post_tool_use(adapter, "search")
    # mypy narrows _current_frame to Literal["tool"] from the previous
    # assertion; the PostToolUse fire mutates it back to "orchestrator"
    # at runtime, but mypy can't see across the await boundary.
    assert adapter._current_frame == "orchestrator"  # type: ignore[comparison-overlap]
    assert adapter._active_tool_call is None


async def _capture_writes(adapter: DapAdapter) -> list[dict[str, Any]]:
    """Patch the adapter's `_write` so outbound envelopes accumulate
    in the returned list instead of attempting to hit a real
    transport. Used by every M3.6 unit test that fires a synthesized
    `_on_breakpoint` without a connected serve loop.
    """
    sent: list[dict[str, Any]] = []

    async def _capture(envelope: dict[str, Any]) -> None:
        sent.append(envelope)

    adapter._write = _capture  # type: ignore[method-assign]
    return sent


async def _fire_with_auto_resume(coro: Awaitable[None], adapter: DapAdapter) -> None:
    """Launch `coro` (typically a hook fire that ends in
    `_on_breakpoint`'s parked `await self._continue_event.wait()`),
    then on the next event-loop tick set the continue event so the
    parked coroutine returns. Mirrors the editor's
    `continue`-after-stopped flow without needing a serve loop.
    """
    task: asyncio.Task[None] = asyncio.create_task(coro)  # type: ignore[arg-type]
    # Yield control so the parked coroutine reaches the wait().
    await asyncio.sleep(0)
    adapter._continue_event.set()
    await task


async def test_step_in_pauses_at_next_pre_tool_use() -> None:
    """A `stepIn` request followed by a `PreToolUse` event synthesizes a
    breakpoint via the adapter's `_on_breakpoint` parking path — same
    code path the turn-boundary breakpoints use. The test releases
    the park on the next tick so the adapter doesn't hang.
    """
    adapter = DapAdapter()
    adapter._step_mode = "step_in"
    sent = await _capture_writes(adapter)

    await _fire_with_auto_resume(_fire_pre_tool_use(adapter, "search"), adapter)

    # The step flag cleared on consumption.
    assert adapter._step_mode is None
    # The breakpoint synthesized a `stopped` event.
    stopped = [m for m in sent if m.get("type") == "event" and m.get("event") == "stopped"]
    assert len(stopped) == 1
    assert stopped[0]["body"]["reason"] == "breakpoint"


async def test_step_in_twice_walks_through_two_tool_dispatches() -> None:
    """Two consecutive `stepIn` requests pause on two consecutive
    `PreToolUse` events. Pins the "walk through tool dispatches one
    at a time" behavior the task description calls out.
    """
    adapter = DapAdapter()
    sent = await _capture_writes(adapter)

    # First step_in → first PreToolUse fires the breakpoint.
    adapter._step_mode = "step_in"
    await _fire_with_auto_resume(_fire_pre_tool_use(adapter, "search"), adapter)
    assert adapter._step_mode is None
    # First PostToolUse completes — frame snaps to orchestrator.
    await _fire_post_tool_use(adapter, "search")

    # Second step_in → second PreToolUse fires the breakpoint.
    adapter._step_mode = "step_in"
    await _fire_with_auto_resume(_fire_pre_tool_use(adapter, "fetch"), adapter)
    assert adapter._step_mode is None

    # Two `stopped` events synthesized — one per dispatch.
    stopped = [m for m in sent if m.get("type") == "event" and m.get("event") == "stopped"]
    assert len(stopped) == 2


async def test_step_in_predicate_fallback_when_no_further_pre_tool_use() -> None:
    """`step_in` with no follow-up `PreToolUse` falls back to firing at
    the next turn boundary so the editor's button never silently
    no-ops. Documented as the explicit fallback in the module
    docstring.
    """
    from harness.debug.context import DebugContext
    from harness.prompts import text as text_msg

    adapter = DapAdapter()
    adapter._step_mode = "step_in"

    ctx = DebugContext([text_msg("user", "hi")])
    # No PreToolUse fires; the predicate is consulted at the next
    # turn boundary and fires anyway.
    assert adapter.break_on_predicate(ctx) is True
    assert adapter._step_mode is None


async def test_step_out_from_tool_frame_arms_next_event_break() -> None:
    """Stepping out from a tool frame fires step_out, which the
    PostToolUse listener consumes by arming `_break_on_next_event`.
    The very next event (next PreToolUse or next turn boundary)
    becomes the pause point.
    """
    adapter = DapAdapter()
    # Simulate entering a tool frame.
    await _fire_pre_tool_use(adapter, "search")
    assert adapter._current_frame == "tool"

    # Editor sends stepOut while paused in the tool frame.
    adapter._step_mode = "step_out"

    # PostToolUse fires (dispatch completes). The listener should arm
    # the next-event trap and clear the step flag.
    await _fire_post_tool_use(adapter, "search")
    assert adapter._step_mode is None
    assert adapter._break_on_next_event is True


async def test_step_out_aftermath_pauses_at_next_pre_tool_use_if_one_arrives() -> None:
    """After step_out arms `_break_on_next_event`, the very next
    `PreToolUse` becomes a breakpoint — the "walk to the next event
    in the orchestrator frame" semantic.
    """
    adapter = DapAdapter()
    sent = await _capture_writes(adapter)

    # Arm the trap directly (mimics post-step_out state).
    adapter._break_on_next_event = True
    await _fire_with_auto_resume(_fire_pre_tool_use(adapter, "fetch"), adapter)
    assert adapter._break_on_next_event is False

    stopped = [m for m in sent if m.get("type") == "event" and m.get("event") == "stopped"]
    assert len(stopped) == 1


async def test_step_out_aftermath_pauses_at_next_turn_boundary_if_no_pre_tool_use() -> None:
    """If no follow-up `PreToolUse` arrives before the next turn
    boundary, `break_on_predicate` consumes `_break_on_next_event` and
    pauses at the turn boundary instead. Documented fallback.
    """
    from harness.debug.context import DebugContext
    from harness.prompts import text as text_msg

    adapter = DapAdapter()
    adapter._break_on_next_event = True
    ctx = DebugContext([text_msg("user", "hi")])
    assert adapter.break_on_predicate(ctx) is True
    assert adapter._break_on_next_event is False


async def test_step_out_from_orchestrator_frame_falls_back_to_step_over() -> None:
    """`stepOut` from an orchestrator frame has no outer frame to
    return to; the adapter promotes it to step_over so the editor's
    button isn't ignored. Pin this by driving the request handler
    directly with a synthesized DAP envelope.
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

        # The default frame at pause is None (no tool event yet); the
        # stepOut handler should promote to step_over rather than
        # step_out.
        assert adapter._current_frame is None
        await send({"seq": 10, "type": "request", "command": "stepOut"})
        await recv()
        assert adapter._step_mode == "step_over"

        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


async def test_step_in_request_sets_step_in_flag() -> None:
    """The `stepIn` DAP request now sets `_step_mode = "step_in"`
    (pre-1.1.0 this was aliased to `"step_over"`). Pin the new
    semantic at the request-handler level.
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

        await send({"seq": 10, "type": "request", "command": "stepIn"})
        await recv()
        assert adapter._step_mode == "step_in"

        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


async def test_step_out_request_from_tool_frame_sets_step_out_flag() -> None:
    """When the adapter is in a tool frame at the time `stepOut`
    arrives, the request handler sets `_step_mode = "step_out"`
    (the true frame-aware semantic, not the orchestrator-only
    fallback).
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

        # Simulate being inside a tool frame.
        adapter._current_frame = "tool"

        await send({"seq": 10, "type": "request", "command": "stepOut"})
        await recv()
        assert adapter._step_mode == "step_out"

        await _wait_event(recv, "terminated", timeout=1.0)
    finally:
        serve_task.cancel()


async def test_next_runs_to_next_turn_boundary_regardless_of_tool_events() -> None:
    """`next` (step_over) ignores tool events entirely. Even after a
    `PreToolUse` fires, the predicate at the next turn boundary
    pauses — there is no early stop inside the tool-use loop.
    """
    from harness.debug.context import DebugContext
    from harness.prompts import text as text_msg

    adapter = DapAdapter()
    adapter._step_mode = "step_over"

    # Tool events fire mid-loop; the step_in/step_out hooks DO NOT
    # consume a step_over flag (it lives on break_on_predicate).
    await _fire_pre_tool_use(adapter, "search")
    assert adapter._step_mode == "step_over"
    await _fire_post_tool_use(adapter, "search")
    assert adapter._step_mode == "step_over"

    # Turn boundary check — predicate fires.
    ctx = DebugContext([text_msg("user", "hi")])
    assert adapter.break_on_predicate(ctx) is True
    assert adapter._step_mode is None


async def test_pause_request_honored_at_next_pre_tool_use() -> None:
    """A `pause` request set while the runner is between tool events
    is honored at the next `PreToolUse` (mid-turn responsiveness),
    not deferred to the next turn boundary.
    """
    adapter = DapAdapter()
    sent = await _capture_writes(adapter)

    adapter._pause_requested = True
    await _fire_with_auto_resume(_fire_pre_tool_use(adapter, "search"), adapter)
    assert adapter._pause_requested is False

    stopped = [m for m in sent if m.get("type") == "event" and m.get("event") == "stopped"]
    assert len(stopped) == 1


async def test_attach_hooks_integration_with_orchestrator_tool_loop() -> None:
    """End-to-end: build a real orchestrator that fires a tool call,
    confirm the adapter sees `PreToolUse` and updates frame state
    (proving the wiring works through `HookRunner`, not just via
    direct hook-listener invocation).
    """
    from harness.hooks.events import PostToolUse, PreToolUse
    from harness.prompts.messages import ContentBlock, Message
    from harness.tools.schema import ToolCall

    class _Args(BaseModel):
        pass

    async def _noop(_a: _Args) -> str:
        return "ok"

    tool = Tool(name="probe", description="", input_model=_Args, handler=_noop)
    dispatcher = Dispatcher([tool])
    hooks = HookRunner()

    adapter = DapAdapter()
    adapter.attach_hooks(hooks)

    # Synthesize a runner that emits one tool_use cycle then a final
    # plain assistant reply.
    class _ToolRunner:
        def __init__(self) -> None:
            self._calls = 0

        async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
            self._calls += 1
            # Manually drive a single PreToolUse → dispatch → PostToolUse
            # cycle the way a real runner would.
            call = ToolCall(name="probe", arguments={}, id="c1")
            await hooks.emit(PreToolUse(call=call))
            result = await dispatcher.dispatch(call)
            await hooks.emit(PostToolUse(call=call, result=result))
            return Message(role="assistant", content=[ContentBlock(type="text", text="done")])

    runner = _ToolRunner()
    orchestrator = Orchestrator(dispatcher, hooks, runner)
    agent = SubAgent(name="t", system_prompt="", model="demo", allowed_tools=["probe"])

    history: list[Message] = [Message(role="user", content=[ContentBlock(type="text", text="hi")])]
    reply = await orchestrator.run(agent, history)
    assert reply.role == "assistant"
    # After the full cycle: frame should be back to orchestrator.
    assert adapter._current_frame == "orchestrator"
    assert adapter._active_tool_call is None


# ---------------------------------------------------------------------------
# Codex S3 regression — tool-frame turn_index must reflect real session
# progress
#
# Pre-fix: `_break_in_tool_frame` synthesized
# `DebugContext([], last_call=call, turn_index=0)` for every hook-driven
# pause. The DAP `stackTrace` response maps `ctx.turn_index + 1` to the
# source line, so a `stepIn` inside a tool during turn 5 still reported
# line 1 — the editor's source view was mislocated. Post-fix: the
# adapter tracks `_last_known_turn_index` from real turn-boundary
# checks (`break_on_predicate`) and breakpoint pauses (`_on_breakpoint`),
# and `_break_in_tool_frame` uses that value instead of zero.


async def test_break_in_tool_frame_uses_last_known_turn_index_not_zero() -> None:
    """Direct unit test: after `break_on_predicate` observes turn 3,
    a hook-synthesized tool-frame pause must report turn_index=3
    (not 0). This is the precise codex-finding regression.
    """
    from harness.debug.context import DebugContext
    from harness.prompts import text as text_msg

    adapter = DapAdapter()
    # Patch `_write` to a no-op so `_on_breakpoint`'s `stopped` event
    # doesn't try to hit a real transport — same trick the other
    # hook-driven tests use.
    await _capture_writes(adapter)

    # Drive `break_on_predicate` past three turn boundaries the way
    # `DebugRunner` would — each consult tracks the current
    # turn_index. The predicate returns False each time (no breakpoint
    # configured), but `_last_known_turn_index` updates regardless.
    for turn in range(4):  # turn boundaries 0, 1, 2, 3
        ctx = DebugContext([text_msg("user", "hi")], turn_index=turn)
        adapter.break_on_predicate(ctx)
    assert adapter._last_known_turn_index == 3

    # Editor presses stepIn; the PreToolUse listener fires inside the
    # tool dispatch that begins next.
    adapter._step_mode = "step_in"
    await _fire_with_auto_resume(_fire_pre_tool_use(adapter, "search"), adapter)

    # The synthesized context inside `_break_in_tool_frame` must carry
    # the tracked turn_index (3), not the pre-fix hard-coded 0.
    # `_current_ctx` is reset to None on resume, but the breakpoint
    # context state lives in the DAP stackTrace payload as line N+1.
    # We don't get the ctx back directly here — verify via a follow-up
    # stackTrace test below. For the unit-level assertion, check that
    # `_last_known_turn_index` survived the pause.
    assert adapter._last_known_turn_index == 3


async def test_on_breakpoint_updates_last_known_turn_index() -> None:
    """The other half of the belt-and-suspenders: when a turn-boundary
    breakpoint fires and the editor then steps into a tool, the
    synthesized tool-frame pause must inherit the turn_index from the
    breakpoint that just fired, not fall back to zero.
    """
    from harness.debug.context import DebugContext
    from harness.prompts import text as text_msg

    adapter = DapAdapter()
    # Patch `_write` to a no-op so the `stopped` event doesn't try to
    # hit a real transport.
    await _capture_writes(adapter)

    # Simulate a real breakpoint pause at turn 5 (the codex example).
    ctx_at_turn_5 = DebugContext([text_msg("user", "hi")], turn_index=5)
    # `_on_breakpoint` parks on `_continue_event`; release on the next
    # tick the way the editor's `continue` would.
    park_task: asyncio.Task[None] = asyncio.create_task(adapter._on_breakpoint(ctx_at_turn_5))
    await asyncio.sleep(0)
    adapter._continue_event.set()
    await park_task

    # `_last_known_turn_index` should now reflect the paused turn.
    assert adapter._last_known_turn_index == 5


async def test_stack_trace_reports_correct_line_for_tool_frame_pause() -> None:
    """Pin the user-visible behavior: when a tool-frame pause emits
    `stopped`, the DAP `stackTrace` response must report the source
    line corresponding to the *real* turn index, not 1.

    Drives `_break_in_tool_frame` directly with `_last_known_turn_index`
    pre-set to turn 5 (mirroring the codex finding's example: "during
    turn 5"). Then asserts the captured `stackTrace` envelope shows
    `line == 6` (turn 5 + 1). Pre-fix this was 1 — every tool-frame
    pause mislocated the editor's source view to the first turn.
    """
    adapter = DapAdapter()
    adapter.synthesize_source = lambda: [
        "turn 0",
        "turn 1",
        "turn 2",
        "turn 3",
        "turn 4",
        "turn 5",
    ]
    sent = await _capture_writes(adapter)

    # Pretend the runner has already produced five assistant turns,
    # so the next `break_on` consult would be at turn 5. Driving the
    # predicate updates `_last_known_turn_index` the same way a real
    # `DebugRunner` would.
    from harness.debug.context import DebugContext
    from harness.prompts import text as text_msg

    for turn in range(6):
        ctx = DebugContext([text_msg("user", "hi")], turn_index=turn)
        adapter.break_on_predicate(ctx)
    assert adapter._last_known_turn_index == 5

    # Editor presses stepIn while paused at the turn-5 boundary; the
    # `PreToolUse` listener fires on the next dispatch and the
    # tool-frame pause synthesizes its own `DebugContext`. We want to
    # inspect the held context's `turn_index` via the same DAP
    # `stackTrace` code path the editor would use, so we keep
    # `_current_ctx` alive by *not* signaling continue until after we
    # call `_on_stackTrace` directly.
    adapter._step_mode = "step_in"

    # Start the tool-frame pause; it will park on `_continue_event`.
    pause_task: asyncio.Task[None] = asyncio.create_task(_fire_pre_tool_use(adapter, "search"))
    # Yield once so the listener runs through to the `_continue_event.wait()`.
    await asyncio.sleep(0)

    # While the tool-frame pause is parked, call the DAP stackTrace
    # handler — same code path the editor's request walks.
    await adapter._on_stackTrace(seq=999, args={})

    # Release the park so the test can wind down cleanly.
    adapter._continue_event.set()
    await pause_task

    # Find the stackTrace response we just emitted (most recent
    # response with command == "stackTrace").
    stack_responses = [
        m for m in sent if m.get("type") == "response" and m.get("command") == "stackTrace"
    ]
    assert len(stack_responses) == 1
    frames = stack_responses[0]["body"]["stackFrames"]
    assert len(frames) == 1
    # The fix: line == 6 (turn 5 + 1). Pre-fix this was 1.
    assert frames[0]["line"] == 6, (
        f"tool-frame pause must map to the real turn's source line "
        f"(6 = turn 5 + 1); got {frames[0]['line']} — regression of codex S3"
    )
