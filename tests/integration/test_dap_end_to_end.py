"""DAP end-to-end integration test.

Drives the full happy-path lifecycle a real editor would speak —
``initialize`` → ``launch`` → ``setBreakpoints`` → ``continue`` →
``stopped`` → ``evaluate`` → ``continue`` → ``terminated`` —
through an in-process pipe pair, and pins the M1.8 invariant
(``terminated`` / ``exited`` emitted at most once each) under that
realistic driving sequence.

Differs from ``tests/debug/test_dap.py`` in scope:

- ``test_dap.py`` exercises one command per test against a minimal
  CannedRunner. The cases there cover edge paths (disconnect during
  breakpoint, lifecycle race) but never drive every DAP request in
  one trajectory.
- This test wires a Dispatcher + HookRunner + DebugRunner + Orchestrator
  through the adapter and runs the full lifecycle. A regression in any
  hand-off (the ``initialize`` → ``initialized`` ordering, the
  ``setBreakpoints`` → adapter state mutation, the ``stopped`` /
  ``continued`` parking, the post-session ``terminated`` + ``exited``)
  fails this test.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from harness.agents import Orchestrator, SubAgent
from harness.debug.dap import DapAdapter
from harness.debug.dap_protocol import read_message, write_message
from harness.debug.runner import DebugRunner
from harness.hooks import HookRunner
from harness.prompts import text
from harness.runner import CannedRunner
from harness.tools import Dispatcher, Tool
from tests.conftest import NoArgs

# ---------------------------------------------------------------------------
# In-memory transport — mirrors tests/debug/test_dap.py so this test exercises
# the same wire format the stdio transport uses, with no special-cased seam.


class _PipeWriter:
    """`StreamWriter` shim that funnels writes into a paired `StreamReader`."""

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader

    def write(self, data: bytes) -> None:
        self._reader.feed_data(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self._reader.feed_eof()


def _make_pipe() -> tuple[asyncio.StreamReader, _PipeWriter]:
    reader = asyncio.StreamReader()
    return reader, _PipeWriter(reader)


# ---------------------------------------------------------------------------


def _build_dispatcher() -> Dispatcher:
    async def noop(_args: NoArgs) -> str:
        return "ok"

    return Dispatcher(
        [
            Tool(
                name="noop",
                description="",
                input_model=NoArgs,
                handler=noop,
                idempotent=True,
            )
        ]
    )


async def _wait_event(
    recv: Callable[[], Awaitable[dict[str, Any]]],
    name: str,
    *,
    timeout: float = 1.0,
    drained: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pull messages until the named event arrives (with timeout)."""
    deadline_left = timeout
    while True:
        msg = await asyncio.wait_for(recv(), timeout=deadline_left)
        if drained is not None:
            drained.append(msg)
        if msg.get("type") == "event" and msg.get("event") == name:
            return msg


async def _drain_pending(
    recv: Callable[[], Awaitable[dict[str, Any]]],
    *,
    quiet_for: float = 0.05,
) -> list[dict[str, Any]]:
    """Drain every queued message until ``quiet_for`` seconds of silence."""
    out: list[dict[str, Any]] = []
    while True:
        try:
            msg = await asyncio.wait_for(recv(), timeout=quiet_for)
        except (TimeoutError, EOFError):
            return out
        out.append(msg)


# ---------------------------------------------------------------------------


async def test_full_lifecycle_initialize_through_terminated_emits_each_event_once(
    make_agent: Callable[..., SubAgent],
) -> None:
    """Drive the editor side through every step of a real debug session
    and confirm both the protocol-level correctness and the M1.8 at-most-once
    invariant for ``terminated`` / ``exited``.

    Sequence (the lifecycle a VS-Code-like editor would actually send):

    1. ``initialize`` — adapter responds with capabilities then emits
       ``initialized``.
    2. ``setBreakpoints`` at line 2 — adapter records turn-index 1 as a
       break point and verifies the line is in the synthesized trajectory.
    3. ``configurationDone`` — adapter acknowledges.
    4. ``launch`` — adapter starts the orchestrator-driven session.
       The first canned reply produces turn 0 (no break); the second
       reply (turn 1) is where the breakpoint fires.
    5. Adapter emits ``stopped`` (breakpoint at turn 1).
    6. ``evaluate`` ``turn_index`` — adapter resolves against the held
       DebugContext, returning ``"1"``.
    7. ``continue`` — adapter emits ``continued`` and the session resumes
       to completion.
    8. Adapter emits ``terminated`` and ``exited`` exactly once each.
    """
    in_reader, in_writer = _make_pipe()
    out_reader, out_writer = _make_pipe()

    adapter = DapAdapter()
    dispatcher = _build_dispatcher()
    hooks = HookRunner()

    # Two canned replies so we have two turns; the breakpoint fires at
    # turn 1 (the SECOND reply, after the editor has set a breakpoint on
    # DAP line 2). The synthesized trajectory exposes two lines so the
    # breakpoint at line 2 verifies cleanly.
    inner = CannedRunner(replies=["reply 0", "reply 1"])
    debug = DebugRunner(
        inner,
        break_on=adapter.break_on_predicate,
        breakpoint_callback=adapter.breakpoint_callback,
        dispatcher=dispatcher,
    )
    orchestrator = Orchestrator(dispatcher, hooks, debug)
    agent = make_agent(name="dap-agent", allowed_tools=["noop"])
    adapter.synthesize_source = lambda: ["turn 0: reply 0", "turn 1: reply 1"]

    async def _run_session() -> None:
        history: list[Any] = []
        for i in range(2):
            history.append(text("user", f"prompt {i}"))
            reply = await orchestrator.run(agent, history)
            history.append(reply)

    adapter.run_session = _run_session

    serve_task = asyncio.create_task(
        adapter.serve(in_reader, out_writer)  # type: ignore[arg-type]
    )

    async def send(msg: dict[str, Any]) -> None:
        await write_message(in_writer, msg)  # type: ignore[arg-type]

    async def recv() -> dict[str, Any]:
        return await read_message(out_reader)

    try:
        # 1) initialize → response + `initialized` event.
        await send({"seq": 1, "type": "request", "command": "initialize"})
        init_resp = await recv()
        assert init_resp["type"] == "response"
        assert init_resp["command"] == "initialize"
        assert init_resp["success"] is True
        assert init_resp["body"]["supportsTerminateRequest"] is True

        initialized_evt = await recv()
        assert initialized_evt["type"] == "event"
        assert initialized_evt["event"] == "initialized"

        # 2) setBreakpoints — request a break at DAP line 2 (= turn 1).
        await send(
            {
                "seq": 2,
                "type": "request",
                "command": "setBreakpoints",
                "arguments": {"breakpoints": [{"line": 2}]},
            }
        )
        bp_resp = await recv()
        assert bp_resp["success"] is True
        assert bp_resp["body"]["breakpoints"][0]["verified"] is True
        assert bp_resp["body"]["breakpoints"][0]["line"] == 2

        # 3) configurationDone.
        await send({"seq": 3, "type": "request", "command": "configurationDone"})
        cd_resp = await recv()
        assert cd_resp["command"] == "configurationDone"
        assert cd_resp["success"] is True

        # 4) launch — starts the session task. Adapter sends only the
        # response synchronously; `stopped` arrives once the breakpoint
        # fires inside the orchestrator turn.
        await send({"seq": 4, "type": "request", "command": "launch"})
        launch_resp = await recv()
        assert launch_resp["command"] == "launch"
        assert launch_resp["success"] is True

        # 5) Wait for the breakpoint — at turn_index == 1. Turn 0 should
        # have produced a canned reply unobserved by the editor.
        stopped = await _wait_event(recv, "stopped", timeout=2.0)
        assert stopped["body"]["reason"] == "breakpoint"
        assert stopped["body"]["threadId"] == DapAdapter.THREAD_ID

        # 6) evaluate `turn_index` against the held DebugContext.
        await send(
            {
                "seq": 5,
                "type": "request",
                "command": "evaluate",
                "arguments": {"expression": "turn_index"},
            }
        )
        eval_resp = await asyncio.wait_for(recv(), timeout=1.0)
        assert eval_resp["success"] is True
        # `turn_index` was 1 by the time the second runner invocation
        # entered: one assistant reply (turn 0) is already in history.
        assert eval_resp["body"]["result"] == "1"
        assert eval_resp["body"]["type"] == "int"

        # 7) continue — releases the breakpoint, orchestrator runs to
        # the end of its loop.
        await send({"seq": 6, "type": "request", "command": "continue"})
        cont_resp = await recv()
        assert cont_resp["command"] == "continue"
        assert cont_resp["success"] is True
        assert cont_resp["body"]["allThreadsContinued"] is True

        continued_evt = await _wait_event(recv, "continued", timeout=1.0)
        assert continued_evt["body"]["threadId"] == DapAdapter.THREAD_ID

        # 8) Adapter emits `terminated` once the session task finishes.
        # Drain everything queued and assert each lifecycle endpoint
        # exactly once — the M1.8 invariant under a real driving flow.
        drained: list[dict[str, Any]] = []
        await _wait_event(recv, "terminated", timeout=2.0, drained=drained)
        drained.extend(await _drain_pending(recv))

        events = [m for m in drained if m.get("type") == "event"]
        terminated_count = sum(1 for m in events if m.get("event") == "terminated")
        exited_count = sum(1 for m in events if m.get("event") == "exited")
        assert terminated_count == 1, (
            "expected exactly one 'terminated' event across the lifecycle, "
            f"got {terminated_count}; events={[m.get('event') for m in events]}"
        )
        assert exited_count == 1, (
            "expected exactly one 'exited' event across the lifecycle, "
            f"got {exited_count}; events={[m.get('event') for m in events]}"
        )
    finally:
        if not serve_task.done():
            serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await serve_task
