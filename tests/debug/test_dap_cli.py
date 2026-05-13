"""End-to-end subprocess test for `harness debug --dap`.

Wave 7 wired the DAP server into the CLI but only smoke-tested it
manually. This test spawns the actual `harness debug --dap` process,
speaks DAP over its stdin/stdout, and asserts the full
launch -> break -> continue -> terminated flow round-trips.

The test is a real subprocess (via `asyncio.create_subprocess_exec`,
the no-shell equivalent of `execFile`) so it exercises the same
`connect_read_pipe` / `connect_write_pipe` plumbing the CLI uses for
real editor integrations. A regression there would surface here, not
in production.

`uv run` cold-start is ~1s, so this test is slower than the in-process
DAP tests in `test_dap.py`. Keep it focused on the integration shape;
detailed behavior lives in the in-process tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from harness.agents import SubAgent
from harness.debug.dap_protocol import read_message, write_message
from harness.memory.record import SessionRecord
from harness.prompts import text


@pytest.fixture
def session_path(tmp_path: Path) -> Path:
    """Write a minimal SessionRecord to disk for the CLI to load."""
    record = SessionRecord(
        session_id="cli-smoke",
        agent=SubAgent(
            name="cli-smoke-agent",
            system_prompt="",
            model="demo",
            allowed_tools=[],
        ),
        messages=[
            text("user", "hello"),
            text("assistant", "hi back"),
        ],
    )
    p = tmp_path / "session.json"
    p.write_text(record.model_dump_json(indent=2))
    return p


async def _wait_event(
    reader: asyncio.StreamReader,
    name: str,
    *,
    timeout: float = 5.0,
    drained: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Drain messages until an event with `event == name` arrives."""
    while True:
        msg = await asyncio.wait_for(read_message(reader), timeout=timeout)
        if drained is not None:
            drained.append(msg)
        if msg.get("type") == "event" and msg.get("event") == name:
            return msg


async def _wait_response(
    reader: asyncio.StreamReader,
    request_seq: int,
    *,
    timeout: float = 5.0,
    drained: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Drain messages until a response for `request_seq` arrives."""
    while True:
        msg = await asyncio.wait_for(read_message(reader), timeout=timeout)
        if drained is not None:
            drained.append(msg)
        if msg.get("type") == "response" and msg.get("request_seq") == request_seq:
            return msg


# ---------------------------------------------------------------------------


async def test_dap_cli_round_trips_initialize_launch_continue_terminated(
    session_path: Path,
) -> None:
    """Full editor-style flow against the real `harness debug --dap`
    process: initialize -> launch -> see stopped -> continue -> terminated.
    Validates the subprocess wiring (`connect_read_pipe` /
    `connect_write_pipe` in `harness.debug.cli._serve_stdio`)."""

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "harness.cli",
        "debug",
        "--dap",
        str(session_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert proc.stdin is not None
    assert proc.stdout is not None

    try:
        # Initialize.
        await write_message(proc.stdin, {"seq": 1, "type": "request", "command": "initialize"})
        init_resp = await _wait_response(proc.stdout, 1)
        assert init_resp["success"] is True
        # The `initialized` event must follow the response.
        await _wait_event(proc.stdout, "initialized")

        # setBreakpoints at line 1 (turn_index 0).
        await write_message(
            proc.stdin,
            {
                "seq": 2,
                "type": "request",
                "command": "setBreakpoints",
                "arguments": {"breakpoints": [{"line": 1}]},
            },
        )
        await _wait_response(proc.stdout, 2)

        # configurationDone.
        await write_message(
            proc.stdin,
            {"seq": 3, "type": "request", "command": "configurationDone"},
        )
        await _wait_response(proc.stdout, 3)

        # launch -- kicks off the orchestrator.
        await write_message(
            proc.stdin,
            {"seq": 4, "type": "request", "command": "launch"},
        )
        await _wait_response(proc.stdout, 4)

        # Wait for stopped (breakpoint hit).
        stopped = await _wait_event(proc.stdout, "stopped")
        assert stopped["body"]["reason"] == "breakpoint"

        # continue.
        await write_message(
            proc.stdin,
            {"seq": 5, "type": "request", "command": "continue"},
        )
        await _wait_response(proc.stdout, 5)

        # The session finishes; adapter emits `terminated`.
        await _wait_event(proc.stdout, "terminated")

        # disconnect to wind down cleanly.
        await write_message(
            proc.stdin,
            {"seq": 6, "type": "request", "command": "disconnect"},
        )
        await _wait_response(proc.stdout, 6)
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except TimeoutError:
            proc.terminate()
            await proc.wait()


def test_session_record_json_loads_cleanly(session_path: Path) -> None:
    """Smoke test that the JSON we're round-tripping is valid."""
    parsed = json.loads(session_path.read_text())
    assert parsed["session_id"] == "cli-smoke"
