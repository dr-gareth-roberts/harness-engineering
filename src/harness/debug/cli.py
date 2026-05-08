from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from harness.agents.orchestrator import Orchestrator
from harness.debug.context import DebugContext
from harness.debug.dap import DapAdapter
from harness.debug.runner import DebugAborted, DebugRunner
from harness.hooks.runner import HookRunner
from harness.memory.record import SessionRecord
from harness.prompts.messages import Message
from harness.replay.runner import ReplayMismatch, ReplayRunner
from harness.tools.dispatcher import Dispatcher


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the `harness debug` subcommand on a top-level parser."""
    p = subparsers.add_parser(
        "debug",
        help="Interactive debug REPL on a recorded session (#10)",
    )
    p.add_argument(
        "path",
        help="Path to a recorded SessionRecord (.json or .jsonl)",
    )
    p.add_argument(
        "--break",
        dest="break_spec",
        default="turn=0",
        help="Break condition, e.g. turn=5, tool=delete (default: turn=0)",
    )
    p.add_argument(
        "--dap",
        dest="dap",
        action="store_true",
        help=(
            "Speak the Debug Adapter Protocol over stdio instead of running "
            "the interactive REPL. Used by IDE integrations (VS Code, "
            "neovim DAP, etc.) — the editor launches the process, sends "
            "DAP requests, and drives the same replay-driven debug session."
        ),
    )
    p.set_defaults(func=_cmd_debug)


def load_session(path: str | Path) -> SessionRecord:
    """Load a `SessionRecord` from a `.json` or `.jsonl` file.

    For `.jsonl`, each line is treated as a `SessionRecord` snapshot and the
    last non-empty line wins (so an interrupted-and-resumed run reads as the
    latest state). For `.json`, the whole file is a single record.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        lines = [line for line in raw.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"empty JSONL file: {p}")
        return SessionRecord.model_validate_json(lines[-1])
    return SessionRecord.model_validate_json(raw)


def parse_break_spec(spec: str) -> tuple[str, str]:
    """Parse a `key=value` break specifier.

    Recognized keys: `turn` (integer), `tool` (string, matches the most
    recent tool_use name).
    """
    if "=" not in spec:
        raise ValueError(f"break spec must be key=value, got {spec!r}")
    key, value = spec.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key not in ("turn", "tool"):
        raise ValueError(f"unsupported break key: {key!r} (expected 'turn' or 'tool')")
    return key, value


def _make_break_predicate(key: str, value: str):  # type: ignore[no-untyped-def]
    if key == "turn":
        try:
            target = int(value)
        except ValueError as exc:
            raise ValueError(f"--break turn=<int> requires an integer, got {value!r}") from exc

        def _by_turn(ctx: DebugContext) -> bool:
            return ctx.turn_index == target

        return _by_turn

    # key == "tool"
    target_name = value

    def _by_tool(ctx: DebugContext) -> bool:
        return ctx.last_call is not None and ctx.last_call.name == target_name

    return _by_tool


def _cmd_debug(args: argparse.Namespace) -> int:
    try:
        record = load_session(args.path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"[harness-debug] failed to load session: {exc}")
        return 2

    # `getattr` so programmatic test invocations that build their own
    # Namespace don't have to remember every flag the parser would set.
    if getattr(args, "dap", False):
        return _run_dap_session(record)

    try:
        key, value = parse_break_spec(args.break_spec)
    except ValueError as exc:
        print(f"[harness-debug] bad --break: {exc}")
        return 2

    predicate = _make_break_predicate(key, value)
    replay = ReplayRunner.from_record(record)
    dispatcher = Dispatcher()
    hooks = HookRunner()

    debug = DebugRunner(
        replay,
        break_on=predicate,
        interactive=True,
        dispatcher=dispatcher,
    )
    orchestrator = Orchestrator(dispatcher, hooks, debug)

    print(f"[harness-debug] loaded session {record.session_id}, break={args.break_spec}")

    try:
        asyncio.run(_drive(orchestrator, record))
    except DebugAborted as exc:
        print(f"[harness-debug] aborted: {exc}")
        return 0
    except ReplayMismatch:
        # Reaching the end of the recording is normal once we resume past
        # the last turn — surface it as a clean exit, not a crash.
        print("[harness-debug] replay exhausted; debug session complete")
        return 0
    return 0


def _trajectory_lines(record: SessionRecord) -> list[str]:
    """Synthesize one line per assistant turn for DAP `source` requests.

    Each line summarizes the assistant message: any text blocks first,
    then a parenthetical for tool_use blocks. The result is what an
    editor renders when the breakpoint frame asks for the source.
    """
    lines: list[str] = []
    for msg in record.messages:
        if msg.role != "assistant":
            continue
        parts: list[str] = []
        for block in msg.content:
            if block.type == "text" and block.text:
                parts.append(block.text)
            elif block.type == "tool_use" and block.tool_use is not None:
                parts.append(f"(tool_use {block.tool_use.name})")
        lines.append(" ".join(parts) if parts else "(empty)")
    return lines


def _run_dap_session(record: SessionRecord) -> int:
    """Run the same replay-driven debug session under DAP control.

    The DAP adapter consumes stdin, writes to stdout, and runs the
    orchestrator concurrently. Editor integrations (VS Code launch
    config, neovim-dap, etc.) launch the process and speak DAP over
    those pipes.
    """
    adapter = DapAdapter()
    replay = ReplayRunner.from_record(record)
    dispatcher = Dispatcher()
    hooks = HookRunner()

    debug = DebugRunner(
        replay,
        break_on=adapter.break_on_predicate,
        breakpoint_callback=adapter.breakpoint_callback,
        dispatcher=dispatcher,
    )
    orchestrator = Orchestrator(dispatcher, hooks, debug)

    async def _run() -> None:
        history: list[Message] = []
        for msg in record.messages:
            if msg.role == "user":
                history.append(msg)
                try:
                    reply = await orchestrator.run(record.agent, history)
                except (ReplayMismatch, DebugAborted):
                    return
                history.append(reply)

    adapter.run_session = _run
    adapter.synthesize_source = lambda: _trajectory_lines(record)

    asyncio.run(_serve_stdio(adapter))
    return 0


async def _serve_stdio(adapter: DapAdapter) -> None:
    """Wire the adapter to the process's actual stdin/stdout streams."""
    loop = asyncio.get_event_loop()

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin,
        sys.stdout,
    )
    writer = asyncio.StreamWriter(
        writer_transport,
        writer_protocol,
        None,
        loop,
    )

    await adapter.serve(reader, writer)


async def _drive(orchestrator: Orchestrator, record: SessionRecord) -> None:
    """Replay each user turn through the orchestrator with a clean message
    history, so the breakpoint sees a context shaped like the recorded run.
    """
    history = []
    for msg in record.messages:
        if msg.role == "user":
            history.append(msg)
            try:
                reply = await orchestrator.run(record.agent, history)
            except ReplayMismatch:
                # No more recorded replies — stop replaying.
                return
            history.append(reply)
