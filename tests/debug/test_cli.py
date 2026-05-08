from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pytest

from harness.agents import SubAgent
from harness.debug.cli import (
    _cmd_debug,
    load_session,
    parse_break_spec,
    register,
)
from harness.memory.record import SessionRecord
from harness.prompts.messages import ContentBlock, Message, text
from harness.tools.schema import ToolCall

# ---------- helpers


def _agent() -> SubAgent:
    return SubAgent(name="x", system_prompt="", model="test-model")


def _record_with_two_turns() -> SessionRecord:
    msgs = [
        text("user", "hello"),
        text("assistant", "hi back"),
        text("user", "more"),
        text("assistant", "ok"),
    ]
    return SessionRecord(session_id="s1", agent=_agent(), messages=msgs)


def _record_with_tool_use() -> SessionRecord:
    call = ToolCall(name="delete", arguments={"path": "/x"}, id="tu_1")
    msgs = [
        text("user", "delete that"),
        Message(
            role="assistant",
            content=[ContentBlock(type="tool_use", tool_use=call)],
        ),
    ]
    return SessionRecord(session_id="s2", agent=_agent(), messages=msgs)


def _write_json(tmp_path: Path, record: SessionRecord) -> Path:
    path = tmp_path / f"{record.session_id}.json"
    path.write_text(record.model_dump_json(), encoding="utf-8")
    return path


def _write_jsonl(tmp_path: Path, records: list[SessionRecord]) -> Path:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(r.model_dump_json() for r in records) + "\n",
        encoding="utf-8",
    )
    return path


# ---------- load_session


def test_load_session_from_json(tmp_path: Path) -> None:
    record = _record_with_two_turns()
    path = _write_json(tmp_path, record)

    loaded = load_session(path)
    assert loaded.session_id == "s1"
    assert len(loaded.messages) == 4


def test_load_session_from_jsonl_takes_last_line(tmp_path: Path) -> None:
    older = SessionRecord(
        session_id="s1",
        agent=_agent(),
        messages=[text("user", "old")],
    )
    newer = SessionRecord(
        session_id="s1",
        agent=_agent(),
        messages=[text("user", "old"), text("assistant", "new")],
    )
    path = _write_jsonl(tmp_path, [older, newer])

    loaded = load_session(path)
    assert len(loaded.messages) == 2


def test_load_session_empty_jsonl_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty JSONL"):
        load_session(path)


# ---------- parse_break_spec


def test_parse_break_spec_turn() -> None:
    assert parse_break_spec("turn=5") == ("turn", "5")


def test_parse_break_spec_tool() -> None:
    assert parse_break_spec("tool=delete") == ("tool", "delete")


def test_parse_break_spec_invalid_key() -> None:
    with pytest.raises(ValueError):
        parse_break_spec("frobnicate=5")


def test_parse_break_spec_no_equals() -> None:
    with pytest.raises(ValueError):
        parse_break_spec("turn:5")


# ---------- register attaches the subcommand


def test_register_attaches_debug_subcommand() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)

    args = parser.parse_args(["debug", "/tmp/x.json", "--break", "turn=3"])
    assert args.command == "debug"
    assert args.path == "/tmp/x.json"
    assert args.break_spec == "turn=3"
    assert args.func is not None


# ---------- _cmd_debug end-to-end (spec test #7)


def test_cmd_debug_breaks_at_configured_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec test #7 (no pexpect): build an argparse Namespace, monkeypatch
    sys.stdin to a StringIO that issues `resume`, and assert _cmd_debug
    returns 0 with breakpoint output captured."""
    record = _record_with_two_turns()
    path = _write_json(tmp_path, record)

    # Stdin scripted: resume immediately each time the REPL opens.
    monkeypatch.setattr(sys, "stdin", io.StringIO("resume\nresume\nresume\n"))

    args = argparse.Namespace(
        path=str(path),
        break_spec="turn=0",
        func=_cmd_debug,
    )
    rc = _cmd_debug(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "loaded session s1" in captured.out
    assert "paused at turn" in captured.out


def test_cmd_debug_with_tool_break(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _record_with_tool_use()
    path = _write_json(tmp_path, record)

    monkeypatch.setattr(sys, "stdin", io.StringIO("last_call\nresume\n"))

    args = argparse.Namespace(
        path=str(path),
        break_spec="tool=delete",
        func=_cmd_debug,
    )
    rc = _cmd_debug(args)

    captured = capsys.readouterr()
    assert rc == 0
    # `last_call` printed via REPL; tool name appears in stdout.
    assert "delete" in captured.out


def test_cmd_debug_aborted_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _record_with_two_turns()
    path = _write_json(tmp_path, record)

    monkeypatch.setattr(sys, "stdin", io.StringIO("abort\n"))

    args = argparse.Namespace(
        path=str(path),
        break_spec="turn=0",
        func=_cmd_debug,
    )
    rc = _cmd_debug(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "aborted" in captured.out


def test_cmd_debug_bad_path_returns_error_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        path=str(tmp_path / "does-not-exist.json"),
        break_spec="turn=0",
        func=_cmd_debug,
    )
    rc = _cmd_debug(args)

    assert rc == 2
    assert "failed to load" in capsys.readouterr().out


def test_cmd_debug_bad_break_spec_returns_error_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _record_with_two_turns()
    path = _write_json(tmp_path, record)

    args = argparse.Namespace(
        path=str(path),
        break_spec="garbage",
        func=_cmd_debug,
    )
    rc = _cmd_debug(args)

    assert rc == 2
    assert "bad --break" in capsys.readouterr().out


# ---------- top-level dispatcher integration


def test_top_level_main_dispatches_debug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Confirms the cli.py dispatcher actually forwards to _cmd_debug."""
    from harness.cli import main

    record = _record_with_two_turns()
    path = _write_json(tmp_path, record)
    monkeypatch.setattr(sys, "stdin", io.StringIO("resume\nresume\nresume\n"))

    rc = main(["debug", str(path), "--break", "turn=0"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "loaded session" in captured.out


def test_top_level_main_no_command_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from harness.cli import main

    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "harness" in out
    assert "debug" in out
