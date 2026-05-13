from __future__ import annotations

import io
import json
from pathlib import Path

from harness.telemetry import (
    JSONLSink,
    MemorySink,
    MultiSink,
    NullSink,
    Telemetry,
    ToolDispatched,
)


def make_event(name: str = "echo") -> ToolDispatched:
    return ToolDispatched(
        tool_name=name,
        call_id="c1",
        arguments={"x": 1},
        is_error=False,
        duration_ms=1.0,
    )


async def test_null_sink_returns_none() -> None:
    # `emit` returns None by signature; awaiting it is a smoke check.
    await NullSink().emit(make_event())


async def test_memory_sink_collects_in_order() -> None:
    sink = MemorySink()
    events = [make_event(f"t{i}") for i in range(3)]
    for e in events:
        await sink.emit(e)
    # MemorySink.events is `list[TelemetryEvent]` — the abstract base
    # has no `tool_name` attribute. Narrow to the concrete type the
    # test created.
    assert all(isinstance(e, ToolDispatched) for e in sink.events)
    assert [e.tool_name for e in sink.events if isinstance(e, ToolDispatched)] == [
        "t0",
        "t1",
        "t2",
    ]


async def test_jsonl_sink_to_stream_writes_one_line_per_event() -> None:
    buf = io.StringIO()
    sink = JSONLSink(buf)
    await sink.emit(make_event("a"))
    await sink.emit(make_event("b"))

    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["tool_name"] for p in parsed] == ["a", "b"]
    assert all(p["kind"] == "tool.dispatched" for p in parsed)


async def test_jsonl_sink_to_path_appends(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    sink = JSONLSink(target)
    await sink.emit(make_event("a"))
    await sink.emit(make_event("b"))
    await sink.close()

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["tool_name"] == "a"
    assert json.loads(lines[1])["tool_name"] == "b"

    # Re-open with another sink — must not truncate.
    sink2 = JSONLSink(target)
    await sink2.emit(make_event("c"))
    await sink2.close()
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


async def test_jsonl_sink_held_handle_scale(tmp_path: Path) -> None:
    """1000 events must produce 1000 valid JSON lines in order.

    Pins the held-handle behavior: the file is opened once and writes
    flush per-event, so a tight loop does not pay open+close syscalls
    per emit and no events are lost.
    """
    target = tmp_path / "scale.jsonl"
    sink = JSONLSink(target)
    try:
        for i in range(1000):
            await sink.emit(make_event(f"t{i}"))
    finally:
        await sink.close()

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1000
    parsed = [json.loads(line) for line in lines]
    assert [p["tool_name"] for p in parsed] == [f"t{i}" for i in range(1000)]


async def test_jsonl_sink_async_context_manager_closes(tmp_path: Path) -> None:
    """`async with JSONLSink(p) as sink:` must close the handle on exit."""
    target = tmp_path / "ctx.jsonl"
    async with JSONLSink(target) as sink:
        await sink.emit(make_event("a"))
        # Handle is held while inside the context.
        assert sink._stream is not None

    # After context exit, the handle is released.
    assert sink._stream is None

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["tool_name"] == "a"


async def test_jsonl_sink_emit_after_close_reopens(tmp_path: Path) -> None:
    """Emit, close, emit again — both events present, in order."""
    target = tmp_path / "reopen.jsonl"
    sink = JSONLSink(target)
    await sink.emit(make_event("first"))
    await sink.close()
    # Re-emit after close should reopen the file (append mode).
    await sink.emit(make_event("second"))
    await sink.close()

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["tool_name"] == "first"
    assert json.loads(lines[1])["tool_name"] == "second"


async def test_jsonl_sink_close_is_idempotent(tmp_path: Path) -> None:
    """`close()` called twice must not raise."""
    target = tmp_path / "idemp.jsonl"
    sink = JSONLSink(target)
    await sink.emit(make_event("a"))
    await sink.close()
    # Second close — must be a clean no-op.
    await sink.close()
    # Close before any emit also must not raise.
    sink2 = JSONLSink(target)
    await sink2.close()


async def test_jsonl_sink_close_does_not_close_caller_stream() -> None:
    """For stream-backed sinks, `close()` leaves the caller's stream open."""
    buf = io.StringIO()
    sink = JSONLSink(buf)
    await sink.emit(make_event("a"))
    await sink.close()
    # Caller retains ownership — stream remains usable.
    assert not buf.closed
    # Subsequent emit on the same sink still works against the caller stream.
    await sink.emit(make_event("b"))
    assert not buf.closed
    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["tool_name"] for line in lines] == ["a", "b"]


async def test_multi_sink_fans_out_and_isolates_failures() -> None:
    class Boom:
        async def emit(self, event):  # type: ignore[no-untyped-def]
            raise RuntimeError("kaboom")

    good = MemorySink()
    sink = MultiSink(Boom(), good, NullSink())

    await sink.emit(make_event())
    assert len(good.events) == 1


async def test_telemetry_swallows_sink_exceptions() -> None:
    class Boom:
        async def emit(self, event):  # type: ignore[no-untyped-def]
            raise RuntimeError("kaboom")

    t = Telemetry(Boom())
    # Must not raise:
    await t.emit(make_event())


async def test_telemetry_default_is_null_sink() -> None:
    t = Telemetry()
    # Smoke: emit returns None and does not error.
    await t.emit(make_event())


async def test_jsonify_handles_non_native_arguments() -> None:
    """ToolDispatched coerces non-JSON-native arguments via jsonify().

    Constructed at the dispatcher layer, but we test the coercion helper here
    so the contract is durable.
    """
    from harness.telemetry.events import jsonify

    out = jsonify({"path": Path("/tmp/x"), "n": 1})
    assert isinstance(out, dict)
    assert out["path"] == "/tmp/x"
    assert out["n"] == 1
