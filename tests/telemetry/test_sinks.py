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
    assert await NullSink().emit(make_event()) is None


async def test_memory_sink_collects_in_order() -> None:
    sink = MemorySink()
    events = [make_event(f"t{i}") for i in range(3)]
    for e in events:
        await sink.emit(e)
    assert [e.tool_name for e in sink.events] == ["t0", "t1", "t2"]


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

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["tool_name"] == "a"
    assert json.loads(lines[1])["tool_name"] == "b"

    # Re-open with another sink — must not truncate.
    sink2 = JSONLSink(target)
    await sink2.emit(make_event("c"))
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


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
    assert await t.emit(make_event()) is None


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
