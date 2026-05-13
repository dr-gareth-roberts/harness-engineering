"""Tests for `harness.debug.dap_protocol` framing.

These pin the wire-format contract independent of the adapter — if the
framing layer is wrong, the adapter can't recover, so this is the right
seam to lock down first.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from io import BytesIO
from typing import Any

import pytest

from harness.debug.dap_protocol import DapProtocolError, read_message, write_message


def _make_reader(payload: bytes) -> asyncio.StreamReader:
    """Build a StreamReader pre-loaded with `payload`."""
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    reader.feed_eof()
    return reader


class _CollectingWriter:
    """Stand-in for `asyncio.StreamWriter` that just collects bytes.

    Tests don't need a transport — they need to inspect what the adapter
    would have sent on the wire. Mirrors the StreamWriter surface
    `write_message` actually uses (`write` + `drain`).
    """

    def __init__(self) -> None:
        self._buf = BytesIO()

    def write(self, data: bytes) -> None:
        self._buf.write(data)

    async def drain(self) -> None:
        return None

    @property
    def collected(self) -> bytes:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Round-trip


async def test_write_then_read_recovers_the_same_dict() -> None:
    body = {"seq": 1, "type": "request", "command": "initialize"}
    writer = _CollectingWriter()
    await write_message(writer, body)  # type: ignore[arg-type]
    reader = _make_reader(writer.collected)

    decoded = await read_message(reader)
    assert decoded == body


async def test_write_uses_content_length_framing_with_blank_line_separator() -> None:
    """Pin the wire format itself — Content-Length headers + CRLF blank
    line + JSON body. Editors expect this exact layout."""
    writer = _CollectingWriter()
    await write_message(writer, {"a": 1})  # type: ignore[arg-type]

    raw = writer.collected
    # Headers come first, terminated by CRLF CRLF.
    head, _, tail = raw.partition(b"\r\n\r\n")
    assert head.startswith(b"Content-Length: ")
    # Body is the JSON encoding of {"a": 1}.
    body = json.loads(tail)
    assert body == {"a": 1}
    # Length matches what the header advertised.
    advertised = int(head.removeprefix(b"Content-Length: ").decode())
    assert len(tail) == advertised


# ---------------------------------------------------------------------------
# Header tolerance


async def test_content_length_header_is_case_insensitive() -> None:
    """The DAP spec inherits HTTP header conventions — case-insensitive."""
    body = b'{"hi":1}'
    payload = b"content-LENGTH: 8\r\n\r\n" + body
    decoded = await read_message(_make_reader(payload))
    assert decoded == {"hi": 1}


async def test_extra_headers_are_tolerated() -> None:
    """DAP traffic in the wild sometimes carries `Content-Type` too — we
    must not choke on unrecognized headers."""
    body = b'{"x":2}'
    payload = b"Content-Type: application/vnd.debugadapter\r\nContent-Length: 7\r\n\r\n" + body
    decoded = await read_message(_make_reader(payload))
    assert decoded == {"x": 2}


# ---------------------------------------------------------------------------
# Malformed input


async def test_missing_content_length_raises() -> None:
    payload = b"X-Other: 1\r\n\r\n{}"
    with pytest.raises(DapProtocolError, match="missing Content-Length"):
        await read_message(_make_reader(payload))


async def test_invalid_content_length_raises() -> None:
    payload = b"Content-Length: not-a-number\r\n\r\n{}"
    with pytest.raises(DapProtocolError, match="invalid Content-Length"):
        await read_message(_make_reader(payload))


async def test_negative_content_length_raises() -> None:
    payload = b"Content-Length: -1\r\n\r\n"
    with pytest.raises(DapProtocolError, match="negative Content-Length"):
        await read_message(_make_reader(payload))


async def test_malformed_header_line_raises() -> None:
    """A line in the header block with no `:` is malformed."""
    payload = b"NoColonHere\r\nContent-Length: 2\r\n\r\n{}"
    with pytest.raises(DapProtocolError, match="malformed header line"):
        await read_message(_make_reader(payload))


async def test_invalid_json_body_raises() -> None:
    payload = b"Content-Length: 5\r\n\r\nnotjs"
    with pytest.raises(DapProtocolError, match="invalid JSON body"):
        await read_message(_make_reader(payload))


async def test_non_object_body_raises() -> None:
    """DAP bodies must be JSON objects — arrays, strings, numbers are
    spec violations even if they parse."""
    payload = b"Content-Length: 2\r\n\r\n[]"
    with pytest.raises(DapProtocolError, match="must be a JSON object"):
        await read_message(_make_reader(payload))


# ---------------------------------------------------------------------------
# EOF semantics


async def test_clean_eof_before_message_raises_eoferror() -> None:
    """A connection closed cleanly between messages is not an error —
    the editor disconnected. The caller distinguishes this from
    DapProtocolError to handle disconnect vs. malformed-input differently.
    """
    reader = asyncio.StreamReader()
    reader.feed_eof()
    with pytest.raises(EOFError):
        await read_message(reader)


async def test_eof_in_middle_of_headers_is_protocol_error() -> None:
    """A client that closed mid-header block is buggy, not graceful."""
    reader = asyncio.StreamReader()
    reader.feed_data(b"Content-Length: 2\r\n")  # missing the second CRLF + body
    reader.feed_eof()
    with pytest.raises(DapProtocolError, match="truncated headers"):
        await read_message(reader)


async def test_eof_mid_body_raises_incomplete_read() -> None:
    """Headers said 10 bytes but only 3 arrived. `readexactly` raises
    `IncompleteReadError` — we let it bubble; the caller treats it as a
    fatal stream error."""
    reader = asyncio.StreamReader()
    reader.feed_data(b"Content-Length: 10\r\n\r\n{}")  # only 2 of 10 bytes
    reader.feed_eof()
    with pytest.raises(asyncio.IncompleteReadError):
        await read_message(reader)


# ---------------------------------------------------------------------------
# Concurrent reader (proving __aiter__-style usage works)


async def test_two_messages_back_to_back_are_decoded_in_order() -> None:
    """A real session sends many messages on one stream — the framing
    must walk forward without dropping or merging adjacent messages."""
    writer = _CollectingWriter()
    await write_message(writer, {"seq": 1, "command": "initialize"})  # type: ignore[arg-type]
    await write_message(writer, {"seq": 2, "command": "launch"})  # type: ignore[arg-type]
    reader = _make_reader(writer.collected)

    first = await read_message(reader)
    second = await read_message(reader)
    assert first["command"] == "initialize"
    assert second["command"] == "launch"


async def test_non_ascii_body_round_trips_via_utf8() -> None:
    """JSON bodies are UTF-8 — Content-Length is the *byte* count, not
    the character count. A body with a 4-byte emoji must be framed by
    its byte length."""
    body: dict[str, Any] = {"text": "hi 🚀"}  # noqa: RUF001
    writer = _CollectingWriter()
    await write_message(writer, body)  # type: ignore[arg-type]
    decoded = await read_message(_make_reader(writer.collected))
    assert decoded == body


# Mypy struggles with the StreamWriter Protocol — keep _CollectingWriter
# usage local and assert the surface explicitly.


def test_collecting_writer_satisfies_used_streamwriter_surface() -> None:
    w: Callable[..., Any] = _CollectingWriter()  # type: ignore[assignment]
    assert hasattr(w, "write")
    assert hasattr(w, "drain")
