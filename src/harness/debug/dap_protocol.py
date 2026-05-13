"""DAP wire-protocol framing.

DAP messages are JSON bodies preceded by an HTTP-like header block. The
only required header is `Content-Length` (octets in the body). A blank
line separates headers from body. Example:

    Content-Length: 119\\r\\n
    \\r\\n
    {"seq":153,"type":"request","command":"next","arguments":{"threadId":3}}

This module owns the raw bytes ↔ structured-message conversion. Higher
layers (`harness.debug.dap`, `harness.debug.dap_messages`) deal with
typed message shapes; this layer just ensures we get one valid JSON
body in and out at a time.

Reader/writer are typed against `asyncio.StreamReader` /
`asyncio.StreamWriter` so the same code drives stdio (via
`connect_read_pipe`/`connect_write_pipe`) and in-memory test pairs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


class DapProtocolError(Exception):
    """Raised when the wire format is broken — malformed headers, missing
    `Content-Length`, truncated body, invalid JSON. Conventionally the
    adapter should disconnect after one of these; the editor's state is
    not recoverable.
    """


_HEADER_TERMINATOR = b"\r\n\r\n"
_LINE_TERMINATOR = b"\r\n"


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one DAP message from `reader` and return its JSON body as a dict.

    Raises:
        DapProtocolError: malformed headers, missing/invalid Content-Length,
            truncated body, or non-JSON / non-object body.
        EOFError: stream closed cleanly between messages (the caller should
            treat this as "client disconnected" rather than an error).
    """
    raw_headers = await _read_until(reader, _HEADER_TERMINATOR)
    if raw_headers == b"":
        # Clean EOF before any data — the editor disconnected.
        raise EOFError("DAP stream closed before next message")

    headers = _parse_headers(raw_headers)
    raw_length = headers.get("content-length")
    if raw_length is None:
        raise DapProtocolError(f"missing Content-Length header in {headers!r}")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise DapProtocolError(f"invalid Content-Length: {raw_length!r}") from exc
    if length < 0:
        raise DapProtocolError(f"negative Content-Length: {length}")

    body = await reader.readexactly(length)
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DapProtocolError(f"invalid JSON body: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise DapProtocolError(f"DAP body must be a JSON object, got {type(decoded).__name__}")
    return decoded


async def write_message(writer: asyncio.StreamWriter, body: dict[str, Any]) -> None:
    """Encode `body` as JSON and write it framed with `Content-Length`.

    The write is flushed via `drain()` so callers can rely on bytes being
    on the wire before `await write_message(...)` returns.
    """
    encoded = json.dumps(body, default=str).encode("utf-8")
    header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
    writer.write(header + encoded)
    await writer.drain()


# ---------------------------------------------------------------------------
# Internals


async def _read_until(reader: asyncio.StreamReader, terminator: bytes) -> bytes:
    """Read from `reader` until `terminator` is seen and return everything
    that came before it (without the terminator).

    Returns `b""` on clean EOF before any bytes are read — used to signal
    a graceful disconnect by the caller, not malformed input.
    """
    try:
        # readuntil includes the terminator; strip it before returning.
        data = await reader.readuntil(terminator)
    except asyncio.IncompleteReadError as exc:
        if exc.partial == b"":
            return b""
        raise DapProtocolError(
            f"truncated headers (got {len(exc.partial)} bytes before EOF)"
        ) from exc
    return data[: -len(terminator)]


def _parse_headers(raw: bytes) -> dict[str, str]:
    """Parse `Header: Value\\r\\nHeader: Value` into a lowercase-keyed dict.

    Header names are case-insensitive per the DAP spec (which mirrors
    HTTP). We lowercase keys so callers can do a flat dict lookup.
    """
    out: dict[str, str] = {}
    for line in raw.split(_LINE_TERMINATOR):
        if not line:
            continue
        if b":" not in line:
            raise DapProtocolError(f"malformed header line: {line!r}")
        name, _, value = line.partition(b":")
        out[name.decode("ascii").strip().lower()] = value.decode("ascii").strip()
    return out
