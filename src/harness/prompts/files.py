from __future__ import annotations

from pathlib import Path

from harness.prompts.messages import ContentBlock

TRUNCATION_MARKER = "\n... [truncated]"


def attach_file(path: str | Path, *, max_bytes: int = 200_000) -> ContentBlock:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")

    data = p.read_bytes()
    if len(data) > max_bytes:
        body = data[:max_bytes].decode("utf-8", errors="replace") + TRUNCATION_MARKER
    else:
        body = data.decode("utf-8", errors="replace")

    return ContentBlock(type="file", path=str(p), text=body)
