from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from harness.prompts.messages import ContentBlock, ImageRef

TRUNCATION_MARKER = "\n... [truncated]"


def attach_file(
    path: str | Path | None = None,
    *,
    file_id: str | None = None,
    max_bytes: int = 200_000,
) -> ContentBlock:
    """Build a `file` ContentBlock from a local path or a Files API id.

    Two modes:

    - `attach_file(path=...)` — read the file off disk; the runner
      will inline it as text (`<file path=...>...</file>`).
      `max_bytes` caps the inlined size; over-cap files are
      truncated with a marker. The historical default behavior.
    - `attach_file(file_id="file_...")` — Wave 12 #8: reference an
      Anthropic Files API document by id. The AnthropicRunner
      translates this to a `document` content block referring to the
      uploaded file rather than inlining text. Vendors without an
      equivalent API (OpenAI-compat) fall back to a textual
      `<file file_id=...>` placeholder; users on those providers
      should keep the path-based mode.

    Exactly one of `path` or `file_id` must be supplied.
    """
    if (path is None) == (file_id is None):
        raise ValueError("attach_file: pass exactly one of `path` or `file_id`")

    if file_id is not None:
        return ContentBlock(type="file", file_id=file_id)

    assert path is not None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")

    data = p.read_bytes()
    if len(data) > max_bytes:
        body = data[:max_bytes].decode("utf-8", errors="replace") + TRUNCATION_MARKER
    else:
        body = data.decode("utf-8", errors="replace")

    return ContentBlock(type="file", path=str(p), text=body)


def attach_image(
    path: str | Path | None = None,
    *,
    url: str | None = None,
    media_type: str | None = None,
) -> ContentBlock:
    """Build an `image` ContentBlock from a local path or a URL.

    - `attach_image(path=...)` reads the file, base64-encodes it, and
      embeds it inline. `media_type` is auto-detected from the file
      extension when omitted; pass it explicitly for files without an
      extension or with ambiguous ones.
    - `attach_image(url=..., media_type=...)` references the image by
      URL — the model fetches it. `media_type` is required since we
      can't sniff a remote resource without fetching.

    Exactly one of `path` or `url` must be supplied. Wave 12 #7.
    """
    if (path is None) == (url is None):
        raise ValueError("attach_image: pass exactly one of `path` or `url`")

    if url is not None:
        if media_type is None:
            raise ValueError("attach_image(url=...) requires media_type")
        return ContentBlock(
            type="image",
            image=ImageRef(source="url", media_type=media_type, data=url),
        )

    assert path is not None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such image: {p}")

    if media_type is None:
        guessed, _ = mimetypes.guess_type(str(p))
        if guessed is None:
            raise ValueError(
                f"attach_image: cannot infer media_type for {p}; pass media_type=... explicitly"
            )
        media_type = guessed

    encoded = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    return ContentBlock(
        type="image",
        image=ImageRef(source="base64", media_type=media_type, data=encoded),
    )
