from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from harness.tools.schema import ToolCall, ToolResult

BlockType = Literal["text", "tool_use", "tool_result", "file", "image"]
Role = Literal["user", "assistant", "system"]


class ImageRef(BaseModel):
    """Reference to an image attached to a message.

    Two source modes (Wave 12 #7):

    - `source="base64"` — `data` is the base64-encoded image bytes.
      Used for in-memory / on-disk images. `media_type` is the MIME
      type (`image/png`, `image/jpeg`, `image/gif`, `image/webp`).
    - `source="url"` — `data` is a URL the model fetches itself.
      `media_type` is informational; the server determines the actual
      type from the URL response.

    Translation:

    - Anthropic: `{"type": "image", "source": {"type": "base64"|"url",
      "media_type": ..., "data": ...}}`. URL source requires
      vision-capable model + recent SDK.
    - OpenAI / OpenAI-compatible: `{"type": "image_url", "image_url":
      {"url": "data:image/...;base64,..." | "https://..."}}`. The
      base64 path embeds the bytes as a data URL.
    """

    source: Literal["base64", "url"]
    media_type: str
    data: str


class ContentBlock(BaseModel):
    type: BlockType
    text: str | None = None
    tool_use: ToolCall | None = None
    tool_result: ToolResult | None = None
    path: str | None = None
    cache: bool = False
    # Wave 12: optional image payload (when type == "image") and
    # optional Anthropic Files API file id (when type == "file" and
    # the caller wants a Files-API document block instead of inline
    # text). Both nullable so existing callers and serialized records
    # stay compatible.
    image: ImageRef | None = None
    file_id: str | None = None


class Message(BaseModel):
    role: Role
    content: list[ContentBlock]


def text(role: Role, s: str, *, cache: bool = False) -> Message:
    return Message(role=role, content=[ContentBlock(type="text", text=s, cache=cache)])


def assistant_tool_use(call: ToolCall) -> Message:
    return Message(role="assistant", content=[ContentBlock(type="tool_use", tool_use=call)])


def user_tool_result(result: ToolResult) -> Message:
    return Message(role="user", content=[ContentBlock(type="tool_result", tool_result=result)])
