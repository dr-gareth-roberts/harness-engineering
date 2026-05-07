from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from harness.tools.schema import ToolCall, ToolResult

BlockType = Literal["text", "tool_use", "tool_result", "file"]
Role = Literal["user", "assistant", "system"]


class ContentBlock(BaseModel):
    type: BlockType
    text: str | None = None
    tool_use: ToolCall | None = None
    tool_result: ToolResult | None = None
    path: str | None = None
    cache: bool = False


class Message(BaseModel):
    role: Role
    content: list[ContentBlock]


def text(role: Role, s: str, *, cache: bool = False) -> Message:
    return Message(role=role, content=[ContentBlock(type="text", text=s, cache=cache)])


def assistant_tool_use(call: ToolCall) -> Message:
    return Message(role="assistant", content=[ContentBlock(type="tool_use", tool_use=call)])


def user_tool_result(result: ToolResult) -> Message:
    return Message(role="user", content=[ContentBlock(type="tool_result", tool_result=result)])
