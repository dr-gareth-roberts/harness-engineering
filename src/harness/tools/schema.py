from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

ToolHandler = Callable[[BaseModel], Awaitable[Any] | Any]


class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    idempotent: bool = False
    """If True, this tool can be safely re-executed with the same arguments —
    no side effects beyond what the first call produced. Reserved for the
    speculative-execution feature (`harness.speculate`); ignored everywhere
    else. Mark `True` only for read-only tools (`search`, `read_file`, etc.).
    """

    def json_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    id: str | None = None


class ToolResult(BaseModel):
    id: str | None = None
    content: Any = None
    is_error: bool = False
