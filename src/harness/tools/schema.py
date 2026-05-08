from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

# The handler signature uses `Any` for the input parameter rather than
# `BaseModel` because Python's `Callable` is contravariant in its
# arguments — a function `(args: MySpecificModel) -> str` is NOT
# assignable to `Callable[[BaseModel], ...]` even though every
# `MySpecificModel` IS a `BaseModel`. Tool authors write handlers that
# expect their specific input model; `Any` lets them do that without a
# `# type: ignore` at every call site. The actual input validation
# still happens at runtime via `input_model.model_validate(...)` in
# the dispatcher, so the type-laxity matches the runtime guarantee.
ToolHandler = Callable[[Any], Awaitable[Any] | Any]


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
