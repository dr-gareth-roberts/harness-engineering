from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from harness.tools.schema import Tool, ToolCall, ToolResult


class Dispatcher:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def tools_schema(self) -> list[dict[str, Any]]:
        return [tool.json_schema() for tool in self._tools.values()]

    async def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                id=call.id,
                content=f"unknown tool: {call.name!r}",
                is_error=True,
            )

        try:
            args = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(id=call.id, content=str(exc), is_error=True)

        try:
            result = tool.handler(args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 - surfaced as ToolResult, not swallowed
            return ToolResult(id=call.id, content=str(exc), is_error=True)

        return ToolResult(id=call.id, content=result, is_error=False)
