from __future__ import annotations

import inspect
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from harness.tools.schema import Tool, ToolCall, ToolResult

if TYPE_CHECKING:
    from harness.telemetry.recorder import Telemetry


class Dispatcher:
    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._telemetry = telemetry
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def tools_schema(self) -> list[dict[str, Any]]:
        return [tool.json_schema() for tool in self._tools.values()]

    async def dispatch(self, call: ToolCall) -> ToolResult:
        start = time.perf_counter()
        result = await self._dispatch_inner(call)
        duration_ms = (time.perf_counter() - start) * 1000.0

        if self._telemetry is not None:
            from harness.telemetry.events import ToolDispatched, jsonify

            await self._telemetry.emit(
                ToolDispatched(
                    tool_name=call.name,
                    call_id=call.id,
                    arguments=jsonify(call.arguments),
                    is_error=result.is_error,
                    duration_ms=duration_ms,
                )
            )
        return result

    async def _dispatch_inner(self, call: ToolCall) -> ToolResult:
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
