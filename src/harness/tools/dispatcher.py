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
    """Routes `ToolCall`s to their registered handlers and produces `ToolResult`s.

    Validation errors (from the tool's input model) and handler exceptions are both
    converted to `ToolResult(is_error=True, content=str(exc))` — the model sees the
    error in its tool-result block and can self-correct. The dispatcher never raises
    from a handler bug.

    Exception discipline: this conversion contract is intentionally asymmetric with
    `HookRunner.emit`, which propagates handler exceptions instead. See
    `docs/contracts/user-code-execution.md` for how exceptions from hook handlers
    vs tool handlers vs sink emit are handled.
    """

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

    @property
    def tools(self) -> dict[str, Tool]:
        """Read-only snapshot of the registered tools, keyed by name.

        Returns a fresh dict on each call so callers can't mutate the
        registry. Use this when you need access to `Tool` metadata
        (e.g. `idempotent` for `harness.speculate`); use `tools_schema`
        for the JSON-schema export the runner sends to the model.
        """
        return dict(self._tools)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        # Open a span_scope for the dispatch so the emitted
        # `ToolDispatched` event correlates back to the orchestrator
        # turn that produced this call. The scope is conditional on
        # telemetry being configured — without it, no `span_id` would
        # ever propagate, so the scope is wasted work.
        if self._telemetry is None:
            start = time.perf_counter()
            return await self._dispatch_inner(call)

        async with self._telemetry.span_scope():
            start = time.perf_counter()
            result = await self._dispatch_inner(call)
            duration_ms = (time.perf_counter() - start) * 1000.0

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
