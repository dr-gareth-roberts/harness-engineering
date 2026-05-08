"""AnthropicRunner — drives an Anthropic Messages-API tool-use loop.

Implements the `Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]`
protocol from `harness.agents.orchestrator`. Reuses the existing `Dispatcher`
for tool execution and fires `PreToolUse` / `PostToolUse` events around each
dispatch — so `harness.policy` policies attached as hooks Just Work.

Caveats:
- `HookDecision.replacement` is ignored; only `block` is honoured.
- `cache_control` is rendered 1:1 from `ContentBlock.cache=True`. Anthropic
  caps the request at 4 cache breakpoints; the runner does not enforce that
  cap. Use `compact()` or trim the prefix before calling.
- `pause_turn` and `refusal` stop reasons surface as `RuntimeError`.
- File blocks are inlined as text (`<file path=...>\n...\n</file>`); Files API
  integration is deferred.
"""

from __future__ import annotations

import json
from typing import Any, Literal

try:
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError as exc:
    raise ImportError(
        "harness.runner.anthropic requires the anthropic SDK. "
        "Install with: pip install 'harness-engineering[anthropic]'"
    ) from exc

from harness.agents.definition import SubAgent
from harness.hooks.events import PostAssistantMessage, PostToolUse, PreToolUse
from harness.hooks.runner import HookRunner
from harness.prompts.messages import ContentBlock, Message
from harness.runner.protocols import PrefixWatcherProtocol
from harness.tools.dispatcher import Dispatcher
from harness.tools.schema import ToolCall, ToolResult

ThinkingMode = Literal["adaptive", "disabled"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


def _serialize_tool_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict | list):
        return json.dumps(content, default=str)
    return str(content)


def _translate_block_in(block: ContentBlock) -> dict[str, Any] | None:
    out: dict[str, Any]
    if block.type == "text":
        out = {"type": "text", "text": block.text or ""}
    elif block.type == "tool_use" and block.tool_use is not None:
        out = {
            "type": "tool_use",
            "id": block.tool_use.id,
            "name": block.tool_use.name,
            "input": block.tool_use.arguments,
        }
    elif block.type == "tool_result" and block.tool_result is not None:
        tr = block.tool_result
        out = {
            "type": "tool_result",
            "tool_use_id": tr.id,
            "content": _serialize_tool_content(tr.content),
            "is_error": tr.is_error,
        }
    elif block.type == "file":
        out = {
            "type": "text",
            "text": f"<file path={block.path}>\n{block.text or ''}\n</file>",
        }
    else:
        return None

    if block.cache:
        out["cache_control"] = {"type": "ephemeral"}
    return out


def _translate_in(messages: list[Message]) -> tuple[list[dict[str, Any]], str | None]:
    """Split harness messages into (api_messages, system_prefix).

    System messages are extracted out of the conversation flow because the
    Anthropic API takes `system` as a top-level param, not as a message role.
    Multiple system messages are joined with double newlines in encounter order.
    """
    system_parts: list[str] = []
    api_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            for block in msg.content:
                if block.type == "text" and block.text:
                    system_parts.append(block.text)
            continue

        api_blocks: list[dict[str, Any]] = []
        for block in msg.content:
            translated = _translate_block_in(block)
            if translated is not None:
                api_blocks.append(translated)
        if api_blocks:
            api_messages.append({"role": msg.role, "content": api_blocks})

    system = "\n\n".join(system_parts) if system_parts else None
    return api_messages, system


def _translate_out(api_message: anthropic.types.Message) -> Message:
    """Translate an Anthropic SDK Message into a harness assistant Message."""
    blocks: list[ContentBlock] = []
    for block in api_message.content:
        if block.type == "text":
            blocks.append(ContentBlock(type="text", text=block.text))
        elif block.type == "tool_use":
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name=block.name, arguments=dict(block.input), id=block.id),
                )
            )
        # thinking, redacted_thinking, server_tool_use, etc. are not surfaced in MVP.
    return Message(role="assistant", content=blocks)


class AnthropicRunner:
    """Closes the Anthropic Messages-API tool-use loop on behalf of an Orchestrator.

    Construct once per agent surface (a `Dispatcher` + `HookRunner`), then pass
    as the `runner` argument to `Orchestrator`.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        hooks: HookRunner,
        *,
        client: AsyncAnthropic | None = None,
        max_tokens: int = 16_000,
        thinking_mode: ThinkingMode = "adaptive",
        effort: Effort | None = None,
        max_iterations: int = 10,
        prefix_watcher: PrefixWatcherProtocol | None = None,
        speculator: object | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.hooks = hooks
        self._client = client if client is not None else AsyncAnthropic()
        self._max_tokens = max_tokens
        self._thinking_mode: ThinkingMode = thinking_mode
        self._effort: Effort | None = effort
        self._max_iterations = max_iterations
        self._prefix_watcher = prefix_watcher
        self._speculator = speculator
        # `speculator` is reserved for the speculative-execution feature
        # (`harness.speculate`); the runner accepts it now so adding the
        # feature later doesn't require a constructor signature change.

    def _build_request(
        self,
        agent: SubAgent,
        api_messages: list[dict[str, Any]],
        system_from_messages: str | None,
    ) -> dict[str, Any]:
        tools = [
            schema
            for schema in self.dispatcher.tools_schema()
            if schema["name"] in agent.allowed_tools
        ]

        system_parts = [s for s in (agent.system_prompt, system_from_messages) if s]

        kwargs: dict[str, Any] = {
            "model": agent.model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if tools:
            kwargs["tools"] = tools
        if self._thinking_mode == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
        if self._effort is not None:
            kwargs["output_config"] = {"effort": self._effort}
        return kwargs

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        api_messages, system = _translate_in(messages)
        request = self._build_request(agent, api_messages, system)

        for _ in range(self._max_iterations):
            if self._prefix_watcher is not None:
                await self._prefix_watcher.fingerprint(request)
            async with self._client.messages.stream(**request) as stream:
                response = await stream.get_final_message()

            assistant_message = _translate_out(response)
            await self.hooks.emit(PostAssistantMessage(message=assistant_message))

            if response.stop_reason in ("end_turn", "stop_sequence"):
                return assistant_message

            if response.stop_reason != "tool_use":
                raise RuntimeError(
                    f"Unexpected stop_reason from model: {response.stop_reason!r}. "
                    "AnthropicRunner does not handle 'pause_turn' or 'refusal' yet."
                )

            request["messages"] = [
                *request["messages"],
                {"role": "assistant", "content": response.content},
            ]

            tool_result_blocks: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                call = ToolCall(name=block.name, arguments=dict(block.input), id=block.id)

                decisions = await self.hooks.emit(PreToolUse(call=call))
                blocked = next((d for d in decisions if d.block), None)
                if blocked is not None:
                    result = ToolResult(
                        id=block.id,
                        content=blocked.reason or "blocked by hook",
                        is_error=True,
                    )
                else:
                    result = await self.dispatcher.dispatch(call)

                await self.hooks.emit(PostToolUse(call=call, result=result))

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _serialize_tool_content(result.content),
                        "is_error": result.is_error,
                    }
                )

            request["messages"] = [
                *request["messages"],
                {"role": "user", "content": tool_result_blocks},
            ]

        raise RuntimeError(
            f"Tool-use loop exceeded max_iterations={self._max_iterations}. "
            "Increase the cap, constrain the tool surface, or shorten the conversation."
        )
