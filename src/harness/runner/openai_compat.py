"""OpenAICompatRunner — drives a tool-use loop against any OpenAI-compatible
chat completions endpoint.

Works against OpenAI itself plus the wider ecosystem of OSS servers that
speak the same protocol — vLLM, llama.cpp's server, Ollama, LM Studio,
Together, Groq, Anyscale, and so on. Construction takes an optional
`base_url` (defaults to OpenAI's) and an `api_key` (defaults to "none"
when a local `base_url` is supplied — local servers usually don't check
it).

Caveats (mirroring `AnthropicRunner`):
- `HookDecision.replacement` is ignored; only `block` is honoured.
- Stop reasons other than `stop`/`length`/`tool_calls` raise `RuntimeError`.
- `cache_control` markers from `harness.prompts` have no effect — the
  OpenAI Chat Completions API has no equivalent (caching is server-side
  / opaque on most providers).
- File blocks are inlined as text (`<file path=...>\n...\n</file>`); image
  parts and the rich content shape are deferred.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError as exc:
    raise ImportError(
        "harness.runner.openai_compat requires the openai SDK. "
        "Install with: pip install 'harness-engineering[openai-compat]'"
    ) from exc

from harness.agents.definition import SubAgent
from harness.hooks.events import PostToolUse, PreToolUse
from harness.hooks.runner import HookRunner
from harness.prompts.messages import ContentBlock, Message, text
from harness.runner.protocols import PrefixWatcherProtocol
from harness.tools.dispatcher import Dispatcher
from harness.tools.schema import ToolCall, ToolResult


def _serialize_tool_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict | list):
        return json.dumps(content, default=str)
    return str(content)


def _translate_tools(tools_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap harness `Tool.json_schema()` results in OpenAI's nested function-tool shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools_schema
    ]


def _translate_in(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate harness messages to OpenAI chat.completions format.

    OpenAI uses a flat list with role-based dispatch:
    - system → {"role": "system", "content": str}
    - user → {"role": "user", "content": str}
    - assistant → {"role": "assistant", "content": str, "tool_calls": [...]?}
    - tool result → {"role": "tool", "tool_call_id": str, "content": str}

    System messages and tool_use blocks live on different message types, so
    we split a harness Message into multiple API entries when necessary.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            joined = "\n\n".join(
                b.text or "" for b in msg.content if b.type == "text" and b.text
            )
            if joined:
                out.append({"role": "system", "content": joined})
            continue

        text_parts: list[str] = []
        tool_uses: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        for block in msg.content:
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type == "tool_use" and block.tool_use is not None:
                tool_uses.append(block.tool_use)
            elif block.type == "tool_result" and block.tool_result is not None:
                tool_results.append(block.tool_result)
            elif block.type == "file":
                text_parts.append(
                    f"<file path={block.path}>\n{block.text or ''}\n</file>"
                )

        if msg.role == "assistant":
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts),
            }
            if tool_uses:
                entry["tool_calls"] = [
                    {
                        "id": tu.id or "",
                        "type": "function",
                        "function": {
                            "name": tu.name,
                            "arguments": json.dumps(tu.arguments),
                        },
                    }
                    for tu in tool_uses
                ]
            out.append(entry)
        elif msg.role == "user":
            for tr in tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.id or "",
                        "content": _serialize_tool_content(tr.content),
                    }
                )
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})

    return out


def _translate_out(api_message: Any) -> Message:
    """Translate an OpenAI choice's `message` field into a harness assistant Message."""
    blocks: list[ContentBlock] = []

    content = getattr(api_message, "content", None)
    if content:
        blocks.append(ContentBlock(type="text", text=content))

    tool_calls = getattr(api_message, "tool_calls", None) or []
    for tc in tool_calls:
        if tc.type != "function":
            continue
        try:
            arguments = json.loads(tc.function.arguments)
        except (ValueError, AttributeError):
            arguments = {}
        blocks.append(
            ContentBlock(
                type="tool_use",
                tool_use=ToolCall(
                    name=tc.function.name,
                    arguments=arguments,
                    id=tc.id,
                ),
            )
        )

    return Message(role="assistant", content=blocks)


class OpenAICompatRunner:
    """`Runner` implementation for any OpenAI-compatible Chat Completions endpoint.

    Use with OpenAI directly, or point `base_url` at a local server:

        OpenAICompatRunner(dispatcher, hooks, base_url="http://localhost:11434/v1")

    will work against Ollama; substitute the relevant URL for vLLM, llama.cpp,
    LM Studio, Together, Groq, etc.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        hooks: HookRunner,
        *,
        client: AsyncOpenAI | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = 16_000,
        max_iterations: int = 10,
        prefix_watcher: PrefixWatcherProtocol | None = None,
        speculator: object | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.hooks = hooks
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            if api_key is not None:
                kwargs["api_key"] = api_key
            elif base_url is not None:
                # Local servers usually don't check the key but the SDK requires one.
                kwargs["api_key"] = "none"
            self._client = AsyncOpenAI(**kwargs)
        self._max_tokens = max_tokens
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
    ) -> dict[str, Any]:
        tools_schema = [
            t for t in self.dispatcher.tools_schema() if t["name"] in agent.allowed_tools
        ]
        kwargs: dict[str, Any] = {
            "model": agent.model,
            "messages": api_messages,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if tools_schema:
            kwargs["tools"] = _translate_tools(tools_schema)
        return kwargs

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        all_messages: list[Message] = []
        if agent.system_prompt:
            all_messages.append(text("system", agent.system_prompt))
        all_messages.extend(messages)

        api_messages = _translate_in(all_messages)
        request = self._build_request(agent, api_messages)

        for _ in range(self._max_iterations):
            if self._prefix_watcher is not None:
                await self._prefix_watcher.fingerprint(request)
            response = await self._client.chat.completions.create(**request)
            choice = response.choices[0]
            finish_reason = choice.finish_reason

            if finish_reason in ("stop", "length"):
                return _translate_out(choice.message)

            if finish_reason != "tool_calls":
                raise RuntimeError(
                    f"Unexpected finish_reason from model: {finish_reason!r}. "
                    "OpenAICompatRunner does not handle 'content_filter' or other "
                    "reasons yet."
                )

            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": choice.message.content or "",
            }
            tool_calls = list(choice.message.tool_calls or [])
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            request["messages"] = [*request["messages"], assistant_entry]

            tool_result_entries: list[dict[str, Any]] = []
            for tc in tool_calls:
                if tc.type != "function":
                    continue
                try:
                    arguments = json.loads(tc.function.arguments)
                except ValueError:
                    arguments = {}

                call = ToolCall(name=tc.function.name, arguments=arguments, id=tc.id)

                decisions = await self.hooks.emit(PreToolUse(call=call))
                blocked = next((d for d in decisions if d.block), None)
                if blocked is not None:
                    result = ToolResult(
                        id=tc.id,
                        content=blocked.reason or "blocked by hook",
                        is_error=True,
                    )
                else:
                    result = await self.dispatcher.dispatch(call)

                await self.hooks.emit(PostToolUse(call=call, result=result))

                tool_result_entries.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _serialize_tool_content(result.content),
                    }
                )

            request["messages"] = [*request["messages"], *tool_result_entries]

        raise RuntimeError(
            f"Tool-use loop exceeded max_iterations={self._max_iterations}. "
            "Increase the cap, constrain the tool surface, or shorten the conversation."
        )
