"""AnthropicRunner — drives an Anthropic Messages-API tool-use loop.

Implements the `Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]`
protocol from `harness.agents.orchestrator`. Reuses the existing `Dispatcher`
for tool execution and fires `PreToolUse` / `PostToolUse` events around each
dispatch — so `harness.policy` policies attached as hooks Just Work.

Caveats:
- `HookDecision` honors `block` (short-circuit to is_error result) and
  `replacement` (PreToolUse: skip dispatch, use supplied result;
  PostToolUse: rewrite the dispatched result before sending back).
- `cache_control` is rendered 1:1 from `ContentBlock.cache=True`. The runner
  enforces Anthropic's 4-cache-breakpoint cap client-side: a request with
  more raises `CacheBreakpointLimitExceeded` *before* the SDK call,
  surfacing the failure at the harness boundary instead of the API boundary.
- `pause_turn` and `refusal` stop reasons fire `PauseTurn` / `Refusal`
  hook events and the partial assistant message is returned. Callers
  can register hooks to react (re-invoke on pause, log on refusal).
- File blocks are inlined as text (`<file path=...>\n...\n</file>`); Files API
  integration is deferred.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
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
from harness.hooks.events import (
    PauseTurn,
    PostAssistantMessage,
    PostToolUse,
    PreToolUse,
    Refusal,
)
from harness.hooks.runner import HookRunner
from harness.prompts.messages import ContentBlock, Message
from harness.runner.protocols import PrefixWatcherProtocol, SpeculatorProtocol
from harness.streaming import (
    MessageEnd,
    StreamEvent,
    TextDelta,
    ToolUseEnd,
    ToolUseStart,
)
from harness.tools.dispatcher import Dispatcher
from harness.tools.schema import ToolCall, ToolResult

ThinkingMode = Literal["adaptive", "disabled"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


# Anthropic's API caps each request at this many `cache_control` markers
# across all messages + system blocks. Going over yields a 400 from the
# API; we surface it client-side as a typed exception instead.
_CACHE_BREAKPOINT_LIMIT = 4


class CacheBreakpointLimitExceeded(ValueError):
    """The translated request carries more than 4 `cache_control` markers.

    The Anthropic API rejects such requests; this exception surfaces the
    failure at the harness boundary so the caller gets a clear, typed
    error instead of an opaque 400. The message names the count we
    saw and points at `harness.prompts.compact` (or trimming
    `ContentBlock.cache=True` markers) as the resolution.
    """


def _count_cache_breakpoints(request: dict[str, Any]) -> int:
    """Count `cache_control` markers in the translated request shape.

    Walks every message's content list looking for blocks with a
    `cache_control` key. The runner's `_translate_block_in` is the only
    place these markers are emitted, so the count is exact.
    """
    total = 0
    for msg in request.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "cache_control" in block:
                total += 1
    return total


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
    elif block.type == "image" and block.image is not None:
        # Anthropic vision shape (Wave 12 #7): {"type":"image",
        # "source":{"type":"base64"|"url","media_type":...,"data":...}}.
        # Both source modes use the same envelope; the SDK validates
        # `media_type` against the supported list.
        out = {
            "type": "image",
            "source": {
                "type": block.image.source,
                "media_type": block.image.media_type,
                "data": block.image.data,
            },
        }
    elif block.type == "file" and block.file_id is not None:
        # Wave 12 #8 — Anthropic Files API integration. Reference the
        # uploaded file by id; the API resolves the content server-side.
        out = {
            "type": "document",
            "source": {"type": "file", "file_id": block.file_id},
        }
    elif block.type == "file":
        # Path-based fallback: inline the file's text contents as a
        # text block, the historical pre-Wave-12 behavior.
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
        speculator: SpeculatorProtocol | None = None,
        timeout_s: float | None = None,
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
        # Per-iteration timeout. None = no timeout (default; matches the
        # SDK's own behavior). When set, the entire stream-and-iterate
        # phase per iteration is wrapped in `asyncio.wait_for`. Note this
        # is per *iteration*, not per *call* — a 5s timeout on a tool-use
        # loop with 3 iterations gives the model up to 15s wall-clock
        # total. Retry/backoff is intentionally deferred (Wave 10 +
        # streaming + speculator state make a clean retry semantic
        # non-trivial; see docs/plan.md).
        self._timeout_s = timeout_s

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

        # Running history that grows with each iteration. The speculator's
        # `begin` sees this so its predictions reflect in-loop turns the
        # caller never sees (intermediate assistant tool_use messages, the
        # synthesized tool_result message we feed back to the model, etc.).
        running_history: list[Message] = list(messages)

        for _ in range(self._max_iterations):
            # Surface the cache-breakpoint cap before any IO so the caller
            # gets a clear typed error instead of an opaque API 400. Per
            # iteration because tool_results we feed back may themselves
            # carry `cache_control`.
            count = _count_cache_breakpoints(request)
            if count > _CACHE_BREAKPOINT_LIMIT:
                raise CacheBreakpointLimitExceeded(
                    f"Anthropic caps cache breakpoints at "
                    f"{_CACHE_BREAKPOINT_LIMIT}; got {count}. "
                    "Remove some `ContentBlock.cache=True` markers, or "
                    "use `harness.prompts.compact` to trim the prefix."
                )

            if self._prefix_watcher is not None:
                await self._prefix_watcher.fingerprint(request)

            if self._speculator is not None:
                await self._speculator.begin(
                    history=running_history,
                    agent=agent,
                    dispatcher=self.dispatcher,
                    hooks=self.hooks,
                )

            try:
                async with self._stream_with_timeout(request) as stream:
                    # Iterate the stream's high-level events as they
                    # arrive. The only one we react to is
                    # `content_block_stop` for `tool_use` blocks —
                    # surface them to the speculator so it can mark
                    # matching pending speculations as observed before
                    # the model finishes generating. After the loop,
                    # `get_final_message` returns the accumulated
                    # message; the SDK's `until_done()` is a no-op once
                    # the stream is consumed, mirrored by the test fake.
                    async for event in stream:
                        if self._speculator is None:
                            continue
                        if getattr(event, "type", None) != "content_block_stop":
                            continue
                        block = getattr(event, "content_block", None)
                        if block is None or getattr(block, "type", None) != "tool_use":
                            continue
                        call = ToolCall(
                            name=block.name,
                            arguments=dict(block.input),
                            id=block.id,
                        )
                        await self._speculator.observe(call)
                    response = await stream.get_final_message()

                # Stream is now fully arrived. Cancel any pending
                # speculations the model didn't claim, before we move on
                # to dispatching its emitted tool_use blocks. This frees
                # the handler runtime that would otherwise burn through
                # the dispatch phase until `end()` finally cancels it.
                # `end()` (in the finally block) still acts as a safety
                # net for anything still pending.
                if self._speculator is not None:
                    await self._speculator.cancel_unobserved()

                assistant_message = _translate_out(response)
                running_history.append(assistant_message)
                await self.hooks.emit(PostAssistantMessage(message=assistant_message))

                if response.stop_reason in ("end_turn", "stop_sequence"):
                    return assistant_message

                if response.stop_reason == "pause_turn":
                    # Server-side pause (typically a long-running tool
                    # exceeded the per-turn budget). Surface as an event
                    # and return the partial assistant message; the caller
                    # can re-invoke with this message appended to resume.
                    await self.hooks.emit(PauseTurn(message=assistant_message, reason="pause_turn"))
                    return assistant_message

                if response.stop_reason == "refusal":
                    # The model refused. Surface as an event and return
                    # the refusal-only assistant message; the caller can
                    # inspect blocks and decide what to do.
                    await self.hooks.emit(Refusal(message=assistant_message))
                    return assistant_message

                if response.stop_reason != "tool_use":
                    raise RuntimeError(
                        f"Unexpected stop_reason from model: {response.stop_reason!r}. "
                        "Known reasons handled: end_turn, stop_sequence, tool_use, "
                        "pause_turn, refusal."
                    )

                request["messages"] = [
                    *request["messages"],
                    {"role": "assistant", "content": response.content},
                ]

                tool_result_blocks: list[dict[str, Any]] = []
                synthesized_result_blocks: list[ContentBlock] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    call = ToolCall(name=block.name, arguments=dict(block.input), id=block.id)

                    # Speculator's hit-check runs BEFORE the runner's own
                    # hook + dispatch cycle. On hit, the speculator has
                    # already fired PreToolUse/PostToolUse around its own
                    # dispatch, so the runner skips both for this call.
                    speculative_result: ToolResult | None = None
                    if self._speculator is not None:
                        speculative_result = await self._speculator.try_resolve(call)

                    if speculative_result is not None:
                        result = speculative_result
                    else:
                        pre_decisions = await self.hooks.emit(PreToolUse(call=call))
                        # PreToolUse hook decisions: `block` short-circuits to
                        # an is_error result; `replacement=ToolResult(...)`
                        # short-circuits dispatch with the supplied result
                        # (id patched to the model's call id). First matching
                        # decision wins.
                        blocked = next((d for d in pre_decisions if d.block), None)
                        replaced = next(
                            (d for d in pre_decisions if isinstance(d.replacement, ToolResult)),
                            None,
                        )
                        if blocked is not None:
                            result = ToolResult(
                                id=block.id,
                                content=blocked.reason or "blocked by hook",
                                is_error=True,
                            )
                        elif replaced is not None:
                            assert isinstance(replaced.replacement, ToolResult)
                            result = ToolResult(
                                id=block.id,
                                content=replaced.replacement.content,
                                is_error=replaced.replacement.is_error,
                            )
                        else:
                            result = await self.dispatcher.dispatch(call)

                        post_decisions = await self.hooks.emit(
                            PostToolUse(call=call, result=result)
                        )
                        # PostToolUse can rewrite the result before it goes
                        # back to the model — typical use is sanitization
                        # (redact secrets in the result, normalize errors).
                        post_replacement = next(
                            (d for d in post_decisions if isinstance(d.replacement, ToolResult)),
                            None,
                        )
                        if post_replacement is not None:
                            assert isinstance(post_replacement.replacement, ToolResult)
                            result = ToolResult(
                                id=block.id,
                                content=post_replacement.replacement.content,
                                is_error=post_replacement.replacement.is_error,
                            )

                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _serialize_tool_content(result.content),
                            "is_error": result.is_error,
                        }
                    )
                    synthesized_result_blocks.append(
                        ContentBlock(type="tool_result", tool_result=result)
                    )

                request["messages"] = [
                    *request["messages"],
                    {"role": "user", "content": tool_result_blocks},
                ]
                # Mirror into running_history so the next iteration's
                # speculator.begin sees the tool_results we just sent back.
                running_history.append(Message(role="user", content=synthesized_result_blocks))
            finally:
                if self._speculator is not None:
                    await self._speculator.end()

        raise RuntimeError(
            f"Tool-use loop exceeded max_iterations={self._max_iterations}. "
            "Increase the cap, constrain the tool surface, or shorten the conversation."
        )

    async def run_stream(
        self,
        agent: SubAgent,
        messages: list[Message],
    ) -> AsyncIterator[StreamEvent]:
        """Run the same tool-use loop as `__call__` but yield streaming
        events as the model generates.

        Per the wave-13a advisor review, this is a parallel method to
        `__call__` rather than a refactor — `__call__`'s 150 lines of
        tool-use-loop / speculator / hook / cache-cap / timeout /
        replacement / pause-refusal logic is too well tested to risk
        moving wholesale into a generator. The duplication is
        intentional and bounded; refactoring to share the loop body is
        a follow-up wave once both paths are proven.

        Yield order, per iteration:
        - `TextDelta(text=...)` — once per SDK `text` delta event.
        - `ToolUseStart(call=...)` — once per `content_block_stop` for
          a tool_use block, *after* speculator.observe but *before*
          the runner's hook + dispatch cycle.
        - `ToolUseEnd(call=..., result=...)` — once per dispatched
          tool call, after the result is finalized.

        At the very end (`end_turn` / `stop_sequence` / `pause_turn` /
        `refusal`), yields exactly one `MessageEnd(message=...)` and
        returns. `MessageEnd.message` matches what `__call__` would
        have returned.
        """
        api_messages, system = _translate_in(messages)
        request = self._build_request(agent, api_messages, system)

        running_history: list[Message] = list(messages)

        for _ in range(self._max_iterations):
            count = _count_cache_breakpoints(request)
            if count > _CACHE_BREAKPOINT_LIMIT:
                raise CacheBreakpointLimitExceeded(
                    f"Anthropic caps cache breakpoints at "
                    f"{_CACHE_BREAKPOINT_LIMIT}; got {count}. "
                    "Remove some `ContentBlock.cache=True` markers, or "
                    "use `harness.prompts.compact` to trim the prefix."
                )

            if self._prefix_watcher is not None:
                await self._prefix_watcher.fingerprint(request)

            if self._speculator is not None:
                await self._speculator.begin(
                    history=running_history,
                    agent=agent,
                    dispatcher=self.dispatcher,
                    hooks=self.hooks,
                )

            try:
                async with self._stream_with_timeout(request) as stream:
                    async for event in stream:
                        # Text deltas: yield as TextDelta.
                        event_type = getattr(event, "type", None)
                        if event_type == "text":
                            text_value = getattr(event, "text", None)
                            if text_value:
                                yield TextDelta(text=text_value)
                            continue

                        # tool_use block done: surface to speculator AND
                        # yield ToolUseStart so the caller can react
                        # before the runner dispatches. The dispatch
                        # itself happens after `get_final_message`.
                        if (
                            event_type == "content_block_stop"
                            and getattr(getattr(event, "content_block", None), "type", None)
                            == "tool_use"
                        ):
                            block = event.content_block
                            call = ToolCall(
                                name=block.name,
                                arguments=dict(block.input),
                                id=block.id,
                            )
                            if self._speculator is not None:
                                await self._speculator.observe(call)
                            yield ToolUseStart(call=call)
                    response = await stream.get_final_message()

                if self._speculator is not None:
                    await self._speculator.cancel_unobserved()

                assistant_message = _translate_out(response)
                running_history.append(assistant_message)
                await self.hooks.emit(PostAssistantMessage(message=assistant_message))

                # Terminal stop reasons: emit MessageEnd and return.
                if response.stop_reason in ("end_turn", "stop_sequence"):
                    yield MessageEnd(message=assistant_message)
                    return

                if response.stop_reason == "pause_turn":
                    await self.hooks.emit(PauseTurn(message=assistant_message, reason="pause_turn"))
                    yield MessageEnd(message=assistant_message)
                    return

                if response.stop_reason == "refusal":
                    await self.hooks.emit(Refusal(message=assistant_message))
                    yield MessageEnd(message=assistant_message)
                    return

                if response.stop_reason != "tool_use":
                    raise RuntimeError(
                        f"Unexpected stop_reason from model: {response.stop_reason!r}. "
                        "Known reasons handled: end_turn, stop_sequence, tool_use, "
                        "pause_turn, refusal."
                    )

                request["messages"] = [
                    *request["messages"],
                    {"role": "assistant", "content": response.content},
                ]

                tool_result_blocks: list[dict[str, Any]] = []
                synthesized_result_blocks: list[ContentBlock] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    call = ToolCall(name=block.name, arguments=dict(block.input), id=block.id)

                    speculative_result: ToolResult | None = None
                    if self._speculator is not None:
                        speculative_result = await self._speculator.try_resolve(call)

                    if speculative_result is not None:
                        result = speculative_result
                    else:
                        pre_decisions = await self.hooks.emit(PreToolUse(call=call))
                        blocked = next((d for d in pre_decisions if d.block), None)
                        replaced = next(
                            (d for d in pre_decisions if isinstance(d.replacement, ToolResult)),
                            None,
                        )
                        if blocked is not None:
                            result = ToolResult(
                                id=block.id,
                                content=blocked.reason or "blocked by hook",
                                is_error=True,
                            )
                        elif replaced is not None:
                            assert isinstance(replaced.replacement, ToolResult)
                            result = ToolResult(
                                id=block.id,
                                content=replaced.replacement.content,
                                is_error=replaced.replacement.is_error,
                            )
                        else:
                            result = await self.dispatcher.dispatch(call)

                        post_decisions = await self.hooks.emit(
                            PostToolUse(call=call, result=result)
                        )
                        post_replacement = next(
                            (d for d in post_decisions if isinstance(d.replacement, ToolResult)),
                            None,
                        )
                        if post_replacement is not None:
                            assert isinstance(post_replacement.replacement, ToolResult)
                            result = ToolResult(
                                id=block.id,
                                content=post_replacement.replacement.content,
                                is_error=post_replacement.replacement.is_error,
                            )

                    yield ToolUseEnd(call=call, result=result)

                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _serialize_tool_content(result.content),
                            "is_error": result.is_error,
                        }
                    )
                    synthesized_result_blocks.append(
                        ContentBlock(type="tool_result", tool_result=result)
                    )

                request["messages"] = [
                    *request["messages"],
                    {"role": "user", "content": tool_result_blocks},
                ]
                running_history.append(Message(role="user", content=synthesized_result_blocks))
            finally:
                if self._speculator is not None:
                    await self._speculator.end()

        raise RuntimeError(
            f"Tool-use loop exceeded max_iterations={self._max_iterations}. "
            "Increase the cap, constrain the tool surface, or shorten the conversation."
        )

    def _stream_with_timeout(self, request: dict[str, Any]) -> Any:
        """Return a stream context manager, optionally wrapped in a timeout.

        When `timeout_s` is None we hand back the SDK's own context manager
        unchanged. When set, we wrap entry + exit + iteration in
        `asyncio.wait_for` via a small adapter. Keeping the wrap in a
        helper rather than inlining it keeps the iteration loop body
        readable.
        """
        ctx = self._client.messages.stream(**request)
        if self._timeout_s is None:
            return ctx
        return _TimeoutStreamCtx(ctx, self._timeout_s)


class _TimeoutStreamCtx:
    """Wraps an Anthropic stream context manager with a per-iteration timeout.

    `asyncio.wait_for` enforces the deadline on every awaited operation
    (`__aenter__`, every `async for` step inside, `__aexit__`). If the
    deadline expires the SDK call is cancelled and `TimeoutError` bubbles
    up so the caller can decide whether to retry / give up.
    """

    def __init__(self, inner: Any, timeout_s: float) -> None:
        self._inner = inner
        self._timeout_s = timeout_s
        self._stream: Any | None = None

    async def __aenter__(self) -> Any:
        self._stream = await asyncio.wait_for(self._inner.__aenter__(), timeout=self._timeout_s)
        return _TimeoutStream(self._stream, self._timeout_s)

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            return await asyncio.wait_for(
                self._inner.__aexit__(exc_type, exc, tb), timeout=self._timeout_s
            )
        except TimeoutError:
            # The inner stream is being torn down; swallow the timeout
            # rather than masking the original exception (if any).
            return False


class _TimeoutStream:
    """Iterator wrapper that applies the timeout to each `__anext__` call."""

    def __init__(self, inner: Any, timeout_s: float) -> None:
        self._inner = inner
        self._timeout_s = timeout_s

    def __aiter__(self) -> _TimeoutStream:
        return self

    async def __anext__(self) -> Any:
        return await asyncio.wait_for(self._inner.__anext__(), timeout=self._timeout_s)

    async def get_final_message(self) -> Any:
        return await asyncio.wait_for(self._inner.get_final_message(), timeout=self._timeout_s)
