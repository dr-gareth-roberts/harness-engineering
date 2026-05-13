"""Lightweight fakes that mimic the shape of `anthropic.AsyncAnthropic` enough
for the tool-use loop in `harness.runner.anthropic` to exercise.

The real SDK pulls in httpx, returns Pydantic models, and changes shape between
releases. We mimic only the surface the runner actually touches:

    client.messages.stream(**kwargs)        # returns async context manager
        async with that as stream:
            async for event in stream:       # iterate streaming events
                ...
            msg = await stream.get_final_message()  # accumulated message

A canned response is a `FakeMessage(content=[...], stop_reason="...")`. Content
blocks are dataclasses with `.type` and the type-specific fields the runner
reads (`text`, `id`, `name`, `input`).

Streaming events: by default, `_FakeStream.__aiter__` auto-derives a
`FakeContentBlockStopEvent` per block in `response.content`, so existing
tests that don't care about event order still drive the event-aware
runner correctly. Tests that need to script a specific event sequence
(out-of-order blocks, partial deltas, etc.) set `FakeMessage.events` to
a list explicitly. `get_final_message` returns the same `FakeMessage`
whether the stream was iterated or not, mirroring the real SDK's
`until_done()` no-op-after-consumption behavior.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeContentBlockStopEvent:
    """Mimics `anthropic.lib.streaming.ParsedContentBlockStopEvent` enough
    for the runner to inspect `event.type` and `event.content_block`.
    """

    index: int
    content_block: Any
    type: str = "content_block_stop"


@dataclass
class FakeTextEvent:
    """Mimics the SDK's `TextEvent` (one of the high-level events the
    SDK accumulator yields for text deltas). The runner reads
    `event.type == "text"` and `event.text` to build a `TextDelta`.
    """

    text: str
    type: str = "text"


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str
    id: str = "msg_fake"
    role: str = "assistant"
    usage: dict[str, int] = field(default_factory=dict)
    # When `events` is None, `_FakeStream.__aiter__` auto-derives one
    # `content_block_stop` per `content` entry — convenient default so
    # existing tests don't need to script events. Set explicitly to drive
    # specific event sequences (e.g. text-then-tool, multiple tools,
    # zero-event streams).
    events: list[Any] | None = None


class _FakeStream:
    def __init__(
        self,
        response: FakeMessage,
        *,
        enter_delay: float = 0.0,
        exit_delay: float = 0.0,
    ) -> None:
        self._response = response
        self._enter_delay = enter_delay
        self._exit_delay = exit_delay
        self._event_iter: AsyncIterator[Any] | None = None

    async def __aenter__(self) -> _FakeStream:
        if self._enter_delay > 0:
            import asyncio

            await asyncio.sleep(self._enter_delay)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._exit_delay > 0:
            import asyncio

            await asyncio.sleep(self._exit_delay)
        return None

    def __aiter__(self) -> _FakeStream:
        # Return self so callers (including `_TimeoutStream` which calls
        # `__anext__` directly on the inner stream) can drive iteration.
        # The real SDK stream behaves this way — `__aiter__` returns the
        # accumulator object itself, not a fresh async generator.
        self._ensure_iter()
        return self

    async def __anext__(self) -> Any:
        self._ensure_iter()
        assert self._event_iter is not None  # set by _ensure_iter
        return await self._event_iter.__anext__()

    def _ensure_iter(self) -> None:
        if self._event_iter is not None:
            return
        events = self._response.events
        if events is None:
            events = [
                FakeContentBlockStopEvent(index=i, content_block=block)
                for i, block in enumerate(self._response.content)
            ]

        async def _gen() -> AsyncIterator[Any]:
            for event in events:
                yield event

        self._event_iter = _gen()

    async def get_final_message(self) -> FakeMessage:
        return self._response


class FakeMessages:
    def __init__(
        self,
        responses: list[FakeMessage],
        *,
        enter_delay: float = 0.0,
        exit_delay: float = 0.0,
    ) -> None:
        self._responses = list(responses)
        self._enter_delay = enter_delay
        self._exit_delay = exit_delay
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStream:
        if not self._responses:
            raise RuntimeError("FakeMessages: no canned responses left for stream() call")
        self.requests.append(kwargs)
        return _FakeStream(
            self._responses.pop(0),
            enter_delay=self._enter_delay,
            exit_delay=self._exit_delay,
        )


class FakeAsyncAnthropic:
    """Stand-in for `anthropic.AsyncAnthropic` with a scriptable `messages.stream`.

    `enter_delay` lets tests inject a sleep into `messages.stream(...)`'s
    `__aenter__` to drive enter-time timeout behavior. `exit_delay` does
    the same for `__aexit__` so tests can pin teardown-timeout paths.
    Both default to 0 = no sleep, matching the prior tests.
    """

    def __init__(
        self,
        responses: list[FakeMessage],
        *,
        enter_delay: float = 0.0,
        exit_delay: float = 0.0,
    ) -> None:
        self.messages = FakeMessages(
            responses,
            enter_delay=enter_delay,
            exit_delay=exit_delay,
        )
