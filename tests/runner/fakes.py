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
    def __init__(self, response: FakeMessage) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        events = self._response.events
        if events is None:
            events = [
                FakeContentBlockStopEvent(index=i, content_block=block)
                for i, block in enumerate(self._response.content)
            ]
        for event in events:
            yield event

    async def get_final_message(self) -> FakeMessage:
        return self._response


class FakeMessages:
    def __init__(self, responses: list[FakeMessage]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStream:
        if not self._responses:
            raise RuntimeError("FakeMessages: no canned responses left for stream() call")
        self.requests.append(kwargs)
        return _FakeStream(self._responses.pop(0))


class FakeAsyncAnthropic:
    """Stand-in for `anthropic.AsyncAnthropic` with a scriptable `messages.stream`."""

    def __init__(self, responses: list[FakeMessage]) -> None:
        self.messages = FakeMessages(responses)
