"""Lightweight fakes that mimic the shape of `anthropic.AsyncAnthropic` enough
for the tool-use loop in `harness.runner.anthropic` to exercise.

The real SDK pulls in httpx, returns Pydantic models, and changes shape between
releases. We mimic only the surface the runner actually touches:

    client.messages.stream(**kwargs)        # returns async context manager
        async with that as stream:
            await stream.get_final_message() # returns a message-shaped object

A canned response is a `FakeMessage(content=[...], stop_reason="...")`. Content
blocks are dataclasses with `.type` and the type-specific fields the runner
reads (`text`, `id`, `name`, `input`).
"""

from __future__ import annotations

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
class FakeMessage:
    content: list[Any]
    stop_reason: str
    id: str = "msg_fake"
    role: str = "assistant"
    usage: dict[str, int] = field(default_factory=dict)


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
