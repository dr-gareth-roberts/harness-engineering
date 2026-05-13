"""Tiny fakes that mimic the shape of `openai.AsyncOpenAI` enough to drive
the tool-use loop in `harness.runner.openai_compat`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeOAFunction:
    name: str
    arguments: str  # JSON string, like the real SDK


@dataclass
class FakeOAToolCall:
    id: str
    function: FakeOAFunction
    type: str = "function"


@dataclass
class FakeOAMessage:
    content: str | None = None
    role: str = "assistant"
    tool_calls: list[FakeOAToolCall] | None = None


@dataclass
class FakeOAChoice:
    message: FakeOAMessage
    finish_reason: str
    index: int = 0


@dataclass
class FakeOAResponse:
    choices: list[FakeOAChoice]
    id: str = "chatcmpl_fake"
    usage: dict[str, int] = field(default_factory=dict)


class FakeOACompletions:
    def __init__(self, responses: list[FakeOAResponse], *, create_delay: float = 0.0) -> None:
        self._responses = list(responses)
        self._create_delay = create_delay
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeOAResponse:
        if self._create_delay > 0:
            import asyncio

            await asyncio.sleep(self._create_delay)
        if not self._responses:
            raise RuntimeError("FakeOACompletions: no canned responses left")
        self.requests.append(kwargs)
        return self._responses.pop(0)


class FakeOAChat:
    def __init__(self, responses: list[FakeOAResponse], *, create_delay: float = 0.0) -> None:
        self.completions = FakeOACompletions(responses, create_delay=create_delay)


class FakeAsyncOpenAI:
    """Stand-in for `openai.AsyncOpenAI` with scriptable chat.completions.

    `create_delay` lets tests inject a sleep into `chat.completions.create`
    to drive timeout-related behavior. Default 0 = no sleep.
    """

    def __init__(self, responses: list[FakeOAResponse], *, create_delay: float = 0.0) -> None:
        self.chat = FakeOAChat(responses, create_delay=create_delay)
