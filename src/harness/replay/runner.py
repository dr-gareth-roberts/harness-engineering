from __future__ import annotations

from collections.abc import Sequence

from harness.agents.definition import SubAgent
from harness.memory.record import SessionRecord
from harness.prompts.messages import Message


class ReplayMismatch(RuntimeError):
    """Raised when `ReplayRunner` is asked for a reply it doesn't have."""


class ReplayRunner:
    """Returns canned assistant messages in order. Satisfies the `Runner`
    protocol used by `Orchestrator`.

    Input-blind: ignores the agent and the message list. Useful for testing
    a harness end-to-end without making API calls and for replaying a
    captured `SessionRecord` against a fresh dispatcher / hooks / policies.

    Strict input verification (raise if the messages don't match the recorded
    inputs) is deferred to a follow-up.
    """

    def __init__(self, replies: Sequence[Message]) -> None:
        self._replies: list[Message] = list(replies)
        self._index = 0

    @classmethod
    def from_record(cls, record: SessionRecord) -> ReplayRunner:
        """Build a runner from the assistant messages of a stored session."""
        return cls([m for m in record.messages if m.role == "assistant"])

    @property
    def remaining(self) -> int:
        return len(self._replies) - self._index

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        if self._index >= len(self._replies):
            raise ReplayMismatch(
                f"replay exhausted after {self._index} replies — "
                f"no canned reply for turn {self._index + 1}"
            )
        reply = self._replies[self._index]
        self._index += 1
        return reply
