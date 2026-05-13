from __future__ import annotations

import logging
from collections.abc import Sequence

from harness.agents.definition import SubAgent
from harness.memory.record import SessionRecord
from harness.prompts.messages import Message

logger = logging.getLogger(__name__)


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
    def from_record(cls, record: SessionRecord, *, allow_tool_use: bool = True) -> ReplayRunner:
        """Build a runner from the assistant messages of a stored session.

        .. warning::

            **Tool calls in ``record`` are emitted as-is; they will NOT be
            re-dispatched against any ``Dispatcher``.** ``ReplayRunner`` is
            input-blind: it returns the recorded assistant messages verbatim,
            including any ``tool_use`` blocks, without invoking the
            corresponding tools. Downstream orchestrators that delegate
            tool dispatch to the runner will see the calls in the
            trajectory but no real tool execution will happen.

            Use ``ReplayRunner`` only for sessions where you do not need
            to re-execute tools — e.g. text-only replays, deterministic
            re-runs against fresh hooks / policies, or scaffolding tests.

            For "mutate-and-continue" with real tool dispatch, see
            :mod:`harness.replay.counterfactual`, which wires a real
            runner against a mutated prefix and produces a fresh
            continuation with actual tool execution.

        Args:
            record: Captured session to replay assistant messages from.
            allow_tool_use: When ``True`` (default), tool_use blocks in
                the record are passed through with a one-time WARNING log
                if any are detected. Preserved as the default for
                backward compatibility. Pass ``False`` to suppress the
                warning entirely when you have explicitly acknowledged
                the gap (e.g. in tests that intentionally round-trip
                tool-using records).
        """
        replies = [m for m in record.messages if m.role == "assistant"]
        if allow_tool_use and _has_tool_use(replies):
            logger.warning(
                "ReplayRunner.from_record: record contains assistant tool_use "
                "blocks; they will be emitted as canned replies but NOT "
                "re-dispatched against any Dispatcher. Use "
                "harness.replay.counterfactual for mutate-and-continue with "
                "real tool execution."
            )
        return cls(replies)

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


def _has_tool_use(messages: Sequence[Message]) -> bool:
    """Return True if any message contains a `tool_use` content block."""
    return any(block.type == "tool_use" for m in messages for block in m.content)
