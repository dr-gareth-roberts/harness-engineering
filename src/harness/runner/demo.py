"""Vendor-neutral demo runners — useful for tests, smoke checks, and examples.

These runners satisfy the same `Runner` protocol that `AnthropicRunner` and
`OpenAICompatRunner` do, but ship with the base install (no extras). Treat
them as building blocks: real applications either use a vendor runner or
write their own.
"""

from __future__ import annotations

from harness.agents.definition import SubAgent
from harness.prompts.messages import ContentBlock, Message, text


class EchoRunner:
    """Returns the last user text back as the assistant turn.

    No model, no SDK, no network. Useful for wiring up tests of the
    surrounding pieces (orchestrator + hooks + telemetry + memory) without
    an API key, and as the simplest possible demonstration that the
    `Runner` protocol is vendor-neutral.

    If the conversation has no user text, the runner returns an empty
    assistant message (an empty text block).
    """

    def __init__(self, *, prefix: str = "") -> None:
        self._prefix = prefix

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        last_user_text = ""
        for msg in reversed(messages):
            if msg.role != "user":
                continue
            collected = "".join(
                b.text or "" for b in msg.content if b.type == "text" and b.text
            )
            if collected:
                last_user_text = collected
                break

        return Message(
            role="assistant",
            content=[ContentBlock(type="text", text=self._prefix + last_user_text)],
        )


class CannedRunner:
    """Returns canned text replies in order.

    Conceptually a tiny cousin of `harness.replay.ReplayRunner` — same shape,
    but takes plain strings instead of full `Message` objects, so callers
    can write quick tests without constructing `Message` instances by hand.
    Re-uses the same canned-reply-exhaustion error type as ReplayRunner via
    composition for consistency.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        if self._index >= len(self._replies):
            raise RuntimeError(
                f"CannedRunner exhausted after {self._index} replies — "
                f"no canned reply for turn {self._index + 1}"
            )
        reply = self._replies[self._index]
        self._index += 1
        return text("assistant", reply)
