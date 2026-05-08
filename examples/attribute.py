"""Causal provenance via leave-one-out ablation (`harness.attribute`).

Run with: `uv run python examples/attribute.py`

`attribute(session, target_message_index, runner, agent, ...)` re-runs
the runner once per input chunk with that chunk removed and ranks
chunks by how much the response diverged from the original target.
Higher divergence = higher causal influence.

This example builds a 4-block user message where exactly one block
carries the password the assistant ends up quoting back. The runner is
deterministic: it scans the input messages for the password substring
and returns the same fixed reply when present, or a different fixed
reply when absent. Because Jaccard token-overlap similarity sees the
"with-password" reply as identical to the target (score 0.0) and the
"no-password" reply as different (score > 0.0), the ablation cleanly
identifies the password-carrying block as the most influential.

We also demonstrate `estimate_only=True` first — useful when N+1 runner
calls would be expensive and you want a chunk count before committing.
"""

from __future__ import annotations

import asyncio

from harness.agents import SubAgent
from harness.attribute import JaccardSimilarity, attribute
from harness.memory import SessionRecord
from harness.prompts import Message, text
from harness.prompts.messages import ContentBlock

# Pin both the runner-with-password reply and the recorded target to
# the same literal so Jaccard sees them as identical when the password
# block is *not* ablated. Any wording drift here corrupts the ranking.
_PASSWORD = "rosebud"
_REPLY_WITH = f"the password is {_PASSWORD}"
_REPLY_WITHOUT = "(no password)"


def _agent() -> SubAgent:
    return SubAgent(name="attribute-demo", system_prompt="", model="demo-model")


def _build_session() -> SessionRecord:
    """A single user message with four text blocks plus the assistant's reply.

    At `granularity="block"` we get four chunks (one per `ContentBlock`
    in the prefix message). The third block (index 2) carries the
    password that the assistant's reply quotes.
    """
    user_message = Message(
        role="user",
        content=[
            ContentBlock(type="text", text="hi there, just chatting"),
            ContentBlock(type="text", text="some unrelated context about cats"),
            ContentBlock(type="text", text=f"the password is {_PASSWORD}"),
            ContentBlock(type="text", text="anything else you can tell me?"),
        ],
    )
    assistant_reply = text("assistant", _REPLY_WITH)

    return SessionRecord(
        session_id="attribute-demo",
        agent=_agent(),
        messages=[user_message, assistant_reply],
    )


async def password_quoting_runner(_agent: SubAgent, messages: list[Message]) -> Message:
    """Returns the same reply iff the password appears anywhere in `messages`."""
    for message in messages:
        for block in message.content:
            if block.type == "text" and block.text and _PASSWORD in block.text:
                return text("assistant", _REPLY_WITH)
    return text("assistant", _REPLY_WITHOUT)


async def main() -> int:
    transcript: list[str] = []
    transcript.append("--- attribute (leave-one-out ablation) ---")

    record = _build_session()

    # Step 1: estimate_only — count the chunks (and therefore the
    # would-be runner calls) without invoking the runner once.
    estimate = await attribute(
        record,
        target_message_index=-1,
        runner=password_quoting_runner,
        agent=_agent(),
        granularity="block",
        similarity=JaccardSimilarity(),
        estimate_only=True,
    )
    transcript.append(
        f"  estimate_only: {estimate.estimated_calls} chunk(s) "
        f"(would invoke runner that many times)"
    )

    # Step 2: real ablation. One runner call per chunk; Jaccard scores
    # `1 - similarity(target, ablated_reply)`.
    result = await attribute(
        record,
        target_message_index=-1,
        runner=password_quoting_runner,
        agent=_agent(),
        granularity="block",
        similarity=JaccardSimilarity(),
    )
    transcript.append(f"  actual_calls: {result.actual_calls}; target={result.target_response!r}")

    transcript.append("  top 3 most-influential chunks:")
    for rank, chunk in enumerate(result.top_k(3), start=1):
        transcript.append(
            f"    [{rank}] message={chunk.message_index} block={chunk.block_index} "
            f"score={chunk.score:.3f} preview={chunk.preview!r}"
        )

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
