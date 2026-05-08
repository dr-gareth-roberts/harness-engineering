"""Counterfactual replay — mutate a recorded session and continue.

Run with: `uv run python examples/counterfactual.py`

`harness.replay.counterfactual.counterfactual` takes a recorded
`SessionRecord`, applies a structured mutation at a given turn index,
and asks a fresh runner to produce a new continuation from the
mutated prefix.

This example builds a tiny three-turn session by hand, applies a
`RewriteTurn(index=2, ...)` mutation that swaps the user's second
question, and lets a `CannedRunner` produce a different assistant
reply. Output prints the original timeline next to the
counterfactual, demonstrating that the prefix (turns 0-1) is
preserved and only the tail (turns 2+) is fresh.
"""

from __future__ import annotations

import asyncio

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.memory import SessionRecord
from harness.prompts import Message, text
from harness.replay.counterfactual import RewriteTurn, counterfactual
from harness.runner import CannedRunner
from harness.tools import Dispatcher


def _agent() -> SubAgent:
    return SubAgent(
        name="counterfactual-demo",
        system_prompt="be helpful",
        model="demo-model",
    )


def _build_original_session() -> SessionRecord:
    """A six-turn conversation about jokes, recorded earlier."""
    return SessionRecord(
        session_id="sess-original",
        agent=_agent(),
        messages=[
            text("user", "hi"),
            text("assistant", "hello, how can I help?"),
            text("user", "tell me a joke"),
            text("assistant", "why did the chicken cross the road?"),
            text("user", "why?"),
            text("assistant", "to get to the other side"),
        ],
    )


def _line(role: str, msg: Message) -> str:
    chunks = [b.text for b in msg.content if b.type == "text" and b.text]
    body = "".join(chunks) if chunks else f"<{msg.content[0].type}>"
    return f"{role:>9}: {body}"


async def main() -> int:
    transcript: list[str] = []

    # The original session — six turns, recorded earlier.
    original = _build_original_session()

    transcript.append("--- original session (recorded earlier) ---")
    for msg in original.messages:
        transcript.append(_line(msg.role, msg))

    # The mutation: at index 2 (the user's second turn), the user says
    # something different. Everything from index 3 onward is dropped;
    # the counterfactual runner produces a fresh continuation.
    mutation = RewriteTurn(
        index=2,
        new_message=text("user", "actually, tell me a fact about pelicans"),
    )

    # The counterfactual runner returns a different reply than the
    # original recording. counterfactual() drives this through a fresh
    # Orchestrator — the orchestrator passed in is treated as a
    # configuration bag (its dispatcher/hooks/telemetry are reused but
    # its own runner is ignored).
    runner = CannedRunner(replies=["pelicans can hold three gallons of water in their pouches"])
    config_orch = Orchestrator(Dispatcher(), HookRunner(), runner)

    result = await counterfactual(
        session=original,
        mutation=mutation,
        runner=runner,
        orchestrator=config_orch,
    )

    transcript.append("")
    transcript.append("--- counterfactual session (rewrite at turn 2) ---")
    for msg in result.messages:
        transcript.append(_line(msg.role, msg))

    # Verify what the documentation promises: the prefix (turns 0..1)
    # is identical to the original; turns 2+ differ.
    transcript.append("")
    transcript.append("--- divergence check ---")
    prefix_len = mutation.index  # turns [0, mutation.index) are preserved verbatim
    prefix_matches = result.messages[:prefix_len] == original.messages[:prefix_len]
    transcript.append(f"  prefix [0:{prefix_len}] preserved verbatim: {prefix_matches}")
    transcript.append(f"  original tail length: {len(original.messages) - prefix_len}")
    transcript.append(f"  counterfactual tail length: {len(result.messages) - prefix_len}")
    transcript.append(
        f"  session_id preserved across counterfactual: {result.session_id == original.session_id}"
    )

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
