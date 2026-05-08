from __future__ import annotations

from collections.abc import Awaitable, Callable

from harness.agents.definition import SubAgent
from harness.prompts.messages import Message, text

Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]

_DEFAULT_SUMMARIZER = SubAgent(
    name="summarizer",
    system_prompt=(
        "You compress conversations into terse, factual summaries used as model "
        "context. Capture decisions, file paths, identifiers, and unresolved questions. "
        "Drop pleasantries. Output a single paragraph, no preamble, under 300 words."
    ),
    model="<set-by-runner>",
    allowed_tools=[],
)

_SUMMARY_PREFIX = "[Earlier conversation summary]\n"


def compact(
    messages: list[Message],
    *,
    keep_last: int = 8,
    keep_system: bool = True,
) -> list[Message]:
    """Trim a message history to the last `keep_last` non-system messages.

    Pure function — does not mutate the input. Always returns a new list.
    """
    system = [m for m in messages if m.role == "system"] if keep_system else []
    non_system = [m for m in messages if m.role != "system"]
    tail = non_system[-keep_last:] if keep_last > 0 else []
    return [*system, *tail]


async def summarize_compact(
    messages: list[Message],
    runner: Runner,
    *,
    keep_last: int = 8,
    keep_system: bool = True,
    summary_agent: SubAgent | None = None,
) -> list[Message]:
    """Compact by asking a model to summarize the dropped messages.

    Returns: kept system messages + a synthetic system-role summary block + the
    last `keep_last` non-system messages.

    The runner is vendor-neutral by design — it must satisfy the same shape
    that `Orchestrator` accepts. In practice callers pass an `AnthropicRunner`,
    but a fake or test runner works equally well.
    """
    system = [m for m in messages if m.role == "system"] if keep_system else []
    non_system = [m for m in messages if m.role != "system"]

    if len(non_system) <= keep_last:
        return [*system, *non_system]

    to_summarize = non_system[: len(non_system) - keep_last]
    tail = non_system[-keep_last:] if keep_last > 0 else []

    agent = summary_agent or _DEFAULT_SUMMARIZER
    prompt = text(
        "user",
        "Summarize the conversation that follows. "
        f"There are {len(to_summarize)} messages to compress.",
    )
    summary_message = await runner(agent, [*to_summarize, prompt])

    summary_text = "".join(
        block.text or "" for block in summary_message.content if block.type == "text"
    ).strip()
    if not summary_text:
        summary_text = "(empty summary)"

    summary_block = text("system", _SUMMARY_PREFIX + summary_text)
    return [*system, summary_block, *tail]
