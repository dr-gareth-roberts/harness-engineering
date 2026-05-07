from __future__ import annotations

from harness.prompts.messages import Message


def compact(
    messages: list[Message],
    *,
    keep_last: int = 8,
    keep_system: bool = True,
) -> list[Message]:
    """Trim a message history to the last `keep_last` non-system messages.

    Pure function — does not mutate the input. Always returns a new list.
    """
    # TODO: summarization strategy — drop the dropped messages into a summary block
    # instead of discarding them outright.
    system = [m for m in messages if m.role == "system"] if keep_system else []
    non_system = [m for m in messages if m.role != "system"]
    tail = non_system[-keep_last:] if keep_last > 0 else []
    return [*system, *tail]
