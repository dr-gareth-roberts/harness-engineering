"""Counterfactual replay — mutate a recorded `SessionRecord` and continue.

Take a recorded session, apply one structured mutation at an integer index,
and have a live (or replay-of-different-session) runner produce a fresh
continuation. Useful for "what if the user had said no" / "what if the
search had returned empty" exploration without re-running the entire
session from scratch.

Design choices, see also `designs/standout.md` section #1:

- **Mutations are values, not callbacks.** Each mutation is a frozen
  dataclass with primitive fields, so it serializes, can be stored next
  to a session, and produces reproducible counterfactuals. We never
  accept a callable as a mutation argument.
- **Indexing is by integer, not message identity.** Simpler than
  threading a "before-this-message" semantic through the API.
- **Returns a fresh `SessionRecord`.** The input is deep-copied at the
  start; the original is never mutated in place.
- **No "modify the runner mid-run" magic.** We slice the prefix from
  the original session, apply the mutation to the slice, then call the
  caller's runner once via a fresh `Orchestrator` to produce the new
  continuation. Clean async semantics, no special replay logic.
- **`session_id` and `created_at` are preserved.** The counterfactual is
  conceptually a sibling timeline of the original session, not a new
  session; downstream consumers (memory store, telemetry) decide what
  to do with the shared id.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.agents.orchestrator import Orchestrator, Runner
from harness.memory.record import SessionRecord
from harness.prompts.messages import ContentBlock, Message
from harness.tools.schema import ToolResult


@dataclass(frozen=True)
class RewriteTurn:
    """Replace `original.messages[index]` with `new_message`.

    Everything from `index + 1` onward is dropped; the runner produces
    a fresh continuation starting after the rewritten message.
    """

    index: int
    new_message: Message


@dataclass(frozen=True)
class InsertTurn:
    """Truncate the session after `after` and append `new_message`.

    Despite the name, this is **not** a true splice-insertion: the runner
    is then asked to produce a fresh continuation, and subsequent turns
    from the original session are **not** preserved. Effectively a
    "branch at index `after + 1`" operation — equivalent to
    `RewriteTurn(index=after + 1, new_message=...)` except that
    `after = -1` is allowed (inserts at the very start).

    Semantically true insertion (preserving the tail until the runner
    consumes or rewrites it) is roadmap item M3; for v1.x the name is
    retained for backwards compatibility and the docstring is the source
    of truth on behaviour. See `audit/RELEASE-TODO.md` item M2.8.
    """

    after: int
    new_message: Message


@dataclass(frozen=True)
class DeleteTurn:
    """Delete `original.messages[index]` and everything after it.

    The runner is then asked to continue from the truncated prefix. This
    matches the design-doc behaviour that a delete is "remove this turn
    and re-run forward" rather than "splice it out and pretend it never
    happened" — the latter would leave dangling tool-call/result pairs.
    """

    index: int


@dataclass(frozen=True)
class ReplaceToolResult:
    """Replace the `tool_result` at `original.messages[turn].content[block]`.

    Raises `ValueError` if the targeted block is not a `tool_result`.
    Everything from `turn + 1` onward is dropped; the runner produces
    a fresh continuation that sees the substituted result.
    """

    turn: int
    block: int
    new_result: ToolResult


Mutation = RewriteTurn | InsertTurn | DeleteTurn | ReplaceToolResult


def _check_index(index: int, length: int, *, what: str) -> None:
    if not (0 <= index < length):
        raise IndexError(f"{what} {index} is out of bounds for session with {length} messages")


def _apply_mutation(messages: list[Message], mutation: Mutation) -> list[Message]:
    """Build the prefix (slice + mutation) that the runner will continue from.

    The original `messages` list must already be a deep copy — callers in
    `counterfactual()` ensure that, so this function can hand fragments of
    `messages` straight into the prefix without aliasing the input.
    """
    if isinstance(mutation, RewriteTurn):
        _check_index(mutation.index, len(messages), what="rewrite index")
        return [*messages[: mutation.index], mutation.new_message]

    if isinstance(mutation, InsertTurn):
        # `after = -1` is the documented "insert at the very start" form.
        if mutation.after < -1 or mutation.after >= len(messages):
            raise IndexError(
                f"insert-after index {mutation.after} is out of bounds for "
                f"session with {len(messages)} messages "
                f"(use -1 to insert at the start)"
            )
        # InsertTurn semantics: truncate-then-append, not true splice-insert.
        # We keep the prefix `messages[:after + 1]`, append `new_message`,
        # and drop everything originally at `after + 1` or later. The runner
        # produces a fresh continuation; the original tail is NOT preserved.
        # See the class docstring and roadmap item M2.8 for the rationale.
        cut = mutation.after + 1
        return [*messages[:cut], mutation.new_message]

    if isinstance(mutation, DeleteTurn):
        _check_index(mutation.index, len(messages), what="delete index")
        return list(messages[: mutation.index])

    # ReplaceToolResult
    _check_index(mutation.turn, len(messages), what="tool-result turn index")
    target_message = messages[mutation.turn]
    if not (0 <= mutation.block < len(target_message.content)):
        raise IndexError(
            f"tool-result block index {mutation.block} is out of bounds for "
            f"message at turn {mutation.turn} with "
            f"{len(target_message.content)} blocks"
        )
    target_block = target_message.content[mutation.block]
    if target_block.type != "tool_result":
        raise ValueError(
            f"ReplaceToolResult target at turn={mutation.turn} block="
            f"{mutation.block} has type {target_block.type!r}, "
            "expected 'tool_result'"
        )
    new_blocks = list(target_message.content)
    new_blocks[mutation.block] = ContentBlock(
        type="tool_result",
        tool_result=mutation.new_result,
    )
    new_message = Message(role=target_message.role, content=new_blocks)
    return [*messages[: mutation.turn], new_message]


async def counterfactual(
    session: SessionRecord,
    mutation: Mutation,
    runner: Runner,
    orchestrator: Orchestrator,
) -> SessionRecord:
    """Apply `mutation` to `session`, then continue from the live runner.

    Returns a fresh `SessionRecord` whose `messages` are
    `prefix + [continuation]`, where `prefix` is the original session
    sliced and mutated, and `continuation` is the single `Message` returned
    by `runner` driven through a freshly-built `Orchestrator`.

    The input `session` is deep-copied first; the caller's instance is
    never mutated. `session_id` and `created_at` are preserved so the
    counterfactual reads as a sibling timeline of the original; `updated_at`
    advances naturally when the new record is constructed.

    The passed `orchestrator` is treated as a configuration bag: its
    `dispatcher`, `hooks`, and `telemetry` are reused, but the live runner
    you supply here is the one that produces the continuation — the
    orchestrator's own runner is ignored.
    """
    working = session.model_copy(deep=True)
    prefix = _apply_mutation(working.messages, mutation)

    drive = Orchestrator(
        dispatcher=orchestrator.dispatcher,
        hooks=orchestrator.hooks,
        runner=runner,
        telemetry=orchestrator.telemetry,
    )
    continuation = await drive.run(working.agent, prefix)

    return SessionRecord(
        session_id=working.session_id,
        agent=working.agent,
        messages=[*prefix, continuation],
        metadata=dict(working.metadata),
        created_at=working.created_at,
    )


__all__ = [
    "DeleteTurn",
    "InsertTurn",
    "Mutation",
    "ReplaceToolResult",
    "RewriteTurn",
    "counterfactual",
]
