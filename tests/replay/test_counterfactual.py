from __future__ import annotations

import copy

import pytest

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.memory import SessionRecord
from harness.prompts import ContentBlock, Message, text
from harness.replay import compare_sessions
from harness.replay.counterfactual import (
    DeleteTurn,
    InsertTurn,
    ReplaceToolResult,
    RewriteTurn,
    counterfactual,
)
from harness.runner import CannedRunner
from harness.tools import Dispatcher, ToolCall, ToolResult

# ---------------------------------------------------------------------------
# Fixtures / helpers


def _agent() -> SubAgent:
    return SubAgent(name="bot", system_prompt="be helpful", model="test-model")


def _make_orch(runner: CannedRunner) -> Orchestrator:
    """Build an orchestrator wired to `runner`. Counterfactual should ignore
    this runner and use the one passed positionally — but we still need a
    callable so the constructor is happy."""
    return Orchestrator(Dispatcher(), HookRunner(), runner)


def _record_with_history(
    messages: list[Message] | None = None,
    *,
    session_id: str = "sess_orig",
) -> SessionRecord:
    if messages is None:
        messages = [
            text("user", "hello"),
            text("assistant", "hi there"),
            text("user", "tell me a joke"),
            text("assistant", "why did the chicken..."),
            text("user", "why?"),
            text("assistant", "to get to the other side"),
        ]
    return SessionRecord(
        session_id=session_id,
        agent=_agent(),
        messages=messages,
    )


def _texts(record: SessionRecord) -> list[str | None]:
    out: list[str | None] = []
    for msg in record.messages:
        chunks = [b.text for b in msg.content if b.type == "text" and b.text]
        out.append("".join(chunks) if chunks else None)
    return out


# ---------------------------------------------------------------------------
# 1. Rewrite a user turn — prefix preserved, mutated message present, tail fresh.


async def test_rewrite_user_turn_keeps_prefix_and_uses_runner_for_tail() -> None:
    original = _record_with_history()
    runner = CannedRunner(["fresh continuation"])

    result = await counterfactual(
        session=original,
        mutation=RewriteTurn(index=2, new_message=text("user", "actually, never mind")),
        runner=runner,
        orchestrator=_make_orch(runner),
    )

    assert _texts(result)[:2] == _texts(original)[:2]
    assert result.messages[2].role == "user"
    assert _texts(result)[2] == "actually, never mind"
    assert _texts(result)[3] == "fresh continuation"
    assert len(result.messages) == 4


# ---------------------------------------------------------------------------
# 2. Insert a user turn — truncate-then-append semantics (see InsertTurn docstring).
#
# Despite the name, `InsertTurn` does NOT preserve the original tail: it
# truncates at `after + 1` and appends the new message, then the runner
# produces a fresh continuation. The original test name is retained for
# backwards compatibility; the assertions below codify the real semantics.


async def test_insert_user_turn_grows_history_by_two_at_minimum() -> None:
    original = _record_with_history()
    runner = CannedRunner(["got it"])

    result = await counterfactual(
        session=original,
        mutation=InsertTurn(after=1, new_message=text("user", "wait, one more thing")),
        runner=runner,
        orchestrator=_make_orch(runner),
    )

    # Inserted message lands at index 2 (after = 1 means cut = 2).
    assert _texts(result)[:2] == _texts(original)[:2]
    assert result.messages[2].role == "user"
    assert _texts(result)[2] == "wait, one more thing"
    assert _texts(result)[3] == "got it"
    # Prefix (turns 0..1) + insert + continuation = 4 messages, original had 6.
    # The "≥2 more than the prefix" property holds: 4 ≥ 2 + 2.
    assert len(result.messages) >= len(original.messages[:2]) + 2


async def test_insert_turn_truncates_tail_does_not_preserve_subsequent_turns() -> None:
    """Explicitly pin the truncate-then-append behaviour of `InsertTurn`.

    The original session has 6 messages; an `InsertTurn(after=1, ...)`
    must yield exactly 4 messages (prefix[0:2] + inserted + continuation).
    Subsequent turns from the original session (indices 2..5) are NOT
    preserved — verifying that the implementation does not splice. See
    the `InsertTurn` class docstring and roadmap item M2.8.
    """
    original = _record_with_history()
    runner = CannedRunner(["got it"])
    inserted_text = "wait, one more thing"

    result = await counterfactual(
        session=original,
        mutation=InsertTurn(after=1, new_message=text("user", inserted_text)),
        runner=runner,
        orchestrator=_make_orch(runner),
    )

    # Truncation is total: exactly prefix (2) + inserted (1) + continuation (1).
    assert len(result.messages) == 4
    assert len(original.messages) == 6

    # None of the original tail messages (indices 2..5) appear in the result.
    original_tail_texts = _texts(original)[2:]
    result_texts = _texts(result)
    for tail_text in original_tail_texts:
        if tail_text is None:
            continue
        assert tail_text not in result_texts, (
            f"InsertTurn must truncate the tail, but original turn "
            f"{tail_text!r} leaked into the counterfactual at index "
            f"{result_texts.index(tail_text)}"
        )

    # And the only "new" content past the prefix is the inserted message + runner reply.
    assert result_texts[2] == inserted_text
    assert result_texts[3] == "got it"


async def test_insert_turn_at_minus_one_truncates_entire_session() -> None:
    """`InsertTurn(after=-1, ...)` inserts at the very start, dropping all
    original turns. The result is the inserted message plus one runner
    continuation — nothing else from the original survives."""
    original = _record_with_history()
    runner = CannedRunner(["fresh start"])

    result = await counterfactual(
        session=original,
        mutation=InsertTurn(after=-1, new_message=text("user", "new beginning")),
        runner=runner,
        orchestrator=_make_orch(runner),
    )

    assert len(result.messages) == 2
    assert _texts(result) == ["new beginning", "fresh start"]
    # No original content survives.
    for original_text in _texts(original):
        if original_text is None:
            continue
        assert original_text not in _texts(result)


# ---------------------------------------------------------------------------
# 3. Delete an assistant turn — original turn skipped, new continuation fed back.


async def test_delete_assistant_turn_drops_it_and_continues_from_prefix() -> None:
    original = _record_with_history()
    runner = CannedRunner(["replacement assistant reply"])

    # Index 3 is the assistant "why did the chicken..."
    result = await counterfactual(
        session=original,
        mutation=DeleteTurn(index=3),
        runner=runner,
        orchestrator=_make_orch(runner),
    )

    assert _texts(result)[:3] == _texts(original)[:3]
    # The deleted assistant message must not appear at index 3.
    assert _texts(result)[3] != _texts(original)[3]
    assert _texts(result)[3] == "replacement assistant reply"
    assert len(result.messages) == 4


# ---------------------------------------------------------------------------
# 4. Replace a tool_result — downstream assistant reflects the new result.


class _ReadsToolResultRunner:
    """A tiny runner that echoes back the most recent tool_result content.

    Lets us prove the substituted result reaches whatever continuation runs
    after the prefix — i.e. the substitution actually propagates.
    """

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        last_result = ""
        for msg in reversed(messages):
            for block in reversed(msg.content):
                if block.type == "tool_result" and block.tool_result is not None:
                    last_result = str(block.tool_result.content)
                    break
            if last_result:
                break
        return text("assistant", f"saw tool result: {last_result}")


async def test_replace_tool_result_downstream_reflects_new_value() -> None:
    call = ToolCall(name="search", arguments={"q": "weather"}, id="tu_1")
    original_result = ToolResult(id="tu_1", content="sunny", is_error=False)
    messages = [
        text("user", "what's the weather?"),
        Message(
            role="assistant",
            content=[ContentBlock(type="tool_use", tool_use=call)],
        ),
        Message(
            role="user",
            content=[ContentBlock(type="tool_result", tool_result=original_result)],
        ),
        text("assistant", "it's sunny"),
    ]
    original = _record_with_history(messages, session_id="sess_tool")
    runner = _ReadsToolResultRunner()
    orch = Orchestrator(Dispatcher(), HookRunner(), runner)

    swapped = ToolResult(id="tu_1", content="rainy", is_error=False)
    result = await counterfactual(
        session=original,
        mutation=ReplaceToolResult(turn=2, block=0, new_result=swapped),
        runner=runner,
        orchestrator=orch,
    )

    # Prefix turns 0 and 1 (user prompt + tool_use) are untouched.
    assert _texts(result)[:2] == _texts(original)[:2]
    # Turn 2 is the substituted tool_result message.
    swapped_block = result.messages[2].content[0]
    assert swapped_block.type == "tool_result"
    assert swapped_block.tool_result is not None
    assert swapped_block.tool_result.content == "rainy"
    # Turn 3 is the fresh continuation, and it must have read the new value.
    assert _texts(result)[3] == "saw tool result: rainy"


# ---------------------------------------------------------------------------
# 5. Mutation at index 0 — entire history is regenerated.


async def test_rewrite_at_index_zero_regenerates_entire_history() -> None:
    original = _record_with_history()
    runner = CannedRunner(["greetings, friend"])

    result = await counterfactual(
        session=original,
        mutation=RewriteTurn(index=0, new_message=text("user", "different opener")),
        runner=runner,
        orchestrator=_make_orch(runner),
    )

    # Only the rewritten turn 0 + the single new continuation remain.
    assert len(result.messages) == 2
    assert _texts(result)[0] == "different opener"
    assert _texts(result)[1] == "greetings, friend"


# ---------------------------------------------------------------------------
# 6. Mutation at the last turn — runner invoked once with the mutated tail.


async def test_rewrite_at_last_turn_invokes_runner_once_with_mutated_tail() -> None:
    original = _record_with_history()
    last_index = len(original.messages) - 1  # 5
    runner = CannedRunner(["follow-up"])

    result = await counterfactual(
        session=original,
        mutation=RewriteTurn(
            index=last_index,
            new_message=text("assistant", "actually, here's a different punchline"),
        ),
        runner=runner,
        orchestrator=_make_orch(runner),
    )

    # Prefix 0..4 unchanged.
    assert _texts(result)[:last_index] == _texts(original)[:last_index]
    # The rewritten last assistant turn.
    assert _texts(result)[last_index] == "actually, here's a different punchline"
    # Plus exactly one continuation = original_len + 0 (rewrite) + 1 (continuation)
    assert len(result.messages) == last_index + 2
    assert _texts(result)[last_index + 1] == "follow-up"
    # Runner was used exactly once: a CannedRunner with one reply now exhausted.
    assert runner._index == 1


# ---------------------------------------------------------------------------
# 7. Out-of-bounds index raises IndexError with a clear message.


async def test_out_of_bounds_index_raises_index_error_with_message() -> None:
    original = _record_with_history()
    runner = CannedRunner(["ignored"])
    orch = _make_orch(runner)

    with pytest.raises(IndexError, match="rewrite index 99"):
        await counterfactual(
            session=original,
            mutation=RewriteTurn(index=99, new_message=text("user", "x")),
            runner=runner,
            orchestrator=orch,
        )

    with pytest.raises(IndexError, match="delete index 99"):
        await counterfactual(
            session=original,
            mutation=DeleteTurn(index=99),
            runner=runner,
            orchestrator=orch,
        )

    with pytest.raises(IndexError, match="insert-after index 99"):
        await counterfactual(
            session=original,
            mutation=InsertTurn(after=99, new_message=text("user", "x")),
            runner=runner,
            orchestrator=orch,
        )

    with pytest.raises(IndexError, match="tool-result turn index 99"):
        await counterfactual(
            session=original,
            mutation=ReplaceToolResult(
                turn=99,
                block=0,
                new_result=ToolResult(content="x", is_error=False),
            ),
            runner=runner,
            orchestrator=orch,
        )


# ---------------------------------------------------------------------------
# 8. ReplaceToolResult on a non-tool_result block raises ValueError.


async def test_replace_tool_result_on_non_tool_result_block_raises_value_error() -> None:
    original = _record_with_history()  # all plain text blocks
    runner = CannedRunner(["ignored"])

    with pytest.raises(ValueError, match="expected 'tool_result'"):
        await counterfactual(
            session=original,
            mutation=ReplaceToolResult(
                turn=1,
                block=0,
                new_result=ToolResult(content="x", is_error=False),
            ),
            runner=runner,
            orchestrator=_make_orch(runner),
        )


# ---------------------------------------------------------------------------
# 9. Original SessionRecord is not mutated (defensive deep-copy).


async def test_original_session_record_is_not_mutated() -> None:
    original = _record_with_history()
    snapshot = copy.deepcopy(original)
    runner = CannedRunner(["something new"])

    await counterfactual(
        session=original,
        mutation=RewriteTurn(index=0, new_message=text("user", "totally different")),
        runner=runner,
        orchestrator=_make_orch(runner),
    )

    assert original.model_dump() == snapshot.model_dump()
    # Belt-and-braces: still reads back as the original first-turn user text.
    assert _texts(original)[0] == "hello"


# ---------------------------------------------------------------------------
# 10. Pairs with `compare_sessions` — prefix matches everywhere except mutation point.


async def test_compare_sessions_against_counterfactual_isolates_mutation_point() -> None:
    original = _record_with_history()
    runner = CannedRunner(["a wholly different reply"])

    result = await counterfactual(
        session=original,
        mutation=RewriteTurn(
            index=2,
            new_message=text("user", "let's pivot"),
        ),
        runner=runner,
        orchestrator=_make_orch(runner),
    )

    diff = compare_sessions(original, result)
    # Prefix turns 0 and 1 are untouched.
    assert diff.turns[0].matches is True
    assert diff.turns[1].matches is True
    # Turn 2 is the mutation point and must differ.
    assert diff.turns[2].matches is False
    # Whatever turns 3..N are, they must differ — original had 6 messages,
    # the counterfactual has 4 (prefix 0..1 + mutated 2 + continuation 3),
    # so positional alignment from turn 3 onward never agrees.
    assert all(not turn.matches for turn in diff.turns[2:])
    assert diff.matches is False
