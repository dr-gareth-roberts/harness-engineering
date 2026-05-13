"""ReplayRunner + counterfactual + compare_sessions round-trip (M4.5).

Builds a `SessionRecord` of three turns mixing text + tool_use, applies
a `RewriteTurn` mutation, replays it via a fresh runner, then diffs the
result against the original. Pins that:

- Mutating a turn at index N preserves the diff matches at `< N` and
  flips them at `>= N`. (M2.8 truncate-then-append semantics; the
  surface that `compare_sessions` exposes is per-turn `matches`, not a
  "diff hash" — the task wording is loose; the boolean trail is what
  pins behaviour.)
- An `InsertTurn(after=k, new_message=M)` truncates everything after
  index `k`, appends `M`, then asks the runner for ONE fresh
  continuation. Final length is `k + 1 + 1 + 1` = `k + 3`. The
  original tail is NOT preserved.

Each test uses a fresh canned runner so the continuation is
deterministic and the diff hash differs only at the predicted index.
"""

from __future__ import annotations

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.memory.record import SessionRecord
from harness.prompts import Message, assistant_tool_use, text, user_tool_result
from harness.replay import (
    InsertTurn,
    RewriteTurn,
    compare_sessions,
    counterfactual,
)
from harness.tools import Dispatcher, ToolCall, ToolResult


def _agent() -> SubAgent:
    return SubAgent(
        name="replay-agent",
        system_prompt="",
        model="test-model",
        allowed_tools=["search"],
    )


def _build_three_turn_record() -> tuple[SessionRecord, list[Message]]:
    """Construct a three-turn session: user → assistant(tool_use+text)
    → user(tool_result) → assistant(text), shaped how a real
    captured trajectory would land.

    Returns the record and the assistant-only message list (the form a
    `ReplayRunner` would consume).
    """
    user_q = text("user", "find a thing")
    tool_call = ToolCall(id="t1", name="search", arguments={"query": "thing"})
    # Assistant message bundles the brief assistant text + the tool_use
    # in one message; mimics what a real model often emits.
    assistant_with_tool = Message(
        role="assistant",
        content=[
            *text("assistant", "looking it up...").content,
            *assistant_tool_use(tool_call).content,
        ],
    )
    tool_outcome = user_tool_result(ToolResult(id="t1", content="result-for-thing", is_error=False))
    assistant_final = text("assistant", "the answer is 42")
    messages = [user_q, assistant_with_tool, tool_outcome, assistant_final]

    record = SessionRecord(
        session_id="s-original",
        agent=_agent(),
        messages=messages,
    )
    return record, [assistant_with_tool, assistant_final]


async def test_rewrite_turn_then_compare_diff_only_at_mutated_index() -> None:
    """Rewrite the assistant turn at index 1 with a new tool_use; replay
    produces a fresh continuation; the comparison shows matches at
    indices 0 (the user prompt is unchanged) and a non-match starting
    at the rewritten index.
    """
    record, _ = _build_three_turn_record()

    # Replacement message: assistant emits a different tool_use plus
    # text. Different both structurally (tool args) and textually.
    new_tool_call = ToolCall(id="t2", name="search", arguments={"query": "other"})
    new_message = Message(
        role="assistant",
        content=[
            *text("assistant", "let me try another query").content,
            *assistant_tool_use(new_tool_call).content,
        ],
    )

    # The counterfactual runner produces a single fresh continuation
    # appended to `prefix + [new_message]`. We give it a canned
    # continuation that's text-only so the trajectory tail differs from
    # the original by both structure (no tool_result) and content.
    canned_continuation = text("assistant", "I gave up.")

    async def fresh_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return canned_continuation

    orchestrator = Orchestrator(Dispatcher(), HookRunner(), fresh_runner)
    mutated = await counterfactual(
        record,
        RewriteTurn(index=1, new_message=new_message),
        fresh_runner,
        orchestrator,
    )

    # Length: prefix `[user]` (index 0) + the new turn (index 1) + ONE
    # fresh continuation (index 2). Original was 4 turns; mutated is 3.
    assert len(mutated.messages) == 3
    assert mutated.session_id == record.session_id, (
        "counterfactual preserves session_id (sibling-timeline contract)"
    )

    diff = compare_sessions(record, mutated, name="rewrite-at-1")
    # The user prompt (index 0) is identical → matches.
    assert diff.turns[0].matches is True
    # Index 1: rewritten → no match.
    assert diff.turns[1].matches is False
    # Index 2: original had a tool_result, mutated has a fresh
    # assistant turn → no match.
    assert diff.turns[2].matches is False
    # Index 3: only the original has a 4th turn → no match.
    assert diff.turns[3].matches is False
    # Overall not equal.
    assert diff.matches is False


async def test_insert_turn_truncate_then_append_end_to_end() -> None:
    """`InsertTurn(after=k, new_message=M)` truncates everything
    originally at `k + 1` or later and replaces it with `M` plus one
    fresh continuation. The original turns after `k` are NOT preserved
    — the M2.8 surface contract.
    """
    record, _ = _build_three_turn_record()
    original_length = len(record.messages)
    assert original_length == 4  # sanity for the math below

    # Insert after index 0 (the user prompt). Everything originally at
    # index 1, 2, 3 must be dropped from the resulting prefix.
    new_message = text("user", "wait, change of plans")

    canned_continuation = text("assistant", "OK, new direction.")

    async def fresh_runner(agent: SubAgent, messages: list[Message]) -> Message:
        # The runner sees the *mutated prefix* — prove the prefix shape:
        # [original user] + [new_message] = 2 messages.
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].content[0].text == "wait, change of plans"
        return canned_continuation

    orchestrator = Orchestrator(Dispatcher(), HookRunner(), fresh_runner)
    mutated = await counterfactual(
        record,
        InsertTurn(after=0, new_message=new_message),
        fresh_runner,
        orchestrator,
    )

    # Length: prefix(1) + new_message(1) + continuation(1) = 3. The
    # original tail (turns 1, 2, 3) was dropped, exactly the truncate-
    # then-append semantics M2.8 pins.
    assert len(mutated.messages) == 3
    assert mutated.messages[0].role == "user"
    assert mutated.messages[0].content[0].text == "find a thing"
    assert mutated.messages[1].content[0].text == "wait, change of plans"
    assert mutated.messages[2].content[0].text == "OK, new direction."

    # The original session is untouched — the counterfactual deep-copied.
    assert len(record.messages) == original_length
    assert record.messages[1].role == "assistant"


async def test_compare_sessions_ignores_tool_use_ids_so_replay_round_trips() -> None:
    """`compare_sessions` strips `tool_use.id` / `tool_result.tool_use_id`
    during normalization (the comparator is M2-spec). A `ReplayRunner`-
    driven re-run that produces the same trajectory but re-generates
    fresh ids should still compare equal.

    Without this normalization, two semantically-identical replays would
    look different just because the runner re-emitted fresh `ToolCall.id`
    strings.
    """
    record, _ = _build_three_turn_record()

    # Build a structurally identical record where every tool id differs.
    rewired_messages: list[Message] = []
    for m in record.messages:
        new_blocks = []
        for b in m.content:
            if b.type == "tool_use" and b.tool_use is not None:
                new_blocks.append(
                    b.model_copy(
                        update={
                            "tool_use": ToolCall(
                                id=f"different-{b.tool_use.id}",
                                name=b.tool_use.name,
                                arguments=b.tool_use.arguments,
                            )
                        }
                    )
                )
            elif b.type == "tool_result" and b.tool_result is not None:
                new_blocks.append(
                    b.model_copy(
                        update={
                            "tool_result": ToolResult(
                                id=f"different-{b.tool_result.id}",
                                content=b.tool_result.content,
                                is_error=b.tool_result.is_error,
                            )
                        }
                    )
                )
            else:
                new_blocks.append(b)
        rewired_messages.append(Message(role=m.role, content=new_blocks))

    rewired = SessionRecord(
        session_id="s-rewired",
        agent=_agent(),
        messages=rewired_messages,
    )

    diff = compare_sessions(record, rewired, name="id-only-diff")
    # Every turn matches; ids were the only difference.
    assert diff.matches is True
    assert all(t.matches for t in diff.turns)
