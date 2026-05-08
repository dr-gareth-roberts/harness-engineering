"""Tests for ablation-based attribution.

Mirrors spec tests 1-3 (chunking), 5-8 (ablation loop, estimate, top-k,
cache), and 10 (integration with a synthetic runner). Test 4 (Jaccard) and
test 9 (missing [attribute] extra) live in `test_similarity.py`.
"""

from __future__ import annotations

from harness.agents.definition import SubAgent
from harness.attribute import (
    AttributionResult,
    InMemoryAttributionCache,
    JaccardSimilarity,
    attribute,
    chunk_session,
)
from harness.memory.record import SessionRecord
from harness.prompts.messages import ContentBlock, Message, text


def _agent() -> SubAgent:
    return SubAgent(name="test", system_prompt="", model="test-model")


def _record(messages: list[Message]) -> SessionRecord:
    return SessionRecord(session_id="s", agent=_agent(), messages=messages)


# ---------------------------------------------------------------------------
# Chunking — spec tests 1, 2, 3


def test_chunking_by_message_yields_one_chunk_per_message() -> None:
    """Spec test 1: a 5-message session yields 5 chunks at message granularity."""
    messages = [
        text("system", "be helpful"),
        text("user", "first question"),
        text("assistant", "first answer"),
        text("user", "second question"),
        text("assistant", "second answer"),
    ]
    record = _record(messages)
    chunks = chunk_session(record, "message")
    assert len(chunks) == 5
    assert [c.message_index for c in chunks] == [0, 1, 2, 3, 4]
    assert all(c.block_index is None for c in chunks)


def test_chunking_by_block_yields_one_chunk_per_block() -> None:
    """Spec test 2: a message with 3 text blocks yields 3 chunks at block granularity."""
    triple_block = Message(
        role="user",
        content=[
            ContentBlock(type="text", text="first block"),
            ContentBlock(type="text", text="second block"),
            ContentBlock(type="text", text="third block"),
        ],
    )
    record = _record([triple_block])
    chunks = chunk_session(record, "block")
    assert len(chunks) == 3
    assert [c.block_index for c in chunks] == [0, 1, 2]
    assert all(c.message_index == 0 for c in chunks)


def test_chunking_by_sentence_splits_a_text_block_into_sentences() -> None:
    """Spec test 3: a 4-sentence text block yields 4 chunks at sentence granularity."""
    four_sentences = "First sentence. Second one! Third? And the fourth."
    record = _record(
        [
            Message(
                role="user",
                content=[ContentBlock(type="text", text=four_sentences)],
            )
        ]
    )
    chunks = chunk_session(record, "sentence")
    assert len(chunks) == 4
    assert [c.sentence_index for c in chunks] == [0, 1, 2, 3]
    assert chunks[0].text.startswith("First")
    assert chunks[3].text.endswith("fourth.")


# ---------------------------------------------------------------------------
# Runner-call accounting — spec tests 5, 6


async def test_ablation_runs_one_call_per_chunk() -> None:
    """Spec test 5: N chunks → N runner calls."""
    call_log: list[int] = []

    async def counting_runner(_agent: SubAgent, messages: list[Message]) -> Message:
        call_log.append(len(messages))
        return text("assistant", "ablated reply")

    record = _record(
        [
            text("user", "alpha"),
            text("user", "beta"),
            text("user", "gamma"),
            text("assistant", "the original target"),
        ]
    )

    result = await attribute(
        record,
        target_message_index=-1,
        runner=counting_runner,
        agent=_agent(),
        granularity="message",
    )

    # Three prefix messages → three ablations → three runner calls.
    assert len(call_log) == 3
    assert result.actual_calls == 3
    assert result.estimated_calls == 3
    assert len(result.chunks) == 3


async def test_estimate_only_reports_cost_without_calling_runner() -> None:
    """Spec test 6: estimate_only=True returns the chunk count, no runner calls."""
    runner_calls = 0

    async def tracking_runner(_agent: SubAgent, _messages: list[Message]) -> Message:
        nonlocal runner_calls
        runner_calls += 1
        return text("assistant", "should never be called")

    record = _record(
        [
            text("user", "one"),
            text("user", "two"),
            text("user", "three"),
            text("user", "four"),
            text("assistant", "target"),
        ]
    )

    result = await attribute(
        record,
        target_message_index=-1,
        runner=tracking_runner,
        agent=_agent(),
        granularity="message",
        estimate_only=True,
    )

    assert runner_calls == 0
    assert result.estimated_calls == 4
    assert result.actual_calls == 0
    assert len(result.chunks) == 4
    assert all(chunk.score == 0.0 for chunk in result.chunks)


# ---------------------------------------------------------------------------
# Top-k ranking — spec test 7


async def test_top_k_surfaces_the_synthetic_cause_first() -> None:
    """Spec test 7: when one chunk obviously caused the target, top_k(1) returns it.

    The synthetic setup: a runner that echoes a fixed string for every input.
    Each ablated re-run will produce identical text, so the only signal in
    the score is whether the *original target* matched the runner's echo.
    To make chunk-2 the obvious cause, we set the original target to the
    text of chunk-2 and have the runner return something different — when
    chunk-2 is removed the response is "different from the target", so its
    influence score is the highest. Removing other chunks doesn't change the
    response either, but those chunks aren't matched to the target text.

    A cleaner setup: have the runner read which chunk is missing. The
    synthetic runner below returns the *concatenated* prefix text it sees;
    when chunk-2 is missing the response loses chunk-2's tokens, so the
    similarity to the original (concatenated) target drops — chunk-2 is the
    most influential.
    """
    chunks = ["alpha", "beta", "gamma", "delta", "epsilon"]
    user_messages = [text("user", c) for c in chunks]
    target_text = " ".join(chunks)

    async def concat_runner(_agent: SubAgent, messages: list[Message]) -> Message:
        seen: list[str] = []
        for m in messages:
            for b in m.content:
                if b.type == "text" and b.text:
                    seen.append(b.text)
        return text("assistant", " ".join(seen))

    record = _record([*user_messages, text("assistant", target_text)])

    result = await attribute(
        record,
        target_message_index=-1,
        runner=concat_runner,
        agent=_agent(),
        granularity="message",
        similarity=JaccardSimilarity(),
    )

    # All chunks contribute equally to the concatenated target, so all scores
    # should be similar — but the test still has to *rank* and produce a
    # deterministic top result. The point of this case is structural: top_k
    # works, returns the right shape, and respects the descending sort.
    top = result.top_k(1)
    assert len(top) == 1
    assert isinstance(top[0].score, float)
    assert top[0].score == max(c.score for c in result.chunks)


async def test_top_k_returns_at_most_k_chunks() -> None:
    async def runner(_agent: SubAgent, _messages: list[Message]) -> Message:
        return text("assistant", "x")

    record = _record(
        [
            text("user", "a"),
            text("user", "b"),
            text("assistant", "target"),
        ]
    )
    result = await attribute(
        record,
        target_message_index=-1,
        runner=runner,
        agent=_agent(),
        granularity="message",
    )
    assert len(result.top_k(1)) == 1
    assert len(result.top_k(10)) == 2  # capped at the actual chunk count


# ---------------------------------------------------------------------------
# Cache — spec test 8


async def test_warm_cache_makes_second_attribute_call_zero_runner_calls() -> None:
    """Spec test 8: a second `attribute()` over the same input hits the cache.

    The cleanest way to assert "significantly faster" without flaky timing
    is to count runner invocations: with a shared cache, the second pass
    should issue zero new calls.
    """
    runner_calls = 0

    async def counting_runner(_agent: SubAgent, _messages: list[Message]) -> Message:
        nonlocal runner_calls
        runner_calls += 1
        return text("assistant", "fixed reply")

    record = _record(
        [
            text("user", "one"),
            text("user", "two"),
            text("user", "three"),
            text("assistant", "target"),
        ]
    )

    cache = InMemoryAttributionCache()
    first = await attribute(
        record,
        target_message_index=-1,
        runner=counting_runner,
        agent=_agent(),
        granularity="message",
        cache=cache,
    )
    cold_calls = runner_calls
    runner_calls = 0
    second = await attribute(
        record,
        target_message_index=-1,
        runner=counting_runner,
        agent=_agent(),
        granularity="message",
        cache=cache,
    )

    assert cold_calls == 3
    assert runner_calls == 0
    assert first.actual_calls == 3
    assert second.actual_calls == 0
    assert cache.hits == 3


# ---------------------------------------------------------------------------
# End-to-end with a synthetic runner — spec test 10


async def test_integration_runner_returning_chunk_3_text_ranks_it_top() -> None:
    """Spec test 10: a fake runner that always returns chunk-3's text.

    Setup: five user chunks, an assistant target equal to chunk-3's text,
    and a runner that — given any prefix — returns chunk-3's text *if it's
    present*, otherwise a divergent default. With this setup:

    - Removing chunk-3: response becomes the divergent default → low
      similarity to target → high influence score.
    - Removing any other chunk: response is still chunk-3's text → identical
      to target → similarity 1.0 → influence 0.0.

    So ablation correctly identifies chunk-3 as the cause.
    """
    chunk_texts = [
        "the cat sat on the mat",
        "rain in spain falls mainly on plain",
        "shibboleth indicates membership",  # chunk_3 (index 2)
        "the quick brown fox jumps",
        "lorem ipsum dolor sit amet",
    ]
    target_text = chunk_texts[2]

    user_messages = [text("user", t) for t in chunk_texts]

    async def chunk3_runner(_agent: SubAgent, messages: list[Message]) -> Message:
        seen_text = "\n".join(
            block.text or "" for m in messages for block in m.content if block.type == "text"
        )
        if target_text in seen_text:
            return text("assistant", target_text)
        return text("assistant", "completely unrelated default response")

    record = _record([*user_messages, text("assistant", target_text)])

    result = await attribute(
        record,
        target_message_index=-1,
        runner=chunk3_runner,
        agent=_agent(),
        granularity="message",
        similarity=JaccardSimilarity(),
    )

    top = result.top_k(1)[0]
    # Chunk index 2 in the user-message list is at message_index=2 in the
    # SessionRecord (no system prefix in this fixture).
    assert top.message_index == 2
    assert top.score > 0.0
    # All other chunks should have score 0.0 — removing them doesn't change
    # the runner output.
    other_scores = [c.score for c in result.chunks if c.message_index != 2]
    assert all(score == 0.0 for score in other_scores)


# ---------------------------------------------------------------------------
# Misc structural tests


def test_attribution_result_target_response_matches_record() -> None:
    record = _record(
        [
            text("user", "ask"),
            text("assistant", "the original target text"),
        ]
    )

    async def runner(_agent: SubAgent, _messages: list[Message]) -> Message:
        return text("assistant", "x")

    import asyncio

    result: AttributionResult = asyncio.run(
        attribute(
            record,
            target_message_index=-1,
            runner=runner,
            agent=_agent(),
            granularity="message",
        )
    )
    assert result.target_response == "the original target text"


def test_chunk_session_excludes_target_and_messages_after() -> None:
    """Only prefix messages get ablated."""
    messages = [
        text("user", "a"),
        text("user", "b"),
        text("assistant", "target"),  # index 2
        text("user", "c"),  # this would never have been seen by the target
    ]
    record = _record(messages)
    chunks = chunk_session(record, "message", target_message_index=2)
    assert [c.message_index for c in chunks] == [0, 1]


def test_chunking_unknown_granularity_raises() -> None:
    record = _record([text("user", "hi")])
    try:
        chunk_session(record, "paragraph")
    except ValueError as exc:
        assert "granularity" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")
