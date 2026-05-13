"""Tests for ablation-based attribution.

Mirrors spec tests 1-3 (chunking), 5-8 (ablation loop, estimate, top-k,
cache), and 10 (integration with a synthetic runner). Test 4 (Jaccard) and
test 9 (missing [attribute] extra) live in `test_similarity.py`.
"""

from __future__ import annotations

import base64

from harness.agents.definition import SubAgent
from harness.attribute import (
    AttributionResult,
    InMemoryAttributionCache,
    JaccardSimilarity,
    LengthRatio,
    attribute,
    chunk_session,
)
from harness.memory.record import SessionRecord
from harness.prompts.messages import ContentBlock, ImageRef, Message, text


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

    Setup: an "echo-with-weight" runner returns the concatenation of all
    text it sees, but the second user message is *much longer* than the
    others. When that long chunk is ablated the response shrinks
    dramatically, so the `LengthRatio` similarity to the original target
    drops sharply. Ablating any other (short) chunk barely moves the
    length, so its influence score stays low.

    This deliberately uses a different mechanism from test 10:
    - Test 10 uses a conditional `if target_text in prefix` runner with
      Jaccard similarity (token-overlap signal).
    - Test 7 uses an echoing runner with `LengthRatio` similarity (length
      signal). Same conclusion (causal chunk ranks #1) reached by a
      different code path.
    """
    short = "x"
    long_chunk = "this is a much longer chunk with many more characters than its peers"
    chunk_texts = [short, long_chunk, short, short, short]  # causal at index 1
    causal_index = 1

    async def echo_runner(_agent: SubAgent, messages: list[Message]) -> Message:
        echoed = " ".join(
            block.text or ""
            for m in messages
            for block in m.content
            if block.type == "text" and block.text
        )
        return text("assistant", echoed)

    target_text = " ".join(chunk_texts)
    record = _record([*(text("user", c) for c in chunk_texts), text("assistant", target_text)])

    result = await attribute(
        record,
        target_message_index=-1,
        runner=echo_runner,
        agent=_agent(),
        granularity="message",
        similarity=LengthRatio(),
    )

    top = result.top_k(1)
    assert len(top) == 1
    assert top[0].message_index == causal_index, (
        "ablating the long chunk should produce the largest length-ratio "
        "divergence and rank #1 in top_k"
    )
    other_scores = [c.score for c in result.chunks if c.message_index != causal_index]
    assert top[0].score > max(other_scores)


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


# ---------------------------------------------------------------------------
# Multimodal — image blocks must contribute to ablation (M1.9)


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


async def test_ablating_an_image_message_yields_nonzero_influence() -> None:
    """Spec test M1.9: a chunk that *is* an image block must register influence.

    Regression: `_extract_text` and `_block_text` previously emitted the
    empty string for `image` / `file` blocks, so any prefix-similarity
    function would see no change when an image was ablated — the
    influence score was structurally pinned to 0.0 and multimodal
    regressions were invisible to `attribute()`.

    Setup: a fake runner whose response includes the image's stable
    fingerprint when the image is present, and a fallback otherwise.
    Ablating the image-bearing message strips the fingerprint from the
    prefix, shifts the runner's reply, and Jaccard registers a drop in
    similarity → non-zero influence score for the image chunk.
    """
    image = ImageRef(source="base64", media_type="image/png", data=_b64(b"\x89PNG-cause"))

    image_message = Message(
        role="user",
        content=[ContentBlock(type="image", image=image)],
    )
    record = _record(
        [
            text("user", "alpha"),
            image_message,
            text("user", "gamma"),
            text("assistant", "the runner cites the image fingerprint as cause"),
        ]
    )

    async def image_aware_runner(_agent: SubAgent, messages: list[Message]) -> Message:
        seen_image = any(block.type == "image" for msg in messages for block in msg.content)
        if seen_image:
            return text("assistant", "the runner cites the image fingerprint as cause")
        return text("assistant", "the runner saw no image and falls back to text only")

    result = await attribute(
        record,
        target_message_index=-1,
        runner=image_aware_runner,
        agent=_agent(),
        granularity="message",
        similarity=JaccardSimilarity(),
    )

    image_chunks = [c for c in result.chunks if c.message_index == 1]
    assert len(image_chunks) == 1, "expected exactly one chunk for the image-bearing message"
    image_chunk = image_chunks[0]
    assert image_chunk.score > 0.0, (
        "ablating the image message must produce a non-zero influence score; "
        "previously the score was pinned to 0.0 because image blocks rendered "
        "as the empty string"
    )

    # And the image chunk should outrank the inert text chunks that don't
    # gate the runner's behaviour at all.
    inert_chunks = [c for c in result.chunks if c.message_index in (0, 2)]
    assert all(image_chunk.score >= c.score for c in inert_chunks)


async def test_ablating_an_image_at_block_granularity_surfaces_influence() -> None:
    """Block-granularity ablation must also see image blocks.

    A user message that bundles both prose and an image should split into
    two block chunks; dropping the image block alone shifts the runner
    response and produces non-zero influence.
    """
    image = ImageRef(source="base64", media_type="image/png", data=_b64(b"\x89PNG-block"))
    mixed_message = Message(
        role="user",
        content=[
            ContentBlock(type="text", text="please describe"),
            ContentBlock(type="image", image=image),
        ],
    )
    record = _record(
        [
            mixed_message,
            text("assistant", "image-conditioned reply"),
        ]
    )

    async def image_gated_runner(_agent: SubAgent, messages: list[Message]) -> Message:
        seen_image = any(block.type == "image" for msg in messages for block in msg.content)
        if seen_image:
            return text("assistant", "image-conditioned reply")
        return text("assistant", "no image so no description")

    result = await attribute(
        record,
        target_message_index=-1,
        runner=image_gated_runner,
        agent=_agent(),
        granularity="block",
        similarity=JaccardSimilarity(),
    )

    # Two chunks expected — one per block in the prefix message.
    assert len(result.chunks) == 2
    image_chunk = next(c for c in result.chunks if c.block_index == 1)
    text_chunk = next(c for c in result.chunks if c.block_index == 0)
    assert image_chunk.score > 0.0
    # Removing only the prose leaves the image (and so the gated reply)
    # intact: the runner still says "image-conditioned reply", which is
    # exactly the target, so influence is 0.
    assert text_chunk.score == 0.0
