from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from pydantic import BaseModel

from harness.agents.definition import SubAgent
from harness.memory.record import SessionRecord
from harness.memory.store import InMemoryStore
from harness.prompts.messages import ContentBlock, Message
from harness.speculate.cross_session import (
    _SESSION_SENTINEL,
    CrossSessionPredictor,
    _build_synthetic_history,
)
from harness.tools.schema import Tool, ToolCall


class _Args(BaseModel):
    q: str = ""


def _idempotent(*names: str) -> dict[str, Tool]:
    return {
        name: Tool(
            name=name,
            description="",
            input_model=_Args,
            handler=lambda args: args.q,
            idempotent=True,
        )
        for name in names
    }


def _agent() -> SubAgent:
    return SubAgent(name="t", system_prompt="", model="m", allowed_tools=[])


def _assistant_tool_use(name: str, args: dict[str, object]) -> Message:
    return Message(
        role="assistant",
        content=[
            ContentBlock(
                type="tool_use",
                tool_use=ToolCall(name=name, arguments=args, id=f"id-{name}"),
            )
        ],
    )


def _record(
    session_id: str, calls: list[tuple[str, dict[str, object]]], *, day: int
) -> SessionRecord:
    """Build a SessionRecord with explicit `updated_at` so recency tests are stable."""
    return SessionRecord(
        session_id=session_id,
        agent=_agent(),
        messages=[_assistant_tool_use(name, args) for name, args in calls],
        updated_at=datetime(2026, 1, day, tzinfo=UTC),
        created_at=datetime(2026, 1, day, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# 1. Construction loads the K most-recent records.


def test_from_store_loads_K_most_recent_records() -> None:
    store = InMemoryStore()
    rec_a = _record("a", [("search", {"q": "a"}), ("parse", {"q": "a"})], day=1)
    rec_b = _record("b", [("search", {"q": "b"}), ("parse", {"q": "b"})], day=2)
    rec_c = _record("c", [("search", {"q": "c"}), ("parse", {"q": "c"})], day=3)

    async def setup_and_load() -> CrossSessionPredictor:
        await store.save(rec_a)
        await store.save(rec_b)
        await store.save(rec_c)
        return await CrossSessionPredictor.from_store(store, K=2)

    predictor = asyncio.run(setup_and_load())

    # Synthetic history holds 2 records' messages plus 1 sentinel between
    # them: 2 calls + 1 sentinel + 2 calls = 5 messages.
    assert len(predictor._cross_session_history) == 5

    # Records appear chronologically (oldest of K first, newest last) so the
    # most-recent record's calls sit closest to the current history.
    sentinel_idx = next(
        i
        for i, m in enumerate(predictor._cross_session_history)
        if m.content[0].tool_use is not None and m.content[0].tool_use.name == _SESSION_SENTINEL
    )
    pre_sentinel_args = [
        m.content[0].tool_use.arguments
        for m in predictor._cross_session_history[:sentinel_idx]
        if m.content[0].tool_use is not None
    ]
    post_sentinel_args = [
        m.content[0].tool_use.arguments
        for m in predictor._cross_session_history[sentinel_idx + 1 :]
        if m.content[0].tool_use is not None
    ]
    # K=2 of (a@day1, b@day2, c@day3) → top-2 by recency = [c, b], then
    # reversed to chronological → [b, c]: b first, sentinel, c last.
    assert pre_sentinel_args == [{"q": "b"}, {"q": "b"}]
    assert post_sentinel_args == [{"q": "c"}, {"q": "c"}]


# ---------------------------------------------------------------------------
# 2. K=0 falls back to current-history-only behaviour.


def test_K_zero_falls_back_to_sequence_predictor_on_current_history() -> None:
    store = InMemoryStore()

    async def setup_and_load() -> CrossSessionPredictor:
        await store.save(_record("a", [("search", {"q": "a"})], day=1))
        return await CrossSessionPredictor.from_store(store, K=0)

    predictor = asyncio.run(setup_and_load())
    assert predictor._cross_session_history == []

    # With only the current history, behaviour matches SequencePredictor.
    history = [
        _assistant_tool_use("search", {"q": "x"}),
        _assistant_tool_use("parse", {"q": "x"}),
        _assistant_tool_use("search", {"q": "y"}),  # latest
    ]
    out = predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search", "parse"),
        max_predictions=1,
    )
    assert len(out) == 1
    assert out[0].name == "parse"


# ---------------------------------------------------------------------------
# 3. Bigrams must not bridge session boundaries.


def test_bigrams_do_not_bridge_session_boundaries() -> None:
    """Record A ends with `parse`; record B starts with `answer`. With no
    in-record successor for `parse` and the sentinel separating sessions,
    `answer` must NOT be predicted from a (parse → answer) bigram."""
    rec_a = _record("a", [("search", {"q": "a"}), ("parse", {"q": "a"})], day=1)
    rec_b = _record("b", [("answer", {"q": "b"}), ("search", {"q": "b"})], day=2)

    store = InMemoryStore()

    async def setup_and_load() -> CrossSessionPredictor:
        await store.save(rec_a)
        await store.save(rec_b)
        return await CrossSessionPredictor.from_store(store, K=2)

    predictor = asyncio.run(setup_and_load())
    # Current history latest = parse. Without bridging we have no successor.
    history = [_assistant_tool_use("parse", {"q": "now"})]
    out = predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search", "parse", "answer"),
        max_predictions=2,
    )
    # The only (parse → real-tool) bigram would be the bridged one — and
    # that's exactly what the sentinel must prevent.
    assert [c.name for c in out] == []


# ---------------------------------------------------------------------------
# 4. Cross-session signal aggregates across many records.


def test_cross_session_signal_aggregates_across_records() -> None:
    """Five records each end with (search, parse). Current latest is
    `search`. The cross-session bigram (search → parse) should dominate."""
    store = InMemoryStore()

    async def setup_and_load() -> CrossSessionPredictor:
        for i in range(1, 6):
            await store.save(
                _record(
                    f"r{i}",
                    [("search", {"q": f"q{i}"}), ("parse", {"q": f"q{i}"})],
                    day=i,
                )
            )
        return await CrossSessionPredictor.from_store(store, K=5)

    predictor = asyncio.run(setup_and_load())
    history = [_assistant_tool_use("search", {"q": "now"})]
    out = predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search", "parse"),
        max_predictions=1,
    )
    assert len(out) == 1
    assert out[0].name == "parse"
    # Args inherit from the most-recent (search → parse) pair. Records are
    # placed in chronological order with the newest closest to the current
    # history, so the most recent paired successor is `parse(q="q5")`.
    assert out[0].arguments == {"q": "q5"}


# ---------------------------------------------------------------------------
# 5. Recency: the K most-recent records dominate.


def test_recency_uses_K_most_recent_records_only() -> None:
    """Older records carry one bigram; recent records carry a different
    one. Limiting to K should make the recent bigram the only signal."""
    store = InMemoryStore()

    async def setup_and_load() -> CrossSessionPredictor:
        # Older sessions: search → answer (3 of them, days 1..3).
        for i in range(1, 4):
            await store.save(
                _record(
                    f"old{i}",
                    [("search", {"q": "old"}), ("answer", {"q": "old"})],
                    day=i,
                )
            )
        # Recent sessions: search → parse (2 of them, days 4, 5).
        for i in range(4, 6):
            await store.save(
                _record(
                    f"new{i}",
                    [("search", {"q": "new"}), ("parse", {"q": "new"})],
                    day=i,
                )
            )
        # K=2 → only the two recent ones contribute to bigrams.
        return await CrossSessionPredictor.from_store(store, K=2)

    predictor = asyncio.run(setup_and_load())
    history = [_assistant_tool_use("search", {"q": "now"})]
    out = predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search", "parse", "answer"),
        max_predictions=2,
    )
    assert [c.name for c in out] == ["parse"]


# ---------------------------------------------------------------------------
# 6. Sentinel never appears in output even if the caller passes it through
#    `idempotent_tools`.


def test_sentinel_filtered_from_output_even_when_in_idempotent_tools() -> None:
    """Construct a synthetic history where the sentinel sits *between* two
    records; the current history's latest call matches the call right
    before the sentinel. SequencePredictor would then predict the sentinel
    as a successor — our filter must drop it."""
    rec_a = _record("a", [("search", {"q": "x"})], day=1)
    rec_b = _record("b", [("search", {"q": "y"})], day=2)
    synthetic = _build_synthetic_history([rec_b, rec_a])  # most-recent first

    predictor = CrossSessionPredictor(synthetic)

    # Latest call is `search`. Sentinel is the next "call" after the first
    # `search` in the synthetic history → bigram (search → sentinel) exists.
    history = [_assistant_tool_use("search", {"q": "now"})]

    # Pass the sentinel name into idempotent_tools to force the case.
    tools = _idempotent("search", _SESSION_SENTINEL)
    out = predictor.predict(history=history, idempotent_tools=tools, max_predictions=2)
    # Sentinel filtered; nothing else qualifies (no other successors).
    assert all(c.name != _SESSION_SENTINEL for c in out)


# ---------------------------------------------------------------------------
# 7. predict() is sync — works with no running event loop.


def test_predict_is_sync_outside_an_event_loop() -> None:
    store = InMemoryStore()

    async def load() -> CrossSessionPredictor:
        await store.save(_record("a", [("search", {"q": "x"}), ("parse", {"q": "x"})], day=1))
        return await CrossSessionPredictor.from_store(store, K=1)

    predictor = asyncio.run(load())

    # No event loop running here — predict() must not require one.
    history = [_assistant_tool_use("search", {"q": "now"})]
    out = predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search", "parse"),
        max_predictions=1,
    )
    assert len(out) == 1
    assert out[0].name == "parse"
