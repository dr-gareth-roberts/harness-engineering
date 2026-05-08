"""Cross-session predictor: mine bigrams from past `SessionRecord`s in a store.

Run with: `uv run python examples/cross_session.py`

`harness.speculate.LastCallPredictor` only sees the *current* conversation,
so when the current history is short — or in our case, ends with a single
`search` call — it can only predict a repeat of that call. That's a fine
warm-cache strategy for retry loops, but it ignores the fact that you
might have run hundreds of sessions whose tool sequences encode useful
priors.

`CrossSessionPredictor` extends the bigram model used by
`SequencePredictor` to the union of tool-call sequences from the K most
recent past sessions in a `MemoryStore`. It anchors on whatever tool
just ran in the current history and predicts the most-likely successor
across the corpus of past + current sessions, with sentinel boundaries
so cross-session bigrams cannot bridge unrelated runs.

This example builds an `InMemoryStore`, saves three synthetic sessions
that all follow a clear `search → parse` pattern (each with different
arguments), then constructs a current history that ends in a `search`
call and contrasts the two predictors:

  * `LastCallPredictor`  predicts a repeat of `search`.
  * `CrossSessionPredictor` predicts `parse` — the cross-session
    successor — with arguments inherited from the most recent past
    session's `parse` call.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from pydantic import BaseModel

from harness.agents import SubAgent
from harness.memory import InMemoryStore, SessionRecord
from harness.prompts.messages import ContentBlock, Message
from harness.speculate import CrossSessionPredictor, LastCallPredictor
from harness.tools.schema import Tool, ToolCall


class _Args(BaseModel):
    query: str = ""


def _agent() -> SubAgent:
    return SubAgent(
        name="cross-session-demo",
        system_prompt="",
        model="demo-model",
        allowed_tools=["search", "parse"],
    )


def _idempotent_tools() -> dict[str, Tool]:
    """The set of tools the speculator would consider eligible. The
    predictor's `idempotent_tools` argument is what filters predictions
    to "tools we'd be comfortable speculating on".
    """
    return {
        name: Tool(
            name=name,
            description="",
            input_model=_Args,
            handler=lambda args: args.query,
            idempotent=True,
        )
        for name in ("search", "parse")
    }


def _assistant_tool_use(name: str, arguments: dict[str, object]) -> Message:
    return Message(
        role="assistant",
        content=[
            ContentBlock(
                type="tool_use",
                tool_use=ToolCall(name=name, arguments=arguments, id=f"id-{name}"),
            )
        ],
    )


def _build_record(session_id: str, query: str, *, day: int) -> SessionRecord:
    """A short record: search(query) -> parse(query). The predictable
    bigram is what we're mining. `updated_at` is set explicitly so the
    "K most recent" sort is deterministic across machines.
    """
    return SessionRecord(
        session_id=session_id,
        agent=_agent(),
        messages=[
            _assistant_tool_use("search", {"query": query}),
            _assistant_tool_use("parse", {"query": query}),
        ],
        updated_at=datetime(2026, 1, day, tzinfo=UTC),
        created_at=datetime(2026, 1, day, tzinfo=UTC),
    )


def _format_predictions(label: str, calls: list[ToolCall]) -> str:
    if not calls:
        return f"  {label}: (no predictions)"
    rendered = ", ".join(f"{c.name}({c.arguments})" for c in calls)
    return f"  {label}: [{rendered}]"


async def main() -> int:
    transcript: list[str] = []

    # Three synthetic past sessions, each running the same search → parse
    # pattern with different arguments. `updated_at` increases with `day`
    # so the most recent record (s3) is what arg-inheritance will pick.
    store = InMemoryStore()
    await store.save(_build_record("s1", query="alpha", day=1))
    await store.save(_build_record("s2", query="beta", day=2))
    await store.save(_build_record("s3", query="gamma", day=3))

    transcript.append("--- past sessions saved (3 x search→parse) ---")
    for rec in await store.list(limit=10):
        bigram = " -> ".join(
            block.tool_use.name
            for msg in rec.messages
            for block in msg.content
            if block.type == "tool_use" and block.tool_use is not None
        )
        transcript.append(f"  {rec.session_id} ({rec.updated_at.date()}): {bigram}")

    # Current history: a single `search` call, no `parse` yet. This is
    # the discriminating shape — LastCallPredictor sees only "search just
    # ran" and predicts a repeat, while CrossSessionPredictor uses the
    # past sessions' bigram to predict `parse`.
    current_history = [_assistant_tool_use("search", {"query": "delta"})]
    transcript.append("--- current history ends in: search(query='delta') ---")

    predictor = await CrossSessionPredictor.from_store(store, K=3)
    cross_session = predictor.predict(
        history=current_history,
        idempotent_tools=_idempotent_tools(),
        max_predictions=2,
    )
    last_call = LastCallPredictor(history_window=1).predict(
        history=current_history,
        idempotent_tools=_idempotent_tools(),
        max_predictions=2,
    )

    transcript.append("--- predictions ---")
    transcript.append(_format_predictions("LastCallPredictor", last_call))
    transcript.append(_format_predictions("CrossSessionPredictor", cross_session))

    # Note the args inheritance: `parse({'query': 'gamma'})` reflects the
    # most recent past session's parse call. The arg-inheritance contract
    # is "most recent paired successor", documented in
    # `CrossSessionPredictor`'s coupling-invariant docstring.
    if cross_session and cross_session[0].name == "parse":
        transcript.append(
            f"  → cross-session predicted 'parse' with args from the most "
            f"recent past session: {cross_session[0].arguments}"
        )

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
