from __future__ import annotations

from pydantic import BaseModel

from harness.prompts.messages import ContentBlock, Message
from harness.speculate import LastCallPredictor, SequencePredictor
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


def _user_text(s: str) -> Message:
    return Message(role="user", content=[ContentBlock(type="text", text=s)])


# ---------------------------------------------------------------------------
# LastCallPredictor


def test_last_call_predictor_picks_most_recent_idempotent_call() -> None:
    history = [
        _user_text("first"),
        _assistant_tool_use("search", {"q": "a"}),
        _user_text("second"),
        _assistant_tool_use("search", {"q": "b"}),
    ]
    predictor = LastCallPredictor(history_window=1)
    out = predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search"),
        max_predictions=1,
    )
    assert len(out) == 1
    assert out[0].name == "search"
    assert out[0].arguments == {"q": "b"}


def test_last_call_predictor_returns_empty_when_no_idempotent_tools_called() -> None:
    history = [_user_text("nothing here")]
    predictor = LastCallPredictor()
    assert predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search"),
        max_predictions=2,
    ) == []


def test_last_call_predictor_honors_history_window_then_max_predictions_cap() -> None:
    history = [
        _assistant_tool_use("search", {"q": "a"}),
        _assistant_tool_use("search", {"q": "b"}),
        _assistant_tool_use("search", {"q": "c"}),
    ]
    # history_window=3, but max_predictions=2 → output is truncated to 2.
    predictor = LastCallPredictor(history_window=3)
    out = predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search"),
        max_predictions=2,
    )
    assert [c.arguments for c in out] == [{"q": "c"}, {"q": "b"}]


def test_last_call_predictor_skips_non_idempotent_calls() -> None:
    history = [
        _assistant_tool_use("send_email", {"q": "spam"}),  # not in idempotent_tools
        _assistant_tool_use("search", {"q": "real"}),
    ]
    predictor = LastCallPredictor()
    out = predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search"),
        max_predictions=2,
    )
    assert len(out) == 1
    assert out[0].name == "search"


# ---------------------------------------------------------------------------
# SequencePredictor


def test_sequence_predictor_picks_most_likely_successor_of_latest_call() -> None:
    # Pattern: search → parse, search → parse, search → answer.
    # Most-recent call: search → bigrams: parse=2, answer=1.
    history = [
        _assistant_tool_use("search", {"q": "x"}),
        _assistant_tool_use("parse", {"q": "x"}),
        _assistant_tool_use("search", {"q": "y"}),
        _assistant_tool_use("parse", {"q": "y"}),
        _assistant_tool_use("search", {"q": "z"}),
        _assistant_tool_use("answer", {"q": "z"}),
        _assistant_tool_use("search", {"q": "w"}),  # most recent
    ]
    predictor = SequencePredictor()
    out = predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search", "parse", "answer"),
        max_predictions=1,
    )
    assert len(out) == 1
    assert out[0].name == "parse"
    # Inherits arguments from the most recent `parse` call.
    assert out[0].arguments == {"q": "y"}


def test_sequence_predictor_empty_when_no_bigrams_observed() -> None:
    # Only one call → no bigrams.
    history = [_assistant_tool_use("search", {"q": "x"})]
    predictor = SequencePredictor()
    assert predictor.predict(
        history=history,
        idempotent_tools=_idempotent("search"),
        max_predictions=2,
    ) == []
