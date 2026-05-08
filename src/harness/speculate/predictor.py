"""Predictors decide which tool calls a `Speculator` should pre-execute.

A `Predictor` is a callable-shaped strategy (Protocol) that consumes the
running message history plus the set of idempotent-tools-the-agent-can-
use and returns a list of `ToolCall`s to speculate on. It does not run
the calls — that's the speculator's job.

Two predictors ship:

* :class:`LastCallPredictor` — predicts that the model will repeat its
  most recent N idempotent tool calls. Cheap, surprisingly effective for
  retry / refinement loops where the model re-runs `search` with the
  same query a turn later. No state.
* :class:`SequencePredictor` — counts bigrams over the recent tool-call
  sequence and predicts the most likely successor of the most-recent
  call. Stateless across instances; computes the table fresh each
  invocation.

External strategies satisfy the :class:`Predictor` protocol structurally
— no inheritance required.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from harness.prompts.messages import Message
from harness.tools.schema import Tool, ToolCall


class Predictor(Protocol):
    """Strategy interface for predicting upcoming tool calls.

    The speculator passes the running history and the dict of
    *eligible* tools (already filtered to `idempotent=True` and to the
    agent's `allowed_tools`) so the predictor only has to decide which
    of the eligible tools to bet on, not whether a tool is allowed.
    """

    def predict(
        self,
        history: list[Message],
        idempotent_tools: dict[str, Tool],
        max_predictions: int,
    ) -> list[ToolCall]: ...


def _iter_tool_calls(history: list[Message]) -> list[ToolCall]:
    """Walk every assistant `tool_use` block in encounter order.

    Helper used by both shipped predictors. Returns the calls in the
    order the model emitted them (oldest first).
    """
    calls: list[ToolCall] = []
    for msg in history:
        if msg.role != "assistant":
            continue
        for block in msg.content:
            if block.type == "tool_use" and block.tool_use is not None:
                calls.append(block.tool_use)
    return calls


class LastCallPredictor:
    """Predict that the model will repeat its most recent idempotent calls.

    Walks the history backward, collects the most recent `history_window`
    calls whose tool name is in the eligible set, and returns them in
    reverse-chronological order (most recent first), truncated to the
    speculator's `max_predictions` cap.

    This is the workhorse predictor — common interactive patterns
    (search→refine→search, read→edit→read) repeat the same idempotent
    call shape, and a hit on those saves the round-trip latency.
    """

    def __init__(self, history_window: int = 3) -> None:
        self.history_window = history_window

    def predict(
        self,
        history: list[Message],
        idempotent_tools: dict[str, Tool],
        max_predictions: int,
    ) -> list[ToolCall]:
        out: list[ToolCall] = []
        for call in reversed(_iter_tool_calls(history)):
            if call.name not in idempotent_tools:
                continue
            out.append(call)
            if len(out) >= self.history_window:
                break
        return out[:max_predictions]


class SequencePredictor:
    """Predict the next call from a bigram model over the call sequence.

    Builds a table `(prev_tool_name → Counter(next_tool_name))` from the
    history, then for the most-recent call's name picks the top
    `max_predictions` successors. Predicted `ToolCall.arguments` are
    inherited from the successor instance that *followed* the most
    recent occurrence of the predecessor — i.e. the most recent
    `(predecessor, successor)` pair specifically, not the most recent
    instance of the successor tool overall. So if the most recent
    `(search, parse)` pair was `parse(q="foo")` even though a later
    standalone `parse(q="bar")` exists, we predict `parse(q="foo")`.
    Treats the bigram pair as the unit of evidence; matches what
    "bigram model" usually means.

    No bigram observed for the previous tool yet → returns `[]`. Empty
    history → returns `[]`. The predictor never invents a call from
    thin air.
    """

    def predict(
        self,
        history: list[Message],
        idempotent_tools: dict[str, Tool],
        max_predictions: int,
    ) -> list[ToolCall]:
        calls = _iter_tool_calls(history)
        if len(calls) < 2:
            return []

        # Build bigram counts. Separately, track per-successor the most
        # recent successor *instance* whose predecessor was the latest
        # tool we just saw — that's the args template the bigram-correct
        # interpretation wants.
        bigrams: dict[str, Counter[str]] = {}
        for prev, nxt in zip(calls, calls[1:], strict=False):
            bigrams.setdefault(prev.name, Counter())[nxt.name] += 1

        latest_name = calls[-1].name
        if latest_name not in bigrams:
            return []

        # Walk the pair list backward and remember the first (= most recent)
        # successor instance for each successor name when paired with
        # `latest_name`. This fixes the "took args from a *standalone* later
        # successor instance instead of the bigram-paired one" gap.
        recent_paired: dict[str, ToolCall] = {}
        for prev, nxt in reversed(list(zip(calls, calls[1:], strict=False))):
            if prev.name != latest_name:
                continue
            if nxt.name not in recent_paired:
                recent_paired[nxt.name] = nxt

        ranked = [n for n, _ in bigrams[latest_name].most_common()]
        eligible = [n for n in ranked if n in idempotent_tools]

        out: list[ToolCall] = []
        for name in eligible[:max_predictions]:
            template = recent_paired.get(name)
            if template is None:
                continue
            # Strip the model-assigned id; the speculator will produce a
            # fresh result whose id is rewritten to match the actual call.
            out.append(ToolCall(name=name, arguments=dict(template.arguments)))
        return out
