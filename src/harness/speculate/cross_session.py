"""Cross-session predictor: bigram model over the K most-recent past sessions.

`SequencePredictor` mines bigrams from the *current* conversation only. For
short / cold-start conversations that signal is thin. `CrossSessionPredictor`
extends the bigram model to the union of tool-call sequences from the K most
recent `SessionRecord`s in a `MemoryStore`, anchored on whatever tool just ran
in the current `history`.

Design notes (pinned from the spec / advisor review):

* **K most-recent records** (default 5), not all records. The
  :class:`harness.memory.store.MemoryStore` ``list`` Protocol contract
  requires records be returned ordered by ``updated_at`` descending, so the
  ``[:K]`` slice we receive from ``store.list(limit=K)`` is already the
  K-most-recent set. We still re-sort defensively to remain correct against
  any future non-conformant third-party store.
* **Sentinel between sessions.** Concatenating tool sequences across sessions
  would create spurious bigrams of the form ``(last-call-of-A,
  first-call-of-B)``. We insert a sentinel ``Message`` between records so the
  bigram table built by `SequencePredictor` cannot bridge sessions. We also
  filter the sentinel out of the final predictions defensively.
* **Reuse, don't reimplement.** `predict()` builds a synthetic history list,
  prepends it to the *current* history (so the latest-call anchor is still
  the current one), and delegates to `SequencePredictor`.
* **Async load, sync predict.** Loading from the store needs an event loop;
  `Predictor.predict` is sync. We resolve this with an async classmethod
  factory `from_store(...)` that loads once at construction time. The
  resulting instance is plain-sync usable.

Coupling invariant:
    `CrossSessionPredictor` inherits its arg-inheritance semantics from
    `SequencePredictor` — specifically, the bigram-paired-successor walk
    that picks args from the most-recent paired instance of the predicted
    tool. We deliberately reverse the records to chronological order
    before building the synthetic history so that "most recent paired"
    resolves to the newest session's calls. If `SequencePredictor` ever
    changes its arg-inheritance strategy (e.g. from "most recent paired"
    to "most frequent paired"), this predictor's behavior changes with
    it. The reversal here is load-bearing only against the current
    `SequencePredictor` contract.
"""

from __future__ import annotations

from harness.memory.record import SessionRecord
from harness.memory.store import MemoryStore
from harness.prompts.messages import ContentBlock, Message
from harness.speculate.predictor import SequencePredictor
from harness.tools.schema import Tool, ToolCall

_SESSION_SENTINEL = "__cross_session_boundary__"


def _build_synthetic_history(records: list[SessionRecord]) -> list[Message]:
    """Concatenate records' message lists with a sentinel between each pair.

    The sentinel is a synthetic ``assistant`` message whose only block is a
    ``tool_use`` referencing ``_SESSION_SENTINEL``. ``SequencePredictor``'s
    bigram table will then see ``(real-call → sentinel)`` and ``(sentinel →
    real-call)`` pairs at session boundaries instead of bridging two real
    calls. The sentinel name is filtered out of the final predictions.
    """
    out: list[Message] = []
    for i, rec in enumerate(records):
        if i > 0:
            out.append(
                Message(
                    role="assistant",
                    content=[
                        ContentBlock(
                            type="tool_use",
                            tool_use=ToolCall(name=_SESSION_SENTINEL, arguments={}),
                        )
                    ],
                )
            )
        out.extend(rec.messages)
    return out


class CrossSessionPredictor:
    """Bigram predictor whose evidence is past sessions plus the current one.

    Construct via the async classmethod :meth:`from_store`; passing a
    pre-built synthetic history to ``__init__`` is also supported and is what
    tests use to exercise edge cases without a real store.
    """

    def __init__(self, cross_session_history: list[Message]) -> None:
        self._cross_session_history = cross_session_history

    @classmethod
    async def from_store(cls, store: MemoryStore, *, K: int = 5) -> CrossSessionPredictor:
        records = await store.list(limit=K)
        # Defensive sort: MemoryStore.list contracts recency order, but we
        # re-sort to remain correct against any future non-conformant store.
        records.sort(key=lambda r: r.updated_at, reverse=True)
        # Reverse to chronological order so the most-recent record's tool
        # sequence sits closest to the current history in the bigram walk.
        # SequencePredictor's "most recent paired successor" semantics then
        # inherit args from the most recent past session, not the oldest.
        chronological = list(reversed(records[:K]))
        return cls(_build_synthetic_history(chronological))

    def predict(
        self,
        history: list[Message],
        idempotent_tools: dict[str, Tool],
        max_predictions: int,
    ) -> list[ToolCall]:
        combined = self._cross_session_history + history
        out = SequencePredictor().predict(combined, idempotent_tools, max_predictions)
        return [c for c in out if c.name != _SESSION_SENTINEL]
