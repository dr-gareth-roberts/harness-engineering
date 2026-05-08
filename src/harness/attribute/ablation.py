"""Leave-one-out ablation for causal provenance.

For a chosen target assistant message in a `SessionRecord`, we re-run the
runner against the same prefix-of-history but with each input chunk removed
in turn. A chunk that strongly *caused* the target shows the largest
divergence when removed; the influence score is `1 - similarity(original,
ablated)`, so higher means more influential.

Three granularities are supported:

- `"message"` — drop a whole prefix message
- `"block"` — drop a single `ContentBlock` from a prefix message
- `"sentence"` — replace a sentence inside a text block with empty string

The contract:

- Original target text is never re-rendered; we read it directly from
  `session.messages[target]`.
- Only chunks before `target` are ablated; the target itself and anything
  after it stay out of the chunk list.
- Cache is keyed by `(hash(ablated_messages), target_index)`. With a fresh
  cache the runner is invoked once per chunk; with a warmed cache the
  invocation count goes to zero.
- `estimate_only=True` returns the chunk list and the would-be call count
  without invoking the runner once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from harness.agents.definition import SubAgent
from harness.agents.orchestrator import Runner
from harness.attribute.cache import (
    InMemoryAttributionCache,
    hash_messages,
)
from harness.attribute.similarity import JaccardSimilarity, Similarity
from harness.memory.record import SessionRecord
from harness.prompts.messages import ContentBlock, Message

Granularity = str  # one of: "message", "block", "sentence"

# Splits on the boundary *after* sentence-final punctuation, keeping the
# punctuation attached to the preceding sentence.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

_PREVIEW_LIMIT = 80


@dataclass(frozen=True)
class _ChunkRef:
    """Internal pointer to a piece of session content we may ablate."""

    message_index: int
    block_index: int | None
    sentence_index: int | None
    text: str


@dataclass
class AttributionChunk:
    """One element of an attribution ranking.

    `score` is the influence: `1 - similarity` between the original target
    and the response observed after this chunk was removed. Higher score =
    higher causal influence on the target.
    """

    message_index: int
    block_index: int | None
    score: float
    preview: str
    sentence_index: int | None = None


@dataclass
class AttributionResult:
    """Aggregated output of `attribute()`.

    `chunks` lists every ablated chunk in their original input order, each
    annotated with an influence score. `top_k` returns the most-influential
    chunks first. When `estimate_only=True` was passed, `chunks` is filled
    in but every `score` is `0.0`; the caller should look at
    `estimated_calls` to decide whether to commit to the live run.
    """

    target_response: str
    chunks: list[AttributionChunk] = field(default_factory=list)
    estimated_calls: int = 0
    actual_calls: int = 0

    def top_k(self, k: int) -> list[AttributionChunk]:
        ranked = sorted(self.chunks, key=lambda c: c.score, reverse=True)
        return ranked[:k]


def _extract_text(message: Message) -> str:
    """Concatenate all text-block content in a message.

    Tool-use / tool-result blocks are rendered through their stored text
    fallback when present, otherwise an empty string. The goal is a string
    that's good enough for token-overlap or embedding similarity, not a
    full faithful rendering.
    """
    parts: list[str] = []
    for block in message.content:
        if block.type == "text" and block.text is not None:
            parts.append(block.text)
        elif block.type == "tool_use" and block.tool_use is not None:
            parts.append(f"[tool_use: {block.tool_use.name}]")
        elif block.type == "tool_result" and block.tool_result is not None:
            content = block.tool_result.content
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(str(content))
    return "\n".join(parts)


def _split_sentences(text: str) -> list[str]:
    """Split on `.`/`?`/`!` followed by whitespace, keeping the punctuation."""
    if not text:
        return []
    parts = _SENTENCE_BOUNDARY.split(text)
    # Drop trailing empties that arise from a terminal punctuation+whitespace.
    return [p for p in parts if p]


def _preview(text: str) -> str:
    snippet = text.strip().replace("\n", " ")
    if len(snippet) <= _PREVIEW_LIMIT:
        return snippet
    return snippet[: _PREVIEW_LIMIT - 1].rstrip() + "…"


def _normalize_target(target_message_index: int, n_messages: int) -> int:
    if target_message_index < 0:
        return n_messages + target_message_index
    return target_message_index


def chunk_session(
    record: SessionRecord,
    granularity: Granularity,
    *,
    target_message_index: int | None = None,
) -> list[_ChunkRef]:
    """Enumerate ablation candidates from `record.messages`.

    When `target_message_index` is supplied, only messages strictly before
    the (normalized) target are considered. The target message itself, and
    any messages after it, are excluded — ablating them would either remove
    the very thing we're explaining or change the prefix in a way the target
    couldn't possibly have seen.
    """
    if granularity not in {"message", "block", "sentence"}:
        raise ValueError(
            f"granularity must be 'message', 'block', or 'sentence'; got {granularity!r}"
        )

    n = len(record.messages)
    upper = n if target_message_index is None else _normalize_target(target_message_index, n)
    upper = max(0, min(upper, n))

    chunks: list[_ChunkRef] = []
    for msg_idx in range(upper):
        message = record.messages[msg_idx]
        if granularity == "message":
            text = _extract_text(message)
            chunks.append(
                _ChunkRef(
                    message_index=msg_idx,
                    block_index=None,
                    sentence_index=None,
                    text=text,
                )
            )
            continue

        for block_idx, block in enumerate(message.content):
            block_text = _block_text(block)
            if granularity == "block":
                chunks.append(
                    _ChunkRef(
                        message_index=msg_idx,
                        block_index=block_idx,
                        sentence_index=None,
                        text=block_text,
                    )
                )
            else:  # sentence
                if block.type != "text" or not block.text:
                    # Non-text blocks ablate as a whole; we can't slice them.
                    if block_text:
                        chunks.append(
                            _ChunkRef(
                                message_index=msg_idx,
                                block_index=block_idx,
                                sentence_index=None,
                                text=block_text,
                            )
                        )
                    continue
                for sent_idx, sentence in enumerate(_split_sentences(block.text)):
                    chunks.append(
                        _ChunkRef(
                            message_index=msg_idx,
                            block_index=block_idx,
                            sentence_index=sent_idx,
                            text=sentence,
                        )
                    )
    return chunks


def _block_text(block: ContentBlock) -> str:
    if block.type == "text":
        return block.text or ""
    if block.type == "tool_use" and block.tool_use is not None:
        return f"[tool_use: {block.tool_use.name}]"
    if block.type == "tool_result" and block.tool_result is not None:
        content = block.tool_result.content
        return content if isinstance(content, str) else str(content)
    return ""


def _ablate_messages(
    messages: list[Message],
    chunk: _ChunkRef,
) -> list[Message]:
    """Return a new message list with `chunk` removed."""
    if chunk.message_index >= len(messages):
        return [m.model_copy(deep=True) for m in messages]

    if chunk.block_index is None:
        # Whole-message ablation.
        return [m.model_copy(deep=True) for i, m in enumerate(messages) if i != chunk.message_index]

    out: list[Message] = []
    for idx, message in enumerate(messages):
        if idx != chunk.message_index:
            out.append(message.model_copy(deep=True))
            continue

        new_blocks: list[ContentBlock] = []
        for block_idx, block in enumerate(message.content):
            if block_idx != chunk.block_index:
                new_blocks.append(block.model_copy(deep=True))
                continue
            if chunk.sentence_index is None:
                # Drop the whole block.
                continue
            # Sentence ablation: rewrite the text block with the named
            # sentence removed. Other blocks in the message stay intact.
            sentences = _split_sentences(block.text or "")
            kept = [
                sentence
                for sent_idx, sentence in enumerate(sentences)
                if sent_idx != chunk.sentence_index
            ]
            new_text = " ".join(kept)
            new_blocks.append(
                ContentBlock(
                    type="text",
                    text=new_text,
                    cache=block.cache,
                )
            )
        out.append(Message(role=message.role, content=new_blocks))
    return out


async def attribute(
    session: SessionRecord,
    target_message_index: int,
    runner: Runner,
    agent: SubAgent,
    granularity: Granularity = "message",
    similarity: Similarity | None = None,
    *,
    estimate_only: bool = False,
    cache: InMemoryAttributionCache | None = None,
) -> AttributionResult:
    """Run leave-one-out ablation and rank chunks by causal influence.

    `target_message_index` may be negative; it is normalized against the
    length of `session.messages`. The target's text is read directly from
    the record; we do not re-run the original session.

    For each chunk strictly before the target, we build an ablated message
    list and call `runner(agent, ablated)`. The score is
    `1 - similarity(original_target_text, ablated_response_text)`. The
    cache is consulted before any runner call; the same `(ablated_hash,
    target_index)` returns the same cached response forever.

    With `estimate_only=True` no runner calls are made; the result carries
    `estimated_calls = N` and chunks with `score = 0.0`.
    """
    target = _normalize_target(target_message_index, len(session.messages))
    if target < 0 or target >= len(session.messages):
        raise IndexError(
            f"target_message_index {target_message_index} is out of range "
            f"for a session with {len(session.messages)} messages"
        )

    target_text = _extract_text(session.messages[target])

    refs = chunk_session(session, granularity, target_message_index=target)

    if estimate_only:
        return AttributionResult(
            target_response=target_text,
            chunks=[
                AttributionChunk(
                    message_index=ref.message_index,
                    block_index=ref.block_index,
                    sentence_index=ref.sentence_index,
                    score=0.0,
                    preview=_preview(ref.text),
                )
                for ref in refs
            ],
            estimated_calls=len(refs),
            actual_calls=0,
        )

    sim: Similarity = similarity if similarity is not None else JaccardSimilarity()
    cache_obj = cache if cache is not None else InMemoryAttributionCache()

    chunks_out: list[AttributionChunk] = []
    actual_calls = 0
    prefix_messages = session.messages[:target]

    for ref in refs:
        ablated = _ablate_messages(prefix_messages, ref)
        key = (hash_messages(ablated), target)
        cached = cache_obj.get(key)
        if cached is None:
            response = await runner(agent, ablated)
            cache_obj.put(key, response)
            actual_calls += 1
        else:
            response = cached
        ablated_text = _extract_text(response)
        score = 1.0 - sim(target_text, ablated_text)
        chunks_out.append(
            AttributionChunk(
                message_index=ref.message_index,
                block_index=ref.block_index,
                sentence_index=ref.sentence_index,
                score=score,
                preview=_preview(ref.text),
            )
        )

    return AttributionResult(
        target_response=target_text,
        chunks=chunks_out,
        estimated_calls=len(refs),
        actual_calls=actual_calls,
    )


__all__ = [
    "AttributionChunk",
    "AttributionResult",
    "Granularity",
    "attribute",
    "chunk_session",
]
