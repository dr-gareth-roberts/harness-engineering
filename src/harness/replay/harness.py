from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from harness.agents.definition import SubAgent
from harness.agents.orchestrator import Orchestrator
from harness.memory.record import SessionRecord
from harness.memory.session import Session
from harness.memory.store import InMemoryStore
from harness.prompts.messages import ContentBlock, Message


@dataclass(frozen=True)
class EvalCase:
    """A named series of user prompts to run an agent through."""

    name: str
    prompts: list[str]


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    record: SessionRecord
    duration_ms: float


@dataclass(frozen=True)
class TurnDiff:
    index: int
    role: str | None
    matches: bool
    a_text: str | None
    b_text: str | None


@dataclass(frozen=True)
class SessionDiff:
    name: str
    matches: bool
    turns: list[TurnDiff]


async def run_eval(
    cases: list[EvalCase],
    *,
    orchestrator: Orchestrator,
    agent: SubAgent,
) -> list[EvalResult]:
    """Run each case as a fresh `Session` and capture its `SessionRecord`."""
    results: list[EvalResult] = []
    for case in cases:
        store = InMemoryStore()
        session = Session(orchestrator, agent, store)
        start = time.perf_counter()
        for prompt in case.prompts:
            await session.send(prompt)
        duration_ms = (time.perf_counter() - start) * 1000.0

        record = await store.load(session.session_id)
        if record is None:
            raise RuntimeError(
                f"session {session.session_id!r} vanished after run_eval call — "
                "store backing dropped state"
            )
        results.append(EvalResult(case=case, record=record, duration_ms=duration_ms))
    return results


def _normalize_message(msg: Message) -> tuple[str, str, tuple[Any, ...]]:
    """Reduce a message to a stable comparable shape.

    - Concatenates all `type="text"` blocks per message in encounter order
      (so two text blocks compare equal to one combined block — matches
      "what the user reads").
    - Drops `tool_use.id` and `tool_result.tool_use_id`; those are
      model-assigned and would diverge across runs even when behaviour is
      identical.
    """
    text_chunks: list[str] = []
    structured: list[Any] = []
    for block in msg.content:
        if block.type == "text":
            if block.text:
                text_chunks.append(block.text)
        elif block.type == "tool_use" and block.tool_use is not None:
            tu = block.tool_use
            structured.append(("tool_use", tu.name, _sorted_dict(tu.arguments)))
        elif block.type == "tool_result" and block.tool_result is not None:
            tr = block.tool_result
            structured.append(("tool_result", tr.content, tr.is_error))
        elif block.type == "file":
            structured.append(("file", block.path, block.text))
    return msg.role, "".join(text_chunks), tuple(structured)


def _sorted_dict(d: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(d.items(), key=lambda kv: kv[0]))


def _summary_text(msg: Message | None) -> str | None:
    if msg is None:
        return None
    parts = [b.text for b in msg.content if b.type == "text" and b.text]
    return "".join(parts) if parts else ""


def compare_sessions(a: SessionRecord, b: SessionRecord, *, name: str = "") -> SessionDiff:
    """Diff two session records turn-by-turn, ignoring tool-call IDs."""
    diffs: list[TurnDiff] = []
    all_match = True
    longest = max(len(a.messages), len(b.messages))
    for i in range(longest):
        a_msg = a.messages[i] if i < len(a.messages) else None
        b_msg = b.messages[i] if i < len(b.messages) else None

        if a_msg is None or b_msg is None:
            matches = False
            role = (a_msg or b_msg or _placeholder()).role
        else:
            matches = _normalize_message(a_msg) == _normalize_message(b_msg)
            role = a_msg.role
        if not matches:
            all_match = False
        diffs.append(
            TurnDiff(
                index=i,
                role=role,
                matches=matches,
                a_text=_summary_text(a_msg),
                b_text=_summary_text(b_msg),
            )
        )
    return SessionDiff(name=name, matches=all_match, turns=diffs)


def _placeholder() -> Message:
    return Message(role="user", content=[ContentBlock(type="text", text="")])
