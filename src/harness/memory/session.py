from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from harness.agents.definition import SubAgent
from harness.agents.orchestrator import Orchestrator
from harness.hooks.events import PromptSubmit
from harness.memory.record import SessionNotFound, SessionRecord
from harness.memory.store import MemoryStore
from harness.prompts.messages import Message, text


class PromptBlocked(Exception):
    """Raised by `Session.send` when a `PromptSubmit` hook handler blocks.

    Any `HookDecision(block=True)` returned by a `PromptSubmit` handler — for
    example, a `forbid` contract attached via `attach_contracts` matching the
    user text — causes `Session.send` to raise this exception *before* the
    orchestrator is invoked. The runner never sees the offending prompt.

    Attributes:
        reason: The `reason` field from the first blocking decision, or `None`
            if the handler returned `HookDecision(block=True, reason=None)`.
    """

    def __init__(self, reason: str | None) -> None:
        super().__init__(reason or "PromptSubmit was blocked by a hook handler")
        self.reason = reason


class Session:
    """A multi-turn conversation backed by a `MemoryStore`.

    Each `send()` accumulates a user message, emits a `PromptSubmit` event
    through the orchestrator's `HookRunner` (so any registered contracts /
    policies see the user text before the runner does), runs the orchestrator,
    appends the assistant reply to the in-memory history, and saves a fresh
    `SessionRecord` to the store.

    A `HookDecision(block=True)` returned at `PromptSubmit` raises
    `PromptBlocked` before the orchestrator is invoked — the user message has
    already been appended to in-memory history (so the caller can inspect what
    was rejected) but no `SessionRecord` is persisted.

    Single-writer per session_id. Two concurrent `Session.restore(same_id)`
    instances racing `send()` is last-writer-wins — the second save silently
    overwrites the first. Treat each `Session` as owned by a single coroutine.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        agent: SubAgent,
        store: MemoryStore,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._orch = orchestrator
        self._agent = agent
        self._store = store
        self._session_id = session_id if session_id is not None else uuid4().hex
        self._metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        self._messages: list[Message] = []
        self._created_at = datetime.now(UTC)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    async def send(self, message: str | Message) -> Message:
        msg = text("user", message) if isinstance(message, str) else message
        self._messages.append(msg)
        prompt_text = message if isinstance(message, str) else _extract_prompt_text(msg)
        decisions = await self._orch.hooks.emit(PromptSubmit(prompt=prompt_text))
        for decision in decisions:
            if decision.block:
                raise PromptBlocked(decision.reason)
        reply = await self._orch.run(self._agent, self._messages)
        self._messages.append(reply)
        await self._store.save(self._to_record())
        return reply

    @classmethod
    async def restore(
        cls,
        session_id: str,
        store: MemoryStore,
        orchestrator: Orchestrator,
    ) -> Session:
        record = await store.load(session_id)
        if record is None:
            raise SessionNotFound(session_id)
        s = cls(
            orchestrator,
            record.agent,
            store,
            session_id=session_id,
            metadata=record.metadata,
        )
        s._messages = list(record.messages)
        s._created_at = record.created_at
        return s

    def _to_record(self) -> SessionRecord:
        return SessionRecord(
            session_id=self._session_id,
            agent=self._agent,
            messages=list(self._messages),
            metadata=dict(self._metadata),
            created_at=self._created_at,
            updated_at=datetime.now(UTC),
        )


def _extract_prompt_text(message: Message) -> str:
    """Concatenate the text blocks of a `Message` for `PromptSubmit.prompt`.

    `PromptSubmit.prompt` is a `str`; when `Session.send` is called with a
    `Message` it may carry multiple `ContentBlock`s. Non-text blocks (image,
    file, tool_use, tool_result) contribute no text; if no text blocks exist
    the prompt is the empty string. Multiple text blocks join with newlines.
    """
    parts = [block.text for block in message.content if block.type == "text" and block.text]
    return "\n".join(parts)
