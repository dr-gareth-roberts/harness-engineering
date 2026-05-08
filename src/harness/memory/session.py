from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from harness.agents.definition import SubAgent
from harness.agents.orchestrator import Orchestrator
from harness.memory.record import SessionNotFound, SessionRecord
from harness.memory.store import MemoryStore
from harness.prompts.messages import Message, text


class Session:
    """A multi-turn conversation backed by a `MemoryStore`.

    Each `send()` accumulates a user message, runs the orchestrator, appends the
    assistant reply to the in-memory history, and saves a fresh `SessionRecord`
    to the store.

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
