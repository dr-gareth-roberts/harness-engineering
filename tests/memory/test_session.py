from __future__ import annotations

import pytest

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.memory import InMemoryStore, Session, SessionNotFound
from harness.prompts import Message, text
from harness.tools import Dispatcher


def make_orchestrator(reply_with: str = "ok") -> Orchestrator:
    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", reply_with)

    return Orchestrator(Dispatcher(), HookRunner(), fake_runner)


def make_agent() -> SubAgent:
    return SubAgent(name="bot", system_prompt="be helpful")


async def test_send_accumulates_messages_and_saves() -> None:
    store = InMemoryStore()
    sess = Session(make_orchestrator("hi there"), make_agent(), store)

    reply = await sess.send("hello")
    assert reply.content[0].text == "hi there"
    assert [m.role for m in sess.messages] == ["user", "assistant"]

    saved = await store.load(sess.session_id)
    assert saved is not None
    assert [m.content[0].text for m in saved.messages] == ["hello", "hi there"]


async def test_multiple_sends_accumulate() -> None:
    store = InMemoryStore()
    sess = Session(make_orchestrator("ack"), make_agent(), store)
    await sess.send("first")
    await sess.send("second")
    await sess.send("third")

    saved = await store.load(sess.session_id)
    assert saved is not None
    assert [m.content[0].text for m in saved.messages] == [
        "first",
        "ack",
        "second",
        "ack",
        "third",
        "ack",
    ]


async def test_restore_resumes_from_stored_state() -> None:
    store = InMemoryStore()
    orch = make_orchestrator("ack")
    sess = Session(orch, make_agent(), store, session_id="resume-me")
    await sess.send("hello")

    revived = await Session.restore("resume-me", store, orch)
    assert revived.session_id == "resume-me"
    assert [m.content[0].text for m in revived.messages] == ["hello", "ack"]

    await revived.send("world")
    saved = await store.load("resume-me")
    assert saved is not None
    assert [m.content[0].text for m in saved.messages] == [
        "hello",
        "ack",
        "world",
        "ack",
    ]


async def test_restore_missing_raises_session_not_found() -> None:
    store = InMemoryStore()
    orch = make_orchestrator()
    with pytest.raises(SessionNotFound):
        await Session.restore("does-not-exist", store, orch)


async def test_metadata_round_trips_through_restore() -> None:
    store = InMemoryStore()
    orch = make_orchestrator()
    sess = Session(
        orch,
        make_agent(),
        store,
        session_id="meta-test",
        metadata={"source": "cli", "tag": 7},
    )
    await sess.send("hi")

    revived = await Session.restore("meta-test", store, orch)
    assert revived.metadata == {"source": "cli", "tag": 7}


async def test_send_accepts_message_object_directly() -> None:
    store = InMemoryStore()
    orch = make_orchestrator("ok")
    sess = Session(orch, make_agent(), store)

    await sess.send(text("user", "from-object"))
    saved = await store.load(sess.session_id)
    assert saved is not None
    assert saved.messages[0].content[0].text == "from-object"
