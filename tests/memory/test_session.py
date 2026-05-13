from __future__ import annotations

import pytest

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookDecision, HookRunner, PromptSubmit
from harness.memory import InMemoryStore, PromptBlocked, Session, SessionNotFound
from harness.prompts import Message, text
from harness.prompts.messages import ContentBlock
from harness.tools import Dispatcher


def make_orchestrator(reply_with: str = "ok") -> Orchestrator:
    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", reply_with)

    return Orchestrator(Dispatcher(), HookRunner(), fake_runner)


def make_orchestrator_with_hooks(
    hooks: HookRunner,
    reply_with: str = "ok",
) -> Orchestrator:
    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", reply_with)

    return Orchestrator(Dispatcher(), hooks, fake_runner)


def make_agent() -> SubAgent:
    return SubAgent(name="bot", system_prompt="be helpful", model="test-model")


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


# ---------------------------------------------------------------------------
# M2.6: Session.send emits PromptSubmit through the orchestrator's hook runner.


async def test_send_emits_prompt_submit_with_user_text() -> None:
    """`Session.send("hello")` fires exactly one `PromptSubmit` event whose
    `prompt` attribute holds the user text. This is the boundary that lets
    a `forbid` contract refuse a prompt before the runner is called.
    """
    seen: list[PromptSubmit] = []
    hooks = HookRunner()
    hooks.register(PromptSubmit, lambda e: seen.append(e))

    orch = make_orchestrator_with_hooks(hooks, "hi there")
    sess = Session(orch, make_agent(), InMemoryStore())

    reply = await sess.send("hello")
    assert reply.content[0].text == "hi there"
    assert len(seen) == 1
    assert seen[0].prompt == "hello"


async def test_send_with_message_object_concatenates_text_blocks_for_prompt() -> None:
    """Multi-block `Message` inputs flatten to newline-joined text for
    `PromptSubmit.prompt`. Non-text blocks contribute nothing.
    """
    seen: list[PromptSubmit] = []
    hooks = HookRunner()
    hooks.register(PromptSubmit, lambda e: seen.append(e))

    orch = make_orchestrator_with_hooks(hooks)
    sess = Session(orch, make_agent(), InMemoryStore())

    msg = Message(
        role="user",
        content=[
            ContentBlock(type="text", text="first"),
            ContentBlock(type="text", text="second"),
        ],
    )
    await sess.send(msg)
    assert len(seen) == 1
    assert seen[0].prompt == "first\nsecond"


async def test_send_raises_prompt_blocked_when_handler_returns_block() -> None:
    """A `PromptSubmit` handler returning `HookDecision(block=True)` causes
    `Session.send` to raise `PromptBlocked` and skip the orchestrator entirely.
    """
    runner_calls: list[int] = []
    hooks = HookRunner()

    def block_all(_event: PromptSubmit) -> HookDecision:
        return HookDecision(block=True, reason="blocked-for-test")

    hooks.register(PromptSubmit, block_all)

    async def runner_that_records(_agent: SubAgent, _messages: list[Message]) -> Message:
        runner_calls.append(1)
        return text("assistant", "should not be reached")

    orch = Orchestrator(Dispatcher(), hooks, runner_that_records)
    store = InMemoryStore()
    sess = Session(orch, make_agent(), store)

    with pytest.raises(PromptBlocked) as excinfo:
        await sess.send("trigger")
    assert excinfo.value.reason == "blocked-for-test"
    assert runner_calls == []
    # The user message is in in-memory history (so the caller can inspect what
    # was rejected) but nothing was persisted to the store.
    assert [m.role for m in sess.messages] == ["user"]
    assert await store.load(sess.session_id) is None


async def test_send_proceeds_when_handler_returns_non_blocking_decision() -> None:
    """A `PromptSubmit` handler returning `HookDecision(block=False)` (or
    `None`) does not prevent the orchestrator from running.
    """
    hooks = HookRunner()
    hooks.register(PromptSubmit, lambda _e: HookDecision(block=False))
    orch = make_orchestrator_with_hooks(hooks, "ok")
    sess = Session(orch, make_agent(), InMemoryStore())

    reply = await sess.send("hi")
    assert reply.content[0].text == "ok"


async def test_prompt_submit_fires_before_runner_runs() -> None:
    """Ordering guarantee: `PromptSubmit` handlers see the prompt text
    *before* the runner is invoked, so a forbid contract can short-circuit
    the model round-trip.
    """
    order: list[str] = []
    hooks = HookRunner()

    def on_prompt(_event: PromptSubmit) -> None:
        order.append("prompt")

    hooks.register(PromptSubmit, on_prompt)

    async def runner(_agent: SubAgent, _messages: list[Message]) -> Message:
        order.append("runner")
        return text("assistant", "ok")

    orch = Orchestrator(Dispatcher(), hooks, runner)
    sess = Session(orch, make_agent(), InMemoryStore())

    await sess.send("hello")
    assert order == ["prompt", "runner"]
