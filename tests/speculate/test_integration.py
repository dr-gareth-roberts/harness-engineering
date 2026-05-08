"""End-to-end integration: Speculator + AnthropicRunner via the fake SDK.

Pins the public flow that callers will actually use: a real Speculator
wired into a real AnthropicRunner, with a real Dispatcher carrying a
real Tool. The fake Anthropic client emits a canned tool_use response
that matches the predictor's prediction — so the speculator's task
result is what the model's tool_result block carries, not a fresh
dispatch.

This is doc test #10 from `designs/standout.md` §5.
"""

from __future__ import annotations

from pydantic import BaseModel

from harness.agents import SubAgent
from harness.hooks import HookRunner
from harness.prompts import text
from harness.runner.anthropic import AnthropicRunner
from harness.speculate import LastCallPredictor, Speculator
from harness.telemetry import MemorySink, Telemetry
from harness.tools import Dispatcher, Tool
from harness.tools.schema import ToolCall
from tests.runner.fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeTextBlock,
    FakeToolUseBlock,
)


class _Args(BaseModel):
    q: str = ""


def _agent() -> SubAgent:
    return SubAgent(
        name="t",
        system_prompt="",
        model="test-model",
        allowed_tools=["search"],
    )


def _build_dispatcher(call_log: list[str]) -> Dispatcher:
    def search_handler(args: _Args) -> str:
        call_log.append(args.q)
        return f"results-for-{args.q}"

    return Dispatcher(
        [
            Tool(
                name="search",
                description="",
                input_model=_Args,
                handler=search_handler,
                idempotent=True,
            )
        ]
    )


async def test_speculation_hit_skips_dispatcher_and_emits_hit_telemetry() -> None:
    """End-to-end: history shows a recent search('foo'); the model emits the
    same call again. Speculator predicts and pre-dispatches; on the model's
    tool_use block, try_resolve hits and the runner skips its own dispatch.
    """
    # Conversation history: one prior search('foo').
    history = [
        text("user", "find me foo"),
        # Synthesize an assistant tool_use turn so LastCallPredictor sees it.
        # Use the public Message + ContentBlock surface so we don't depend on
        # internal helpers.
    ]

    # The actual prior turn was an assistant tool_use — build it explicitly.
    from harness.prompts.messages import ContentBlock, Message

    history.append(
        Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name="search", arguments={"q": "foo"}, id="prev"),
                )
            ],
        )
    )
    history.append(
        Message(
            role="user",
            content=[ContentBlock(type="text", text="now search again")],
        )
    )

    # Fake SDK: model emits the *same* search('foo') call, then on the
    # follow-up turn returns end_turn.
    new_tool_use = FakeToolUseBlock(id="model-id", name="search", input={"q": "foo"})
    response_1 = FakeMessage(content=[new_tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="done")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])

    call_log: list[str] = []
    dispatcher = _build_dispatcher(call_log)
    sink = MemorySink()
    speculator = Speculator(
        LastCallPredictor(history_window=1),
        max_speculations=1,
        telemetry=Telemetry(sink=sink),
    )

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=speculator,
    )
    await runner(_agent(), history)

    # The dispatcher's search handler was called exactly once — by the
    # speculator's pre-dispatch, NOT by the runner re-dispatching after
    # the hit. Without speculation, this would have been twice (once by
    # speculator if any, plus once by the runner) — and without ANY
    # speculator, once by the runner alone. The signal is: the call log
    # has exactly one entry, with q="foo".
    assert call_log == ["foo"]

    # Telemetry confirms the hit.
    kinds = [type(e).__name__ for e in sink.events]
    assert "SpeculationLaunched" in kinds
    assert "SpeculationHit" in kinds
    # No misses on this happy path.
    assert "SpeculationMiss" not in kinds


async def test_speculation_miss_falls_back_and_dispatches_normally() -> None:
    """End-to-end miss: prior call was search('foo'), model now calls
    search('bar'). Speculator's pre-dispatch is wasted; runner falls back
    to its normal dispatch path."""
    from harness.prompts.messages import ContentBlock, Message

    history = [
        text("user", "find me foo"),
        Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name="search", arguments={"q": "foo"}, id="prev"),
                )
            ],
        ),
        Message(role="user", content=[ContentBlock(type="text", text="now search bar")]),
    ]

    # Model emits search('bar') — different args from the predicted
    # search('foo').
    new_tool_use = FakeToolUseBlock(id="model-id", name="search", input={"q": "bar"})
    response_1 = FakeMessage(content=[new_tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="done")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])

    call_log: list[str] = []
    dispatcher = _build_dispatcher(call_log)
    sink = MemorySink()
    speculator = Speculator(
        LastCallPredictor(history_window=1),
        max_speculations=1,
        telemetry=Telemetry(sink=sink),
    )

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=speculator,
    )
    await runner(_agent(), history)

    # The runner's normal dispatch path ran for search('bar'). The
    # speculation for search('foo') was scheduled at begin() but the
    # fake-SDK turn is synchronous (no real network wait), so the
    # speculation task never got a chance to actually execute before
    # end() cancelled it. In a real run with a network round-trip this
    # would be ~200ms of wasted work; in this test we observe just the
    # real call. Either way, the model's call resolved correctly.
    assert "bar" in call_log
    assert len(call_log) <= 2  # at most the speculation also ran

    # Telemetry confirms launch + miss (no hit).
    kinds = [type(e).__name__ for e in sink.events]
    assert "SpeculationLaunched" in kinds
    assert "SpeculationMiss" in kinds
    assert "SpeculationHit" not in kinds
