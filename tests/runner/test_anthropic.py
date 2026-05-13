from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from pydantic import BaseModel

from harness.agents import SubAgent
from harness.hooks import HookRunner, PostToolUse, PreToolUse
from harness.policy import AllowList, attach_pre_tool_policies
from harness.prompts import Message, attach_file, text
from harness.prompts.messages import ContentBlock
from harness.runner.anthropic import (
    AnthropicRunner,
    _serialize_tool_content,
    _translate_in,
)
from harness.tools import Dispatcher, Tool
from harness.tools.schema import ToolCall, ToolResult
from tests.runner.fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeTextBlock,
    FakeToolUseBlock,
)

# ---------------------------------------------------------------------------
# helpers


class EchoIn(BaseModel):
    text: str


def _echo_dispatcher(*, calls: list[str] | None = None) -> tuple[Dispatcher, list[str]]:
    log: list[str] = calls if calls is not None else []

    def echo(args: EchoIn) -> str:
        log.append(args.text)
        return args.text

    return (
        Dispatcher(
            [Tool(name="echo", description="Echo it back.", input_model=EchoIn, handler=echo)]
        ),
        log,
    )


def _agent(*, allowed_tools: list[str] | None = None) -> SubAgent:
    return SubAgent(
        name="t",
        system_prompt="You are a small test agent.",
        model="claude-opus-4-7",
        allowed_tools=allowed_tools if allowed_tools is not None else ["echo"],
    )


# ---------------------------------------------------------------------------
# translation


def test_translate_in_extracts_system_messages() -> None:
    msgs: list[Message] = [
        text("system", "be helpful"),
        text("user", "hi"),
        text("system", "be terse"),
        text("user", "again"),
    ]
    api_messages, system = _translate_in(msgs)

    assert system == "be helpful\n\nbe terse"
    assert [m["role"] for m in api_messages] == ["user", "user"]
    assert api_messages[0]["content"] == [{"type": "text", "text": "hi"}]


def test_translate_in_propagates_cache_marker() -> None:
    msgs: list[Message] = [text("user", "cached", cache=True)]
    api_messages, _ = _translate_in(msgs)
    block = api_messages[0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}


def test_translate_in_renders_file_block_as_text(tmp_path: Any) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello")
    msgs: list[Message] = [
        Message(role="user", content=[attach_file(p)]),
    ]
    api_messages, _ = _translate_in(msgs)
    block = api_messages[0]["content"][0]
    assert block["type"] == "text"
    assert "<file path=" in block["text"]
    assert "hello" in block["text"]


def test_serialize_tool_content_handles_dict_and_list() -> None:
    assert _serialize_tool_content("plain") == "plain"
    assert _serialize_tool_content(42) == "42"
    assert _serialize_tool_content({"a": 1}) == '{"a": 1}'
    assert _serialize_tool_content([1, 2]) == "[1, 2]"


# ---------------------------------------------------------------------------
# loop behaviour


async def test_no_tool_happy_path() -> None:
    response = FakeMessage(
        content=[FakeTextBlock(text="hello there")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    result = await runner(_agent(), [text("user", "hi")])

    assert result.role == "assistant"
    assert len(result.content) == 1
    assert result.content[0].text == "hello there"


async def test_one_iteration_tool_loop_dispatches_and_continues() -> None:
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="echoed: hi")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, log = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    result = await runner(_agent(), [text("user", "echo hi")])
    assert log == ["hi"]
    assert result.content[0].text == "echoed: hi"

    # The second request must include the assistant's tool_use turn AND a
    # user turn carrying the tool_result.
    second = client.messages.requests[1]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["user", "assistant", "user"]
    tool_result_block = second["messages"][-1]["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["tool_use_id"] == "tu_1"
    assert tool_result_block["content"] == "hi"
    assert tool_result_block["is_error"] is False


async def test_hook_block_short_circuits_dispatch() -> None:
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, log = _echo_dispatcher()
    hooks = HookRunner()
    attach_pre_tool_policies(hooks, AllowList.of({"approved-only"}))
    runner = AnthropicRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]

    await runner(_agent(allowed_tools=["echo"]), [text("user", "go")])

    assert log == []  # dispatcher's handler never ran
    second = client.messages.requests[1]
    tool_result_block = second["messages"][-1]["content"][0]
    assert tool_result_block["is_error"] is True
    assert "echo" in tool_result_block["content"]


async def test_max_iterations_cap_raises() -> None:
    responses = [
        FakeMessage(
            content=[FakeToolUseBlock(id=f"tu_{i}", name="echo", input={"text": "x"})],
            stop_reason="tool_use",
        )
        for i in range(5)
    ]
    client = FakeAsyncAnthropic(responses=responses)
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        max_iterations=3,
    )

    with pytest.raises(RuntimeError, match="max_iterations=3"):
        await runner(_agent(), [text("user", "loop")])


async def test_unexpected_stop_reason_raises() -> None:
    """A genuinely unknown stop_reason still raises — pause_turn and
    refusal are now handled (surfaced via events), but anything else
    is a runner-level surprise and should fail loudly."""
    response = FakeMessage(content=[FakeTextBlock(text="...")], stop_reason="content_filter")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="content_filter"):
        await runner(_agent(), [text("user", "x")])


async def test_allowed_tools_filter_excludes_unlisted_tools() -> None:
    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    await runner(_agent(allowed_tools=[]), [text("user", "x")])
    request = client.messages.requests[0]
    assert "tools" not in request


async def test_request_includes_thinking_and_effort_when_set() -> None:
    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        thinking_mode="adaptive",
        effort="high",
    )

    await runner(_agent(), [text("user", "x")])
    request = client.messages.requests[0]
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "high"}


async def test_thinking_omitted_when_disabled() -> None:
    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        thinking_mode="disabled",
    )

    await runner(_agent(), [text("user", "x")])
    assert "thinking" not in client.messages.requests[0]


# ---------------------------------------------------------------------------
# missing dep


def test_missing_anthropic_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.delitem(sys.modules, "harness.runner.anthropic", raising=False)

    with pytest.raises(ImportError, match=r"harness-engineering-toolkit\[anthropic\]"):
        importlib.import_module("harness.runner.anthropic")


# ---------------------------------------------------------------------------
# PostAssistantMessage event emission (closes the contracts runtime parity gap)


async def test_runner_emits_post_assistant_message_on_terminal_iteration() -> None:
    """Pins that AnthropicRunner emits PostAssistantMessage when the model
    returns a final assistant turn. Closes the gap that no test in
    `tests/contracts` exercises — they all drive `attach_contracts` directly,
    not through a runner instance.
    """
    from harness.hooks import PostAssistantMessage

    response = FakeMessage(
        content=[FakeTextBlock(text="hello there")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    hooks = HookRunner()
    seen: list[Message] = []

    def capture(event: PostAssistantMessage) -> None:
        seen.append(event.message)

    hooks.register(PostAssistantMessage, capture)

    runner = AnthropicRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]
    result = await runner(_agent(), [text("user", "hi")])

    # One terminal assistant message → one event.
    assert len(seen) == 1
    assert seen[0] is result
    assert seen[0].content[0].text == "hello there"


async def test_runner_emits_post_assistant_message_on_each_loop_iteration() -> None:
    """Per-iteration emission: a tool-use loop with N iterations produces N
    PostAssistantMessage events — including intermediate text-plus-tool-use
    messages that never return to the orchestrator. Without per-iteration
    emission, contracts over assistant text would miss intermediate turns.
    """
    from harness.hooks import PostAssistantMessage

    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(
        content=[FakeTextBlock(text="I'll echo that"), tool_use],
        stop_reason="tool_use",
    )
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="echoed: hi")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, _ = _echo_dispatcher()
    hooks = HookRunner()
    seen: list[Message] = []

    def capture(event: PostAssistantMessage) -> None:
        seen.append(event.message)

    hooks.register(PostAssistantMessage, capture)

    runner = AnthropicRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]
    await runner(_agent(), [text("user", "echo hi")])

    # Two iterations → two events: the intermediate text-plus-tool-use
    # message and the terminal text-only message.
    assert len(seen) == 2
    iter1_text = "".join(b.text or "" for b in seen[0].content if b.type == "text")
    iter2_text = "".join(b.text or "" for b in seen[1].content if b.type == "text")
    assert iter1_text == "I'll echo that"
    assert iter2_text == "echoed: hi"
    # Intermediate message also carried the tool_use block.
    assert any(b.type == "tool_use" for b in seen[0].content)


# ---------------------------------------------------------------------------
# Speculator wiring (Wave 3 Phase 1)


class _StubSpeculator:
    """Test stub satisfying SpeculatorProtocol.

    Records every call. Configurable per-call hit/miss via the `hits`
    dict (call.name -> ToolResult) so tests can pin both branches.
    """

    def __init__(self, hits: dict[str, ToolResult] | None = None) -> None:
        self.hits = dict(hits or {})
        self.begin_calls: list[dict[str, object]] = []
        self.observe_calls: list[ToolCall] = []
        self.cancel_unobserved_calls = 0
        self.try_resolve_calls: list[ToolCall] = []
        self.end_calls = 0

    async def begin(
        self,
        *,
        history: list[Message],
        agent: SubAgent,
        dispatcher: Dispatcher,
        hooks: HookRunner,
    ) -> None:
        self.begin_calls.append({"history_len": len(history), "agent_name": agent.name})

    async def observe(self, call: ToolCall) -> None:
        self.observe_calls.append(call)

    async def cancel_unobserved(self) -> None:
        self.cancel_unobserved_calls += 1

    async def try_resolve(self, call: ToolCall) -> ToolResult | None:
        self.try_resolve_calls.append(call)
        return self.hits.get(call.name)

    async def end(self) -> None:
        self.end_calls += 1


async def test_speculator_begin_and_end_fire_per_iteration() -> None:
    response = FakeMessage(
        content=[FakeTextBlock(text="hi")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    spec = _StubSpeculator()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "hello")])

    # One iteration → one begin/end pair.
    assert len(spec.begin_calls) == 1
    assert spec.end_calls == 1
    # No tool_use blocks → try_resolve never consulted.
    assert spec.try_resolve_calls == []


async def test_speculator_hit_skips_runner_dispatch_and_hooks() -> None:
    """A try_resolve hit means the speculator already fired PreToolUse /
    dispatch / PostToolUse around its own dispatch. The runner must not
    repeat any of that for this call."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="done")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()

    cached = ToolResult(id="tu_1", content="(from speculation)", is_error=False)
    spec = _StubSpeculator(hits={"echo": cached})

    pre_calls: list[ToolCall] = []
    post_calls: list[tuple[ToolCall, ToolResult]] = []
    hooks = HookRunner()
    hooks.register(PreToolUse, lambda e: pre_calls.append(e.call))
    hooks.register(
        PostToolUse,
        lambda e: post_calls.append((e.call, e.result)),
    )

    runner = AnthropicRunner(
        dispatcher,
        hooks,
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "echo hi")])

    # Speculator was consulted with the model's call.
    assert len(spec.try_resolve_calls) == 1
    assert spec.try_resolve_calls[0].name == "echo"

    # Runner did NOT fire its own hooks for the speculatively-resolved call.
    assert pre_calls == []
    assert post_calls == []

    # And it did NOT touch the dispatcher itself.
    assert dispatch_log == []

    # Two iterations → two begin/end pairs.
    assert len(spec.begin_calls) == 2
    assert spec.end_calls == 2


async def test_speculator_miss_falls_back_to_runner_hooks_and_dispatch() -> None:
    """When try_resolve returns None, the runner takes over: PreToolUse,
    dispatch, PostToolUse — same as if no speculator were configured."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="echoed: hi")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()
    spec = _StubSpeculator()  # no hits configured -> always miss

    pre_calls: list[ToolCall] = []
    post_calls: list[tuple[ToolCall, ToolResult]] = []
    hooks = HookRunner()
    hooks.register(PreToolUse, lambda e: pre_calls.append(e.call))
    hooks.register(
        PostToolUse,
        lambda e: post_calls.append((e.call, e.result)),
    )

    runner = AnthropicRunner(
        dispatcher,
        hooks,
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    result = await runner(_agent(), [text("user", "echo hi")])

    # Try_resolve was consulted, returned None.
    assert len(spec.try_resolve_calls) == 1

    # Runner ran the full hook + dispatch cycle.
    assert len(pre_calls) == 1
    assert pre_calls[0].name == "echo"
    assert len(post_calls) == 1
    assert dispatch_log == ["hi"]  # echo handler logs the text it echoed

    # End-to-end output unchanged from the no-speculator case.
    assert result.content[0].text == "echoed: hi"


async def test_speculator_end_fires_even_when_runner_raises() -> None:
    """`begin` and `end` must be paired. If the SDK call (or anything else
    in the iteration) raises, end still fires so the speculator can clean up
    its background tasks."""
    # Use a genuinely unhandled stop_reason — `refusal` is now a known
    # reason that surfaces as an event (no exception), so it wouldn't
    # exercise the begin/end-on-error path this test pins.
    bad_response = FakeMessage(content=[], stop_reason="content_filter")
    client = FakeAsyncAnthropic(responses=[bad_response])
    dispatcher, _ = _echo_dispatcher()
    spec = _StubSpeculator()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    with pytest.raises(RuntimeError, match=r"Unexpected stop_reason"):
        await runner(_agent(), [text("user", "hi")])

    # begin fired, end fired even though the iteration aborted.
    assert len(spec.begin_calls) == 1
    assert spec.end_calls == 1


async def test_running_history_passed_to_speculator_grows_per_iteration() -> None:
    """`begin.history` must reflect the in-loop turns the caller never sees,
    so the predictor can use them to pick its next speculation."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(
        content=[FakeTextBlock(text="echoed: hi")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, _ = _echo_dispatcher()
    spec = _StubSpeculator()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "echo hi")])

    # Iteration 1 sees just the user input.
    assert spec.begin_calls[0]["history_len"] == 1
    # Iteration 2 sees user + assistant(tool_use) + user(tool_result).
    assert spec.begin_calls[1]["history_len"] == 3


# ---------------------------------------------------------------------------
# Wave 6: per-event speculator surfacing


async def test_runner_calls_observe_for_each_tool_use_block_in_stream() -> None:
    """The runner iterates the stream's content_block_stop events and
    surfaces each tool_use block to `speculator.observe` before the
    stream finishes. With two tool_use blocks in one response, observe
    must fire twice — once per block."""
    tool_use_1 = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hello"})
    tool_use_2 = FakeToolUseBlock(id="tu_2", name="echo", input={"text": "world"})
    response_1 = FakeMessage(
        content=[tool_use_1, tool_use_2],
        stop_reason="tool_use",
    )
    response_2 = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, _ = _echo_dispatcher()
    spec = _StubSpeculator()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "echo")])

    # observe fired twice in iteration 1 (one per tool_use block) and
    # zero times in iteration 2 (which has only a text block).
    assert [c.id for c in spec.observe_calls] == ["tu_1", "tu_2"]
    # cancel_unobserved fires once per iteration, regardless of content.
    assert spec.cancel_unobserved_calls == 2


async def test_runner_does_not_observe_text_block_stop_events() -> None:
    """`observe` is for tool_use blocks only — text block stops MUST NOT
    surface to the speculator. Otherwise a chatty response with no tools
    would spam observe calls and confuse the matching logic."""
    response = FakeMessage(
        content=[FakeTextBlock(text="just chatting"), FakeTextBlock(text="more text")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    spec = _StubSpeculator()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "say something")])

    assert spec.observe_calls == []
    # cancel_unobserved still fires after the stream — the speculator
    # decides what to do with it (most likely cancel everything).
    assert spec.cancel_unobserved_calls == 1


async def test_runner_with_speculator_none_iterates_stream_without_error() -> None:
    """When no speculator is configured, the runner still iterates the
    event stream (otherwise the fake's get_final_message wouldn't drive
    accumulation deterministically). It just doesn't observe."""
    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(content=[FakeTextBlock(text="done")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        # speculator=None
    )
    result = await runner(_agent(), [text("user", "echo hi")])

    # Normal dispatch happened — runner didn't trip over event iteration.
    assert dispatch_log == ["hi"]
    assert isinstance(result, Message)


async def test_runner_explicit_events_drive_observe_in_order() -> None:
    """Stream events can be scripted explicitly via `FakeMessage.events`
    to drive a specific arrival order. Pin that the runner observes in
    arrival order, not content-list order."""
    from tests.runner.fakes import FakeContentBlockStopEvent

    tool_use_1 = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "first"})
    tool_use_2 = FakeToolUseBlock(id="tu_2", name="echo", input={"text": "second"})
    text_block = FakeTextBlock(text="thinking")

    # content list order: text, tool_1, tool_2
    # events arrival order: tool_2, text, tool_1 (deliberately scrambled
    # to prove the runner uses event order, not content order)
    response_1 = FakeMessage(
        content=[text_block, tool_use_1, tool_use_2],
        stop_reason="tool_use",
        events=[
            FakeContentBlockStopEvent(index=2, content_block=tool_use_2),
            FakeContentBlockStopEvent(index=0, content_block=text_block),
            FakeContentBlockStopEvent(index=1, content_block=tool_use_1),
        ],
    )
    response_2 = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, _ = _echo_dispatcher()
    spec = _StubSpeculator()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=spec,
    )
    await runner(_agent(), [text("user", "echo")])

    # Order matches arrival, not content order. Text block does NOT
    # surface to observe.
    assert [c.id for c in spec.observe_calls] == ["tu_2", "tu_1"]


async def test_unobserved_speculation_does_not_complete_when_dispatch_diverges() -> None:
    """Wave 6's correctness claim: when the model emits a tool_use that
    no speculation predicted, the runner cancels the speculation via
    `cancel_unobserved` so its handler never runs to completion. The
    runner then dispatches the model's actual call normally.

    Stream-end cancellation can fire fast enough that the speculation
    task hasn't even started yet — that's *more* than the perf claim
    promises (zero handler runtime instead of partial). So we don't
    pin a strict start→cancel→run ordering; we only pin: predicted
    handler did NOT finish, actual handler DID run. The speculator
    unit test `test_cancel_unobserved_runs_fast_when_handler_is_slow`
    is the place that pins the timing claim directly.
    """
    import asyncio

    from harness.speculate import Speculator
    from harness.speculate.predictor import Predictor

    events: list[str] = []

    async def predicted_handler(args: EchoIn) -> str:
        events.append("predicted:start")
        try:
            await asyncio.sleep(10.0)  # very slow — we never want this to win
            events.append("predicted:done")
            return "predicted-done"
        except asyncio.CancelledError:
            events.append("predicted:cancelled")
            raise

    async def actual_handler(args: EchoIn) -> str:
        events.append("actual:run")
        return f"actual-done-{args.text}"

    dispatcher = Dispatcher(
        [
            Tool(
                name="predicted_tool",
                description="",
                input_model=EchoIn,
                handler=predicted_handler,
                idempotent=True,
            ),
            Tool(
                name="actual_tool",
                description="",
                input_model=EchoIn,
                handler=actual_handler,
                idempotent=True,
            ),
        ]
    )

    class FixedPredictor:
        """Always predicts the slow predicted_tool — the model will
        emit actual_tool instead, making this a guaranteed miss."""

        def predict(
            self,
            history: list[Message],
            idempotent_tools: dict[str, Tool],
            max_predictions: int,
        ) -> list[ToolCall]:
            return [ToolCall(name="predicted_tool", arguments={"text": "x"})]

    predictor: Predictor = FixedPredictor()
    speculator = Speculator(predictor, max_speculations=1)

    # Model emits actual_tool, not predicted_tool.
    tool_use = FakeToolUseBlock(id="tu_1", name="actual_tool", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(content=[FakeTextBlock(text="done")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response_1, response_2])

    agent = SubAgent(
        name="t",
        system_prompt="",
        model="test-model",
        allowed_tools=["predicted_tool", "actual_tool"],
    )

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        speculator=speculator,
    )
    await runner(agent, [text("user", "do it")])

    # Correctness claim: actual_tool's handler ran, predicted_tool's
    # handler never finished. The "predicted:start" marker is allowed
    # but optional — cancellation may fire before the task scheduler
    # has given the speculation any CPU time at all (the strongest
    # version of the win).
    assert "actual:run" in events
    assert "predicted:done" not in events


# ---------------------------------------------------------------------------
# Wave 10 #12: cache-breakpoint cap


def test_count_cache_breakpoints_walks_message_content() -> None:
    """Pin the counting helper directly — independent of the runner so we
    can iterate on the API shape without re-running the full integration."""
    from harness.runner.anthropic import _count_cache_breakpoints

    request = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "b"},
                    {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}},
                ],
            },
        ],
    }
    assert _count_cache_breakpoints(request) == 2


def test_count_cache_breakpoints_tolerates_missing_or_string_content() -> None:
    """Non-list `content` (string-shaped messages, missing field) shouldn't
    crash the counter — it should just contribute zero."""
    from harness.runner.anthropic import _count_cache_breakpoints

    request = {
        "messages": [
            {"role": "system", "content": "string-shaped"},
            {"role": "user"},  # no content field at all
        ],
    }
    assert _count_cache_breakpoints(request) == 0


async def test_runner_raises_cache_breakpoint_limit_exceeded_at_five() -> None:
    """Five cache markers across user messages — runner raises BEFORE
    making the SDK call so the caller gets a typed error, not an API 400."""
    from harness.runner.anthropic import CacheBreakpointLimitExceeded

    response = FakeMessage(
        content=[FakeTextBlock(text="ok")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
    )

    # Five separate user messages, each carrying one cache_control marker.
    messages = [
        Message(
            role="user",
            content=[ContentBlock(type="text", text=f"chunk-{i}", cache=True)],
        )
        for i in range(5)
    ]

    with pytest.raises(CacheBreakpointLimitExceeded, match="caps cache breakpoints at 4"):
        await runner(_agent(), messages)

    # SDK call never happened — fake's recorded requests stay empty.
    assert client.messages.requests == []


async def test_runner_accepts_exactly_four_cache_breakpoints() -> None:
    """Boundary case: 4 markers (the documented cap) is allowed."""
    response = FakeMessage(
        content=[FakeTextBlock(text="ok")],
        stop_reason="end_turn",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
    )

    messages = [
        Message(
            role="user",
            content=[ContentBlock(type="text", text=f"chunk-{i}", cache=True)],
        )
        for i in range(4)
    ]

    # Should not raise.
    result = await runner(_agent(), messages)
    assert isinstance(result, Message)


# ---------------------------------------------------------------------------
# Wave 10 #6: per-iteration timeout


async def test_runner_timeout_raises_when_stream_takes_too_long() -> None:
    """timeout_s wraps the stream's __aenter__ in asyncio.wait_for. A
    fake stream that sleeps longer than the timeout raises TimeoutError."""
    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    # 200ms enter delay vs 50ms timeout — timeout wins.
    client = FakeAsyncAnthropic(responses=[response], enter_delay=0.2)
    dispatcher, _ = _echo_dispatcher()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        timeout_s=0.05,
    )

    with pytest.raises(TimeoutError):
        await runner(_agent(), [text("user", "hello")])


async def test_runner_no_timeout_completes_normally_with_slow_stream() -> None:
    """timeout_s=None (default) lets a slow stream finish — no wait_for wrap."""
    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response], enter_delay=0.05)
    dispatcher, _ = _echo_dispatcher()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        # timeout_s=None
    )

    result = await runner(_agent(), [text("user", "hello")])
    assert isinstance(result, Message)


# ---------------------------------------------------------------------------
# Wave 10 #5: HookDecision.replacement


async def test_pre_tool_use_replacement_skips_dispatch_and_uses_supplied_result() -> None:
    """A PreToolUse hook returning HookDecision(replacement=ToolResult(...))
    short-circuits the runner's dispatch — the supplied result goes back
    to the model with the model's tool_use.id patched in."""
    from harness.hooks.events import HookDecision

    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "hi"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(content=[FakeTextBlock(text="done")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()

    hooks = HookRunner()
    hooks.register(
        PreToolUse,
        lambda e: HookDecision(replacement=ToolResult(content="injected", is_error=False)),
    )

    runner = AnthropicRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]
    await runner(_agent(), [text("user", "echo hi")])

    # Dispatcher's handler never ran — replacement short-circuited.
    assert dispatch_log == []
    # The injected result was sent back to the model with id=tu_1.
    second = client.messages.requests[1]
    tool_result_block = second["messages"][-1]["content"][0]
    assert tool_result_block["tool_use_id"] == "tu_1"
    assert tool_result_block["content"] == "injected"
    assert tool_result_block["is_error"] is False


async def test_post_tool_use_replacement_rewrites_result_before_model_sees_it() -> None:
    """A PostToolUse hook returning HookDecision(replacement=ToolResult(...))
    rewrites the dispatched result. Typical use is sanitization."""
    from harness.hooks.events import HookDecision

    tool_use = FakeToolUseBlock(id="tu_1", name="echo", input={"text": "secret-data"})
    response_1 = FakeMessage(content=[tool_use], stop_reason="tool_use")
    response_2 = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response_1, response_2])
    dispatcher, dispatch_log = _echo_dispatcher()

    hooks = HookRunner()
    hooks.register(
        PostToolUse,
        lambda e: HookDecision(
            replacement=ToolResult(content="[REDACTED]", is_error=False),
        ),
    )

    runner = AnthropicRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]
    await runner(_agent(), [text("user", "echo")])

    # Dispatcher DID run (we wanted the side effect).
    assert dispatch_log == ["secret-data"]
    # But the model sees the redacted version, not the raw output.
    second = client.messages.requests[1]
    tool_result_block = second["messages"][-1]["content"][0]
    assert tool_result_block["content"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# Wave 10 #4: pause_turn / refusal as events


async def test_pause_turn_emits_event_and_returns_partial_message() -> None:
    """A stop_reason of `pause_turn` no longer raises. The runner emits a
    PauseTurn event and returns the partial assistant message; the caller
    can re-invoke with this message in history to resume."""
    from harness.hooks import PauseTurn

    response = FakeMessage(
        content=[FakeTextBlock(text="working on it")],
        stop_reason="pause_turn",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()

    seen: list[PauseTurn] = []
    hooks = HookRunner()
    hooks.register(PauseTurn, lambda e: seen.append(e))

    runner = AnthropicRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]
    result = await runner(_agent(), [text("user", "do a long thing")])

    assert isinstance(result, Message)
    assert result.content[0].text == "working on it"
    assert len(seen) == 1
    assert seen[0].message is result
    assert seen[0].reason == "pause_turn"


async def test_refusal_emits_event_and_returns_refusal_message() -> None:
    """A stop_reason of `refusal` no longer raises. The runner emits a
    Refusal event and returns the refusal-only assistant message."""
    from harness.hooks import Refusal

    response = FakeMessage(
        content=[FakeTextBlock(text="I can't help with that.")],
        stop_reason="refusal",
    )
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()

    seen: list[Refusal] = []
    hooks = HookRunner()
    hooks.register(Refusal, lambda e: seen.append(e))

    runner = AnthropicRunner(dispatcher, hooks, client=client)  # type: ignore[arg-type]
    result = await runner(_agent(), [text("user", "do a thing")])

    assert isinstance(result, Message)
    assert "can't help" in (result.content[0].text or "")
    assert len(seen) == 1
    assert seen[0].message is result


# ---------------------------------------------------------------------------
# M1.11: teardown-timeout no longer silently swallowed


async def test_teardown_timeout_logs_and_propagates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the stream's `__aexit__` exceeds `timeout_s`, the runner logs
    at WARNING and propagates TimeoutError rather than swallowing it.

    A swallowed teardown timeout would let the next request inherit a
    connection in an indeterminate state — that's the bug M1.11 fixes.
    """
    import logging

    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    # `__aenter__` and iteration are fast; `__aexit__` sleeps past the
    # 50ms budget. The runner finishes `get_final_message` cleanly and
    # then hits the timeout on teardown.
    client = FakeAsyncAnthropic(responses=[response], exit_delay=0.2)
    dispatcher, _ = _echo_dispatcher()

    runner = AnthropicRunner(
        dispatcher,
        HookRunner(),
        client=client,  # type: ignore[arg-type]
        timeout_s=0.05,
    )

    with (
        caplog.at_level(logging.WARNING, logger="harness.runner.anthropic"),
        pytest.raises(TimeoutError),
    ):
        await runner(_agent(), [text("user", "hi")])

    # Logger emitted a WARNING that names the timeout and connection
    # state — operators reading the log learn why the request failed.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("teardown" in r.message for r in warnings)
    assert any("timeout_s" in r.message for r in warnings)


# ---------------------------------------------------------------------------
# M1.26: lazy client construction


def test_runner_instantiates_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`AnthropicRunner(...)` with no `client=` and no API key in the
    environment must not raise — the SDK client is constructed lazily
    on first call. Pre-M1.26 this raised at __init__.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    dispatcher, _ = _echo_dispatcher()
    # Should not raise even without an API key configured.
    runner = AnthropicRunner(dispatcher, HookRunner())
    # _client stays None until first access.
    assert runner._client is None


def test_runner_uses_injected_client_without_constructing_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-constructed `client=` bypasses the SDK factory entirely.
    Verify by removing the env vars the SDK would read and confirming
    the runner still drives a call through the injected fake.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    response = FakeMessage(content=[FakeTextBlock(text="ok")], stop_reason="end_turn")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    # The fake client was wired in at __init__; no env-driven SDK
    # construction should ever happen for this runner. (The `client=`
    # kwarg already carries a `# type: ignore[arg-type]` because the
    # fake doesn't subclass AsyncAnthropic; the identity check here
    # inherits the same narrowing gap.)
    assert runner._client is client  # type: ignore[comparison-overlap]


async def test_lazy_client_property_caches_after_first_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `client` property constructs `AsyncAnthropic()` lazily and
    caches the instance. Two accesses return the same object.
    """
    sentinel = object()
    call_count = 0

    def fake_factory() -> Any:
        nonlocal call_count
        call_count += 1
        return sentinel

    # Patch the AsyncAnthropic symbol the runner module sees so we can
    # observe construction without needing a real API key.
    monkeypatch.setattr("harness.runner.anthropic.AsyncAnthropic", fake_factory)

    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner())
    assert runner._client is None
    first = runner.client
    second = runner.client
    assert first is sentinel
    assert second is sentinel
    assert call_count == 1  # constructed once, cached thereafter
