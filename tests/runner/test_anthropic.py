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
        client=client,
        max_iterations=3,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="max_iterations=3"):
        await runner(_agent(), [text("user", "loop")])


async def test_unexpected_stop_reason_raises() -> None:
    response = FakeMessage(content=[FakeTextBlock(text="...")], stop_reason="refusal")
    client = FakeAsyncAnthropic(responses=[response])
    dispatcher, _ = _echo_dispatcher()
    runner = AnthropicRunner(dispatcher, HookRunner(), client=client)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="refusal"):
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
        client=client,
        thinking_mode="disabled",  # type: ignore[arg-type]
    )

    await runner(_agent(), [text("user", "x")])
    assert "thinking" not in client.messages.requests[0]


# ---------------------------------------------------------------------------
# missing dep


def test_missing_anthropic_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    monkeypatch.delitem(sys.modules, "harness.runner.anthropic", raising=False)

    with pytest.raises(ImportError, match=r"harness-engineering\[anthropic\]"):
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
    hooks.register(PreToolUse, lambda e: pre_calls.append(e.call) or None)
    hooks.register(
        PostToolUse,
        lambda e: post_calls.append((e.call, e.result)) or None,
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
    hooks.register(PreToolUse, lambda e: pre_calls.append(e.call) or None)
    hooks.register(
        PostToolUse,
        lambda e: post_calls.append((e.call, e.result)) or None,
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
    bad_response = FakeMessage(content=[], stop_reason="refusal")  # unhandled
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
