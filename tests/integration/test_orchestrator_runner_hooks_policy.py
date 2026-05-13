"""Orchestrator + Runner + Hooks + Policy — runner-protocol-conformant fake (M4.5).

Wires every public piece that a real caller would compose to drive a
single multi-tool turn:

- a `Dispatcher` with three concrete `Tool`s,
- a `HookRunner` with a `PreToolUse` `AllowList` policy blocking one
  of them,
- a vendor-neutral fake runner that drives the same hook + dispatch
  cycle the speculator and Anthropic runner implementations use
  internally,
- a real `Orchestrator` running the whole thing.

This suite uses fake runner callables to exercise the
orchestrator/runner protocol surface. End-to-end coverage with
concrete vendor runners (`AnthropicRunner` / `OpenAICompatRunner`
against faked SDK boundaries) is tracked separately.

Pins:

- The orchestrator emits `SessionStart` / `SessionEnd` around the turn.
- The `AllowList` policy returns a `HookDecision(block=True)` for the
  forbidden tool; the runner converts that to an `is_error=True`
  `ToolResult` and skips dispatch.
- The other two tools dispatch through the registered handlers and
  return normal results.
- The final trajectory recorded in the runner's "messages it saw" log
  is the shape we expect: every tool call survives as a `tool_use`
  block plus a paired `tool_result`, blocked or not.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.hooks import (
    HookDecision,
    HookRunner,
    PostToolUse,
    PreToolUse,
    SessionEnd,
    SessionStart,
)
from harness.policy import AllowList, attach_pre_tool_policies
from harness.prompts import Message, assistant_tool_use, text, user_tool_result
from harness.tools import Dispatcher, Tool, ToolCall, ToolResult


class _SearchArgs(BaseModel):
    query: str


class _ReadArgs(BaseModel):
    path: str


class _WriteArgs(BaseModel):
    path: str
    content: str


def _build_dispatcher(call_log: list[tuple[str, dict[str, Any]]]) -> Dispatcher:
    """A dispatcher with three tools — one read-shaped, one write-shaped,
    one search-shaped. Each appends its invocation to `call_log` so the
    test can prove which handlers actually ran.
    """

    async def search(args: _SearchArgs) -> str:
        call_log.append(("search", {"query": args.query}))
        return f"search results for {args.query!r}"

    async def read_file(args: _ReadArgs) -> str:
        call_log.append(("read_file", {"path": args.path}))
        return f"contents of {args.path}"

    async def write_file(args: _WriteArgs) -> str:
        call_log.append(("write_file", {"path": args.path, "content": args.content}))
        return f"wrote {len(args.content)} bytes to {args.path}"

    return Dispatcher(
        [
            Tool(
                name="search",
                description="search the corpus",
                input_model=_SearchArgs,
                handler=search,
                idempotent=True,
            ),
            Tool(
                name="read_file",
                description="read a file",
                input_model=_ReadArgs,
                handler=read_file,
                idempotent=True,
            ),
            Tool(
                name="write_file",
                description="write a file",
                input_model=_WriteArgs,
                handler=write_file,
                idempotent=False,
            ),
        ]
    )


def _make_runner(
    calls_to_make: list[ToolCall],
    dispatcher: Dispatcher,
    hooks: HookRunner,
    seen_history: list[list[Message]],
) -> Any:
    """Build a vendor-neutral runner that simulates the model emitting
    each `ToolCall` in `calls_to_make` and driving the hook + dispatch
    cycle the way `AnthropicRunner` / `OpenAICompatRunner` do.

    Stores the message list it received in `seen_history` so the test
    can assert on what the orchestrator handed in. Returns a single
    final assistant message containing every tool_use + its matching
    tool_result (so the trajectory has the canonical interleaved
    shape).
    """

    async def _runner(agent: SubAgent, messages: list[Message]) -> Message:
        seen_history.append(list(messages))
        trajectory_blocks: list[Any] = []

        for call in calls_to_make:
            # 1) Pre-tool hooks. The first `HookDecision(block=True)`
            #    means the runner must NOT dispatch and must surface an
            #    error result the model can self-correct from.
            decisions = await hooks.emit(PreToolUse(call=call))
            blocked = next((d for d in decisions if d.block), None)
            if blocked is not None:
                result = ToolResult(
                    id=call.id,
                    content=blocked.reason or "blocked by hook",
                    is_error=True,
                )
            else:
                result = await dispatcher.dispatch(call)

            # 2) Post-tool hooks. Observational — the runner emits these
            #    even for blocked calls so contracts / telemetry can see
            #    the full sequence.
            await hooks.emit(PostToolUse(call=call, result=result))

            # 3) Record the tool_use → tool_result pair in the final
            #    assistant message we'll return. Real vendor runners
            #    return the assistant tool_use as one message and the
            #    user tool_result as a follow-up; for this integration
            #    test we flatten them into the same trajectory message
            #    so the assertions can read one block-list.
            trajectory_blocks.extend(assistant_tool_use(call).content)
            trajectory_blocks.extend(user_tool_result(result).content)

        return Message(role="assistant", content=trajectory_blocks)

    return _runner


async def test_allow_list_policy_blocks_one_tool_others_dispatch(
    make_agent: Callable[..., SubAgent],
) -> None:
    """Drive a three-tool sequence; allow `search` + `read_file`,
    block `write_file`. The policy returns a `HookDecision(block=True)`
    on the `write_file` PreToolUse; the runner surfaces it as a
    `ToolResult(is_error=True)` without dispatching the handler.
    """
    dispatcher_log: list[tuple[str, dict[str, Any]]] = []
    dispatcher = _build_dispatcher(dispatcher_log)

    hooks = HookRunner()
    lifecycle_events: list[type] = []

    def _on_session_start(event: SessionStart) -> None:
        lifecycle_events.append(type(event))

    def _on_session_end(event: SessionEnd) -> None:
        lifecycle_events.append(type(event))

    hooks.register(SessionStart, _on_session_start)
    hooks.register(SessionEnd, _on_session_end)

    pre_tool_seen: list[str] = []

    def _on_pre_tool(event: PreToolUse) -> None:
        pre_tool_seen.append(event.call.name)

    hooks.register(PreToolUse, _on_pre_tool)

    post_tool_seen: list[tuple[str, bool]] = []

    def _on_post_tool(event: PostToolUse) -> None:
        post_tool_seen.append((event.call.name, event.result.is_error))

    hooks.register(PostToolUse, _on_post_tool)

    # The AllowList must run AFTER our observational PreToolUse handler
    # so the test sees the call name before the block fires. HookRunner
    # short-circuits on the first `block=True`, so registration order
    # matters.
    attach_pre_tool_policies(hooks, AllowList.of(["search", "read_file"]))

    calls = [
        ToolCall(id="c1", name="search", arguments={"query": "octopus"}),
        ToolCall(id="c2", name="write_file", arguments={"path": "/tmp/x", "content": "no"}),
        ToolCall(id="c3", name="read_file", arguments={"path": "/etc/hosts"}),
    ]
    seen_history: list[list[Message]] = []
    runner = _make_runner(calls, dispatcher, hooks, seen_history)

    orchestrator = Orchestrator(dispatcher, hooks, runner)
    agent = make_agent(allowed_tools=["search", "read_file", "write_file"])
    reply = await orchestrator.run(agent, [text("user", "do the thing")])

    # SessionStart fires before the runner; SessionEnd after.
    assert lifecycle_events == [SessionStart, SessionEnd]

    # Observational PreToolUse saw every call (including the blocked
    # one) before the policy fired and short-circuited the chain.
    assert pre_tool_seen == ["search", "write_file", "read_file"]

    # Dispatch log: search + read_file ran; write_file did NOT.
    assert ("search", {"query": "octopus"}) in dispatcher_log
    assert ("read_file", {"path": "/etc/hosts"}) in dispatcher_log
    assert not any(name == "write_file" for name, _ in dispatcher_log)

    # PostToolUse: error=True only for the blocked one.
    assert post_tool_seen == [
        ("search", False),
        ("write_file", True),
        ("read_file", False),
    ]

    # Trajectory: every tool_use survives, every one has a paired
    # tool_result, blocked or not. The reply shape is the canonical
    # interleave.
    block_kinds = [b.type for b in reply.content]
    assert block_kinds == [
        "tool_use",
        "tool_result",
        "tool_use",
        "tool_result",
        "tool_use",
        "tool_result",
    ]

    # The write_file result block is the blocked one; verify it carries
    # the policy's reason verbatim and is_error.
    blocked_result_block = reply.content[3]
    assert blocked_result_block.type == "tool_result"
    assert blocked_result_block.tool_result is not None
    assert blocked_result_block.tool_result.is_error is True
    assert "write_file" in (blocked_result_block.tool_result.content or "")
    assert "allow-list" in (blocked_result_block.tool_result.content or "")

    # The successful results carry the handler's return value.
    search_result_block = reply.content[1]
    assert search_result_block.tool_result is not None
    assert search_result_block.tool_result.is_error is False
    assert "octopus" in str(search_result_block.tool_result.content)

    # The orchestrator passed the original user message into the runner
    # unchanged.
    assert len(seen_history) == 1
    assert seen_history[0][0].role == "user"


async def test_hook_decision_replacement_field_is_inspectable(
    make_agent: Callable[..., SubAgent],
) -> None:
    """The `HookDecision` returned by an `AllowList` block exposes
    `block`, `reason`, and `replacement` for downstream inspection.
    A real caller can read these fields to surface policy decisions to
    a UI / log without re-running the policy callable. Pins the public
    surface of the data the runner just acted on.
    """
    dispatcher_log: list[tuple[str, dict[str, Any]]] = []
    dispatcher = _build_dispatcher(dispatcher_log)

    hooks = HookRunner()
    collected_decisions: list[HookDecision] = []

    async def collect(event: PreToolUse) -> None:
        # Re-emit the AllowList directly so the test sees the raw
        # decision rather than only the runner's translated ToolResult.
        decision = AllowList.of(["search"])(event)
        if decision is not None:
            collected_decisions.append(decision)

    hooks.register(PreToolUse, collect)
    attach_pre_tool_policies(hooks, AllowList.of(["search"]))

    calls = [
        ToolCall(id="c1", name="search", arguments={"query": "x"}),
        ToolCall(id="c2", name="write_file", arguments={"path": "/tmp/x", "content": ""}),
    ]
    runner = _make_runner(calls, dispatcher, hooks, [])
    orchestrator = Orchestrator(dispatcher, hooks, runner)
    agent = make_agent(allowed_tools=["search", "write_file"])
    await orchestrator.run(agent, [text("user", "go")])

    # Exactly one block-shaped decision was emitted — the write_file one.
    # The search call returned None from the AllowList (allowed).
    assert len(collected_decisions) == 1
    assert collected_decisions[0].block is True
    assert collected_decisions[0].reason is not None
    assert "allow-list" in collected_decisions[0].reason
    assert collected_decisions[0].replacement is None
