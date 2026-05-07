"""End-to-end smoke test that wires all four harness modules together.

Run with: `uv run python examples/end_to_end.py`

There is no real model call. A fake `runner` plays a canned script:
1. The "assistant" emits a tool_use for `echo`.
2. The orchestrator's caller (this script) dispatches the tool.
3. The "assistant" emits a final text message.

Pre/post-tool hooks log each step so you can see the lifecycle.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner, PostToolUse, PreToolUse
from harness.prompts import (
    Message,
    assistant_tool_use,
    attach_file,
    text,
    user_tool_result,
)
from harness.tools import Dispatcher, Tool, ToolCall


class EchoIn(BaseModel):
    text: str


def echo(args: EchoIn) -> str:
    return args.text


def build_dispatcher() -> Dispatcher:
    return Dispatcher(
        [Tool(name="echo", description="Echo back the input.", input_model=EchoIn, handler=echo)]
    )


def build_hooks(transcript: list[str]) -> HookRunner:
    hooks = HookRunner()

    def on_pre(event: PreToolUse) -> None:
        transcript.append(f"[hook:pre]  -> tool={event.call.name} args={event.call.arguments}")

    def on_post(event: PostToolUse) -> None:
        transcript.append(f"[hook:post] <- tool={event.call.name} content={event.result.content!r}")

    hooks.register(PreToolUse, on_pre)
    hooks.register(PostToolUse, on_post)
    return hooks


def build_initial_messages() -> list[Message]:
    here = Path(__file__).resolve()
    return [
        text("system", "You are a small demo assistant."),
        Message(role="user", content=[attach_file(here)]),
        text("user", "Please echo the word 'hello'."),
    ]


async def main() -> int:
    transcript: list[str] = []
    dispatcher = build_dispatcher()
    hooks = build_hooks(transcript)

    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        # Step 1: the model decides to use the echo tool.
        call = ToolCall(name="echo", arguments={"text": "hello"}, id="call-1")
        await hooks.emit(PreToolUse(call=call))
        result = await dispatcher.dispatch(call)
        await hooks.emit(PostToolUse(call=call, result=result))

        # Step 2: the model returns a final text answer.
        return text("assistant", f"Echoed: {result.content}")

    orch = Orchestrator(dispatcher, hooks, fake_runner)
    agent = SubAgent(
        name="demo",
        system_prompt="Demonstrate every harness module in one turn.",
        allowed_tools=["echo"],
    )

    messages = build_initial_messages()

    # Show the synthesized assistant turn and a follow-up tool_result so the
    # transcript captures the full message shape, not just the final answer.
    final = await orch.run(agent, messages)
    intermediate_call = ToolCall(name="echo", arguments={"text": "hello"}, id="call-1")
    intermediate_result = await dispatcher.dispatch(intermediate_call)

    transcript.append("--- transcript ---")
    transcript.append(f"input messages: {len(messages)}")
    for i, m in enumerate(messages):
        transcript.append(f"  [{i}] role={m.role} blocks={[b.type for b in m.content]}")
    transcript.append(f"assistant tool_use: {assistant_tool_use(intermediate_call).model_dump()}")
    transcript.append(f"user tool_result:   {user_tool_result(intermediate_result).model_dump()}")
    transcript.append(f"final assistant message: {final.content[0].text!r}")

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
