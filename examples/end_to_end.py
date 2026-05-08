"""End-to-end smoke test that wires every harness module together.

Run with: `uv run python examples/end_to_end.py`

There is no real model call. A fake `runner` plays a canned script:
1. The "assistant" first attempts a `shell` tool call. A pre-tool policy
   blocks it before the dispatcher runs.
2. The "assistant" retries with the allowed `echo` tool. The dispatcher
   runs it and the post-tool hook logs the result.
3. The "assistant" emits a final text message that references both.

Pre/post-tool hooks log each step so you can see the lifecycle.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner, PostToolUse, PreToolUse
from harness.policy import AllowList, attach_pre_tool_policies
from harness.prompts import (
    Message,
    attach_file,
    text,
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
    attach_pre_tool_policies(hooks, AllowList.of({"echo"}))
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
        # Attempt 1: forbidden tool — the AllowList policy blocks it pre-dispatch.
        forbidden = ToolCall(name="shell", arguments={"command": "echo hi"}, id="call-0")
        decisions = await hooks.emit(PreToolUse(call=forbidden))
        blocked = next((d for d in decisions if d.block), None)
        if blocked is not None:
            transcript.append(f"[policy]    BLOCKED tool={forbidden.name} reason={blocked.reason}")

        # Attempt 2: allowed tool — policy passes, dispatcher runs.
        allowed = ToolCall(name="echo", arguments={"text": "hello"}, id="call-1")
        await hooks.emit(PreToolUse(call=allowed))
        result = await dispatcher.dispatch(allowed)
        await hooks.emit(PostToolUse(call=allowed, result=result))

        return text(
            "assistant",
            f"Skipped forbidden tool; echoed {result.content!r}.",
        )

    orch = Orchestrator(dispatcher, hooks, fake_runner)
    agent = SubAgent(
        name="demo",
        system_prompt="Demonstrate every harness module in one turn.",
        model="demo-model",  # vendor-neutral; the fake runner ignores it
        allowed_tools=["echo"],
    )

    messages = build_initial_messages()
    final = await orch.run(agent, messages)

    transcript.append("--- transcript ---")
    transcript.append(f"input messages: {len(messages)}")
    for i, m in enumerate(messages):
        transcript.append(f"  [{i}] role={m.role} blocks={[b.type for b in m.content]}")
    transcript.append(f"final assistant message: {final.content[0].text!r}")

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
