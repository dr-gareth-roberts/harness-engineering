"""End-to-end smoke test against the real Anthropic API.

Requires:
- `pip install 'harness-engineering[anthropic]'` (or `uv sync --extra anthropic`)
- `ANTHROPIC_API_KEY` in the environment

Demonstrates: a real `Orchestrator` driving an `AnthropicRunner` that closes a
tool-use loop using the existing `harness.tools.Dispatcher`. Pre/post-tool hooks
log every dispatch and a `harness.policy.AllowList` ensures the model can only
call `echo`.

Run with:
    ANTHROPIC_API_KEY=sk-ant-... uv run python examples/anthropic_runner.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner, PostToolUse, PreToolUse
from harness.policy import AllowList, attach_pre_tool_policies
from harness.prompts import text
from harness.runner import AnthropicRunner
from harness.tools import Dispatcher, Tool


class EchoIn(BaseModel):
    text: str


def echo(args: EchoIn) -> str:
    return args.text


async def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; skipping real-API example.", file=sys.stderr)
        return 0

    transcript: list[str] = []
    dispatcher = Dispatcher(
        [Tool(name="echo", description="Echo back the input.", input_model=EchoIn, handler=echo)]
    )

    hooks = HookRunner()
    hooks.register(
        PreToolUse,
        lambda e: transcript.append(  # type: ignore[func-returns-value]
            f"[hook:pre]  -> tool={e.call.name} args={e.call.arguments}"
        ),
    )
    hooks.register(
        PostToolUse,
        lambda e: transcript.append(  # type: ignore[func-returns-value]
            f"[hook:post] <- tool={e.call.name} content={e.result.content!r}"
        ),
    )
    attach_pre_tool_policies(hooks, AllowList.of({"echo"}))

    runner = AnthropicRunner(dispatcher, hooks, max_tokens=2_000, effort="low")
    orch = Orchestrator(dispatcher, hooks, runner)

    agent = SubAgent(
        name="demo",
        system_prompt=(
            "You demonstrate the harness-engineering tool loop. When the user asks "
            "you to echo a word, call the `echo` tool with that exact word, then "
            "summarise what you did in one short sentence."
        ),
        model="claude-opus-4-7",
        allowed_tools=["echo"],
    )

    result = await orch.run(
        agent,
        [text("user", "Please echo the word 'roundtrip', then tell me what you did.")],
    )

    transcript.append("--- final assistant message ---")
    for block in result.content:
        if block.type == "text" and block.text:
            transcript.append(block.text)

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
