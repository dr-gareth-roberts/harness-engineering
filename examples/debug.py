"""Live debug REPL for orchestrator runs (programmatic mode).

Run with: `uv run python examples/debug.py`

`harness.debug.DebugRunner` wraps any runner and adds `pdb`-style
breakpoints. When the configured `break_on(ctx)` predicate returns True,
control is handed to either:

* `breakpoint_callback(ctx)` — a programmatic hook (sync or async).
* An interactive REPL on stdin/stdout (when `interactive=True`).

After the breakpoint exits cleanly:

* If `ctx.mutate(replacement)` was called, the runner returns
  `replacement` directly — the wrapped real_runner is *not* invoked.
* Otherwise, the runner delegates to the wrapped runner.

This example runs the programmatic path. The breakpoint inspects the
paused conversation, fires an ad-hoc tool call via `ctx.fire(...)`,
mutates the next assistant turn, then resumes — all without any
interactive I/O. The interactive REPL flavor is exercised by the
`harness debug` CLI (see `tests/debug/test_cli.py` for shape).
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.debug import DebugContext, DebugRunner
from harness.hooks import HookRunner
from harness.prompts import text
from harness.runner import CannedRunner
from harness.tools import Dispatcher, Tool


class LookupIn(BaseModel):
    key: str


def _build_dispatcher(log: list[str]) -> Dispatcher:
    def lookup(args: LookupIn) -> str:
        log.append(f"lookup({args.key!r})")
        return f"value-of-{args.key}"

    return Dispatcher(
        [
            Tool(
                name="lookup",
                description="Look up a key.",
                input_model=LookupIn,
                handler=lookup,
                idempotent=True,
            ),
        ]
    )


async def main() -> int:
    transcript: list[str] = []
    fire_log: list[str] = []
    dispatcher = _build_dispatcher(fire_log)

    # The wrapped runner would normally produce this canned reply.
    inner = CannedRunner(replies=["original reply from the model"])

    # The breakpoint callback. Fires once when the predicate hits (which
    # is the very first call here, since turn_index starts at 0).
    async def on_break(ctx: DebugContext) -> None:
        transcript.append("--- breakpoint hit ---")
        transcript.append(f"  paused at turn_index={ctx.turn_index}")
        transcript.append(f"  visible messages: {len(ctx.messages)}")
        for i, m in enumerate(ctx.messages):
            blocks = [b.type for b in m.content]
            transcript.append(f"    [{i}] role={m.role} blocks={blocks}")

        # Ad-hoc tool dispatch: doesn't advance the conversation, just
        # gives the operator a way to read state outside the model loop.
        result = await ctx.fire("lookup", {"key": "config-flag"})
        transcript.append(f"  fired lookup(key='config-flag') -> {result.content!r}")

        # Mutate the assistant's reply. The wrapped runner is NOT called
        # because mutate() short-circuits the runner once the breakpoint
        # exits cleanly.
        ctx.mutate(text("assistant", "rewritten by the debugger"))
        transcript.append("  queued mutation: rewritten reply")

        # Mark the breakpoint as resolved cleanly. Calling abort() instead
        # would raise DebugAborted out of the runner.
        ctx.resume()

    # Break on the first call only.
    debug_runner = DebugRunner(
        inner,
        break_on=lambda ctx: ctx.turn_index == 0,
        breakpoint_callback=on_break,
        dispatcher=dispatcher,
    )

    orchestrator = Orchestrator(dispatcher, HookRunner(), debug_runner)
    agent = SubAgent(
        name="debug-demo",
        system_prompt="",
        model="demo-model",
        allowed_tools=["lookup"],
    )

    final = await orchestrator.run(agent, [text("user", "hello")])

    transcript.append("--- after resume ---")
    final_text = next((b.text for b in final.content if b.type == "text"), "")
    transcript.append(f"  orchestrator returned: {final_text!r}")
    transcript.append(f"  ad-hoc dispatches: {fire_log}")
    # Verify the mutation short-circuited the inner runner: CannedRunner
    # tracks consumption via its private `_index`; if it's still 0, the
    # mutation replaced the runner's reply without invoking it.
    transcript.append(
        "  inner CannedRunner consumed: "
        + ("yes" if inner._index > 0 else "no — mutation short-circuited")
    )

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
