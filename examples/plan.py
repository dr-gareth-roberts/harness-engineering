"""Plan-as-contract: enforce an expected sequence of tool calls.

Run with: `uv run python examples/plan.py`

`harness.plan.PlanGuardedRunner` wraps any `Runner` and enforces a
`Plan` (an ordered list of `PlannedToolCall`) on the tool_use blocks
the inner runner emits. In `strict` mode any deviation - wrong tool,
wrong arguments, missing call, extra call - raises `PlanViolation`.

This example declares a three-step plan (search -> parse -> summarize)
with a tiny `Dispatcher` exposing trivial handlers for each tool. Two
fake runners drive it through an `Orchestrator`:

  1. A compliant runner that emits all three tool_use blocks in
     order, in a single assistant message. The guard ticks each
     step's DFA, advances the pointer to the end, and `finalize()`
     confirms plan exhaustion.
  2. A non-compliant runner that emits a *wrong* sequence (it skips
     `parse` and calls `delete` instead). The guard raises
     `PlanViolation` on the offending step.

Both outcomes print so the example exits 0 either way.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.plan import Plan, PlanGuardedRunner, PlannedToolCall, PlanViolation
from harness.prompts import ContentBlock, Message, text
from harness.tools import Dispatcher, Tool, ToolCall

# ----------------------------------------------------------------------
# A minimal tool surface: search / parse / summarize / delete.
#
# Each tool has a trivial input model and handler - we only care that
# the guard recognizes the tool name; the bodies are placeholders.


class SearchIn(BaseModel):
    query: str


class ParseIn(BaseModel):
    document: str


class SummarizeIn(BaseModel):
    text: str


class DeleteIn(BaseModel):
    target: str


def _build_dispatcher() -> Dispatcher:
    return Dispatcher(
        [
            Tool(
                name="search",
                description="Search a corpus.",
                input_model=SearchIn,
                handler=lambda args: f"hits-for-{args.query}",
                idempotent=True,
            ),
            Tool(
                name="parse",
                description="Parse a document.",
                input_model=ParseIn,
                handler=lambda args: f"parsed-{args.document}",
                idempotent=True,
            ),
            Tool(
                name="summarize",
                description="Summarize text.",
                input_model=SummarizeIn,
                handler=lambda args: f"summary-of-{args.text}",
                idempotent=True,
            ),
            Tool(
                name="delete",
                description="Delete something.",
                input_model=DeleteIn,
                handler=lambda args: f"deleted-{args.target}",
            ),
        ]
    )


def _agent() -> SubAgent:
    return SubAgent(
        name="plan-demo",
        system_prompt="follow the plan",
        model="demo-model",
        allowed_tools=["search", "parse", "summarize", "delete"],
    )


def _build_plan() -> Plan:
    """search -> parse -> summarize, in order, no extras (strict)."""
    return Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="parse"),
            PlannedToolCall(tool_name="summarize"),
        ],
        mode="strict",
    )


def _multi_tool_message(*calls: ToolCall) -> Message:
    """Pack several tool_use blocks into a single assistant message.

    The guard's `_iter_tool_uses` walks every tool_use block in the
    returned message, so a single Orchestrator turn can satisfy a
    multi-step plan.
    """
    return Message(
        role="assistant",
        content=[ContentBlock(type="tool_use", tool_use=c) for c in calls],
    )


def _scripted_runner(replies: list[Message]):  # type: ignore[no-untyped-def]
    """Tiny inline runner satisfying the Runner protocol; returns one
    canned `Message` per call. CannedRunner only handles text; here we
    need to return tool_use blocks, so we roll our own.
    """
    iterator = iter(replies)

    async def run(agent: SubAgent, messages: list[Message]) -> Message:
        return next(iterator)

    return run


# ----------------------------------------------------------------------


async def part_1_compliant(transcript: list[str]) -> None:
    transcript.append("--- part 1: compliant runner under plan (strict) ---")

    plan = _build_plan()
    runner = _scripted_runner(
        [
            _multi_tool_message(
                ToolCall(name="search", arguments={"query": "rust"}, id="c1"),
                ToolCall(name="parse", arguments={"document": "doc.txt"}, id="c2"),
                ToolCall(name="summarize", arguments={"text": "abc"}, id="c3"),
            )
        ]
    )
    guard = PlanGuardedRunner(runner, plan)

    orch = Orchestrator(_build_dispatcher(), HookRunner(), guard)
    final = await orch.run(_agent(), [text("user", "do the plan")])

    # End-of-run check: in strict mode, plan must be exhausted.
    guard.finalize()

    tool_names = [b.tool_use.name for b in final.content if b.type == "tool_use" and b.tool_use]
    transcript.append(f"  emitted tool_use sequence: {tool_names}")
    transcript.append(f"  guard.step_index after run: {guard.step_index} / {len(plan.steps)}")
    transcript.append("  finalize() returned cleanly: plan satisfied")


async def part_2_violator(transcript: list[str]) -> None:
    transcript.append("--- part 2: non-compliant runner under plan (strict) ---")

    plan = _build_plan()
    # Wrong sequence: search ok, but then delete instead of parse.
    runner = _scripted_runner(
        [
            _multi_tool_message(
                ToolCall(name="search", arguments={"query": "rust"}, id="c1"),
                ToolCall(name="delete", arguments={"target": "everything"}, id="c2"),
            )
        ]
    )
    guard = PlanGuardedRunner(runner, plan)

    orch = Orchestrator(_build_dispatcher(), HookRunner(), guard)
    raised: PlanViolation | None = None
    try:
        await orch.run(_agent(), [text("user", "do the plan")])
    except PlanViolation as exc:
        raised = exc

    if raised is None:
        transcript.append("  unexpected: no plan violation raised")
        return

    expected_name = raised.expected.tool_name if raised.expected is not None else None
    actual_name = raised.actual.name if raised.actual is not None else None
    transcript.append(f"  PlanViolation raised at step_index={raised.step_index}")
    transcript.append(f"    expected tool: {expected_name!r}")
    transcript.append(f"    actual tool:   {actual_name!r}")
    transcript.append(f"    message: {raised}")


async def main() -> int:
    transcript: list[str] = []
    await part_1_compliant(transcript)
    transcript.append("")
    await part_2_violator(transcript)
    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
