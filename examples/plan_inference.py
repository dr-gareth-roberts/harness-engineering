"""Mine a `Plan` from past successful `SessionRecord`s — no model in the loop.

Run with: `uv run python examples/plan_inference.py`

`harness.plan.infer_plan_from_records` is the offline counterpart to
`harness.plan.derive_plan`. Where `derive_plan` asks a *live* planner
agent to emit a plan up-front, `infer_plan_from_records` mines one from
trajectories you've already paid for: filter by a success predicate,
extract each surviving record's tool_use name sequence, and pick the
modal sequence (with earliest-first-occurrence tiebreak).

The example wires together three pieces:

  1. Build five synthetic `SessionRecord`s. Four follow the modal
     sequence `search → parse → answer`; one outlier runs
     `search → answer` to show the algorithm picks the majority.
  2. Call `infer_plan_from_records(records)` with the default success
     heuristic — assistant-terminated, no error tool_results, no orphan
     tool_uses — and print the inferred steps + mode.
  3. Demonstrate the predicate hook by filtering records with a custom
     `success` lambda that excludes `r1` (which would otherwise vote
     for the modal sequence).
  4. Wire the inferred plan into a `PlanGuardedRunner` whose inner
     runner emits the same `search → parse → answer` sequence; the
     guard accepts the trajectory and `finalize()` succeeds because the
     plan is fully consumed (default mode is `"superset"`).
"""

from __future__ import annotations

import asyncio

from harness.agents import SubAgent
from harness.memory import SessionRecord
from harness.plan import PlanGuardedRunner
from harness.plan.infer import infer_plan_from_records
from harness.prompts import assistant_tool_use, text, user_tool_result
from harness.prompts.messages import ContentBlock, Message
from harness.tools import ToolCall, ToolResult


def _agent() -> SubAgent:
    return SubAgent(
        name="plan-infer-demo",
        system_prompt="",
        model="demo-model",
        allowed_tools=["search", "parse", "answer"],
    )


def _make_record(session_id: str, tool_names: list[str]) -> SessionRecord:
    """Build a record the default success heuristic accepts.

    The heuristic requires: at least one message, last message role is
    "assistant", no error tool_results, every assistant tool_use has a
    matching user tool_result by id. We satisfy all four by emitting,
    per tool name, an `assistant_tool_use` block + a paired
    `user_tool_result` block, then a final assistant text turn.
    """
    messages: list[Message] = [text("user", "do the work")]
    for i, name in enumerate(tool_names):
        call_id = f"{session_id}-{i}"
        messages.append(assistant_tool_use(ToolCall(name=name, arguments={}, id=call_id)))
        messages.append(user_tool_result(ToolResult(id=call_id, content="ok", is_error=False)))
    messages.append(text("assistant", "done"))
    return SessionRecord(session_id=session_id, agent=_agent(), messages=messages)


def _summarise_plan(label: str, plan_steps: list[str], mode: str) -> str:
    return f"  {label}: steps={plan_steps} mode={mode}"


async def main() -> int:
    transcript: list[str] = []

    # 4 records run search→parse→answer (the majority); 1 outlier runs
    # search→answer. The default success heuristic accepts all five
    # because every tool_use is paired with a tool_result and the final
    # message in each is the trailing "assistant: done".
    records = [
        _make_record("r1", ["search", "parse", "answer"]),
        _make_record("r2", ["search", "parse", "answer"]),
        _make_record("r3", ["search", "answer"]),  # outlier
        _make_record("r4", ["search", "parse", "answer"]),
        _make_record("r5", ["search", "parse", "answer"]),
    ]
    transcript.append("--- past records ---")
    for rec in records:
        seq = [
            block.tool_use.name
            for msg in rec.messages
            if msg.role == "assistant"
            for block in msg.content
            if block.type == "tool_use" and block.tool_use is not None
        ]
        transcript.append(f"  {rec.session_id}: {' -> '.join(seq)}")

    # 1) Default predicate: all five are "successful", so the inference
    # picks the modal sequence (search → parse → answer, 4 votes).
    plan = infer_plan_from_records(records)
    inferred_steps = [step.tool_name for step in plan.steps]
    transcript.append("--- inferred plan (default success heuristic) ---")
    transcript.append(_summarise_plan("plan", inferred_steps, plan.mode))

    # 2) Custom predicate: pretend r1 was actually a failure. The
    # remaining majority is still search→parse→answer (3 votes vs 1
    # outlier), so the modal sequence holds — but this demonstrates the
    # hook is exercised end-to-end.
    excluded = {"r1"}

    def _is_successful(record: SessionRecord) -> bool:
        return record.session_id not in excluded

    filtered_plan = infer_plan_from_records(records, success=_is_successful)
    filtered_steps = [step.tool_name for step in filtered_plan.steps]
    transcript.append("--- inferred plan (custom predicate excludes r1) ---")
    transcript.append(_summarise_plan("plan", filtered_steps, filtered_plan.mode))

    # 3) Wire the inferred plan into a PlanGuardedRunner. The "runner"
    # here is a closure satisfying the `Runner` protocol; it returns one
    # assistant message containing the same tool_use sequence the plan
    # encodes. The guard ticks each step's compiled DFA against each
    # tool_use block; in superset mode (the inference default), this
    # passes and `finalize()` exits cleanly.
    expected_calls = [
        ToolCall(name=name, arguments={}, id=f"plan-{i}") for i, name in enumerate(inferred_steps)
    ]
    scripted_message = Message(
        role="assistant",
        content=[ContentBlock(type="tool_use", tool_use=c) for c in expected_calls],
    )

    async def scripted_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return scripted_message

    guard = PlanGuardedRunner(scripted_runner, plan)
    await guard(_agent(), [])
    guard.finalize()  # superset mode + plan exhausted → no PlanViolation

    transcript.append("--- guard verdict ---")
    transcript.append(
        f"  PlanGuardedRunner accepted trajectory; step_index={guard.step_index} "
        f"of {len(plan.steps)} (plan exhausted: "
        f"{guard.step_index == len(plan.steps)})"
    )

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
