"""Speculative tool execution: pre-run predicted calls in parallel with the model.

Run with: `uv run python examples/speculate.py`

`harness.speculate.Speculator` lets a runner overlap the round-trip
latency of an idempotent tool call with the model's own generation
latency. While the model is busy producing the next token stream, the
speculator launches a background task running the *predicted* tool call
through the same `PreToolUse` / dispatch / `PostToolUse` cycle. When the
model finally emits a `tool_use`, `try_resolve` matches it against the
pending speculation and returns the already-computed result — saving
roughly one tool round-trip on a hit.

This example demonstrates that wall-clock win without depending on a
vendor SDK fake. We drive the speculator's `begin` / `try_resolve` / `end`
lifecycle directly: the same shape the integration test
`tests/speculate/test_speculator.py::test_speculation_runs_in_parallel_with_caller_work`
uses to assert the speed-up. Going through `AnthropicRunner` would
require a fake stream + fake `tool_use` block, which is more plumbing
without making the parallelism story any clearer.

Two timings are printed:
  * baseline ("serial") — model wait *then* tool call: ~200ms.
  * speculated ("parallel") — model wait || tool call: ~100ms.

The transcript also surfaces the `SpeculationLaunched` and
`SpeculationHit` telemetry events captured via `MemorySink`.
"""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel

from harness.agents import SubAgent
from harness.hooks import HookRunner
from harness.prompts.messages import ContentBlock, Message
from harness.speculate import LastCallPredictor, Speculator
from harness.speculate.events import SpeculationHit, SpeculationLaunched
from harness.telemetry import MemorySink, Telemetry
from harness.tools import Dispatcher, Tool
from harness.tools.schema import ToolCall

# A "model wait" of 100ms simulates time the runner spends reading a
# token stream. A 100ms tool sleep simulates a real tool round-trip
# (HTTP fetch, file read, etc.). Equal sleeps make the win obvious:
# serial = 200ms, parallel ~= 100ms.
_MODEL_WAIT_S = 0.1
_TOOL_LATENCY_S = 0.1


class _SearchArgs(BaseModel):
    query: str


def _build_dispatcher() -> Dispatcher:
    """A single idempotent `search` tool whose handler sleeps to simulate
    the round-trip a real backed search would incur.
    """

    async def search_handler(args: _SearchArgs) -> str:
        await asyncio.sleep(_TOOL_LATENCY_S)
        return f"hits-for-{args.query}"

    return Dispatcher(
        [
            Tool(
                name="search",
                description="Search a corpus.",
                input_model=_SearchArgs,
                handler=search_handler,
                idempotent=True,
            )
        ]
    )


def _agent() -> SubAgent:
    return SubAgent(
        name="speculate-demo",
        system_prompt="",
        model="demo-model",
        allowed_tools=["search"],
    )


def _previous_call_history(query: str) -> list[Message]:
    """Build a history whose last assistant turn calls `search(query)` —
    so `LastCallPredictor` (window=1) predicts a repeat of that call.
    """
    return [
        Message(role="user", content=[ContentBlock(type="text", text="find rust posts")]),
        Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name="search", arguments={"query": query}, id="prev"),
                )
            ],
        ),
    ]


async def _serial_baseline(transcript: list[str]) -> float:
    """Run "model wait" *then* the tool dispatch — what you get without speculation."""
    transcript.append("--- serial baseline (no speculation) ---")
    dispatcher = _build_dispatcher()
    start = time.perf_counter()
    # Pretend we're waiting for the model's stream to arrive.
    await asyncio.sleep(_MODEL_WAIT_S)
    # Then we dispatch the tool the model asked for.
    result = await dispatcher.dispatch(
        ToolCall(name="search", arguments={"query": "rust"}, id="real-1")
    )
    elapsed = time.perf_counter() - start
    transcript.append(f"  result: {result.content!r}")
    expected_total = _MODEL_WAIT_S + _TOOL_LATENCY_S
    transcript.append(f"  elapsed: {elapsed * 1000:.0f}ms (~{expected_total:.2f}s)")
    return elapsed


async def _speculated(transcript: list[str]) -> tuple[float, MemorySink]:
    """Same workload, but with the speculator running the tool in parallel.

    The lifecycle (begin / `await sleep` / try_resolve / end) is the
    runner's contract with `SpeculatorProtocol`, copied verbatim from the
    integration test. The speculator picks `search(query="rust")` from
    the prior tool_use in history; while we await our 100ms "model
    wait", that tool's 100ms handler runs concurrently. By the time we
    call `try_resolve`, the result is ready — no extra wait.
    """
    transcript.append("--- with Speculator + LastCallPredictor ---")
    dispatcher = _build_dispatcher()
    sink = MemorySink()
    speculator = Speculator(
        predictor=LastCallPredictor(history_window=1),
        max_speculations=1,
        telemetry=Telemetry(sink=sink),
    )

    history = _previous_call_history(query="rust")
    start = time.perf_counter()
    await speculator.begin(
        history=history,
        agent=_agent(),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )
    # Caller does its "model wait" *while* the predicted tool runs.
    await asyncio.sleep(_MODEL_WAIT_S)
    # Model has now (in this fake) emitted the same tool_use the
    # predictor anticipated → speculator hands us the cached result.
    actual_call = ToolCall(name="search", arguments={"query": "rust"}, id="real-1")
    result = await speculator.try_resolve(actual_call)
    await speculator.end()
    elapsed = time.perf_counter() - start

    assert result is not None, "speculation should have hit"
    transcript.append(f"  result: {result.content!r}")
    transcript.append(f"  elapsed: {elapsed * 1000:.0f}ms (~{_MODEL_WAIT_S:.2f}s)")
    return elapsed, sink


def _summarise_events(transcript: list[str], sink: MemorySink) -> None:
    transcript.append("--- captured telemetry ---")
    for event in sink.events:
        if isinstance(event, SpeculationLaunched):
            transcript.append(f"  SpeculationLaunched: tool={event.tool_name}")
        elif isinstance(event, SpeculationHit):
            transcript.append(f"  SpeculationHit: tool={event.tool_name}")
        else:  # SpeculationMiss; not expected on this happy path
            transcript.append(f"  {type(event).__name__}: {event.model_dump()}")


async def main() -> int:
    transcript: list[str] = []
    serial = await _serial_baseline(transcript)
    parallel, sink = await _speculated(transcript)
    _summarise_events(transcript, sink)

    transcript.append("--- summary ---")
    speedup = serial / parallel if parallel > 0 else float("inf")
    transcript.append(
        f"  speculation saved ~{(serial - parallel) * 1000:.0f}ms "
        f"({speedup:.2f}x faster than serial)"
    )
    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
