from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel

from harness.agents import SubAgent
from harness.hooks import HookRunner, PreToolUse
from harness.prompts.messages import ContentBlock, Message
from harness.speculate import LastCallPredictor, Speculator
from harness.speculate.predictor import Predictor
from harness.telemetry import MemorySink, Telemetry
from harness.tools import Dispatcher, Tool
from harness.tools.schema import ToolCall


class _Args(BaseModel):
    q: str = ""


def _agent(allowed: list[str]) -> SubAgent:
    return SubAgent(
        name="t",
        system_prompt="",
        model="test-model",
        allowed_tools=allowed,
    )


def _dispatcher(
    *,
    idempotent: list[str] | None = None,
    non_idempotent: list[str] | None = None,
    handler_log: list[str] | None = None,
    slow_handlers: dict[str, float] | None = None,
) -> Dispatcher:
    """Build a dispatcher with a configurable mix of tools.

    `slow_handlers` lets a test inject `await asyncio.sleep(seconds)` so
    we can prove parallelism via wall-clock measurement.
    """
    log = handler_log if handler_log is not None else []
    sleeps = slow_handlers or {}
    tools: list[Tool] = []

    def make_handler(name: str) -> Any:
        async def handler(args: _Args) -> str:
            log.append(f"{name}({args.q})")
            sleep = sleeps.get(name)
            if sleep is not None:
                await asyncio.sleep(sleep)
            return f"{name}-result-{args.q}"

        return handler

    for name in idempotent or []:
        tools.append(
            Tool(
                name=name,
                description="",
                input_model=_Args,
                handler=make_handler(name),
                idempotent=True,
            )
        )
    for name in non_idempotent or []:
        tools.append(
            Tool(
                name=name,
                description="",
                input_model=_Args,
                handler=make_handler(name),
                idempotent=False,
            )
        )
    return Dispatcher(tools)


def _history_with_call(name: str, args: dict[str, object]) -> list[Message]:
    return [
        Message(role="user", content=[ContentBlock(type="text", text="hi")]),
        Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name=name, arguments=args, id="prev"),
                )
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Cap + filtering


async def test_max_speculations_caps_concurrent_dispatches() -> None:
    log: list[str] = []
    dispatcher = _dispatcher(idempotent=["a", "b", "c", "d"], handler_log=log)

    class FixedPredictor:
        """Returns four predictions; speculator must cap at max_speculations."""

        def predict(
            self,
            history: list[Message],
            idempotent_tools: dict[str, Tool],
            max_predictions: int,
        ) -> list[ToolCall]:
            return [
                ToolCall(name="a", arguments={"q": "1"}),
                ToolCall(name="b", arguments={"q": "2"}),
                ToolCall(name="c", arguments={"q": "3"}),
                ToolCall(name="d", arguments={"q": "4"}),
            ]

    speculator = Speculator(FixedPredictor(), max_speculations=2)
    await speculator.begin(
        history=[],
        agent=_agent(["a", "b", "c", "d"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )
    # Give the event loop a chance to run the scheduled tasks. In the
    # real runner, the model's stream call provides this naturally.
    await asyncio.sleep(0.01)
    await speculator.end()

    # Only the first two predictions were dispatched.
    assert sorted(log) == ["a(1)", "b(2)"]


async def test_speculator_refuses_non_idempotent_tools_by_default() -> None:
    log: list[str] = []
    dispatcher = _dispatcher(non_idempotent=["send_email"], handler_log=log)

    class FixedPredictor:
        def predict(
            self,
            history: list[Message],
            idempotent_tools: dict[str, Tool],
            max_predictions: int,
        ) -> list[ToolCall]:
            # Predicting send_email — but it's non-idempotent so the speculator
            # should never let it through.
            return [ToolCall(name="send_email", arguments={"q": "hi"})]

    speculator = Speculator(FixedPredictor(), max_speculations=2)
    await speculator.begin(
        history=[],
        agent=_agent(["send_email"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )
    await speculator.end()

    # Handler was never called — non-idempotent filtered out at begin time.
    assert log == []


async def test_only_idempotent_false_lets_through_any_allowed_tool() -> None:
    log: list[str] = []
    dispatcher = _dispatcher(non_idempotent=["send_email"], handler_log=log)

    class FixedPredictor:
        def predict(
            self,
            history: list[Message],
            idempotent_tools: dict[str, Tool],
            max_predictions: int,
        ) -> list[ToolCall]:
            return [ToolCall(name="send_email", arguments={"q": "hi"})]

    speculator = Speculator(FixedPredictor(), max_speculations=2, only_idempotent=False)
    await speculator.begin(
        history=[],
        agent=_agent(["send_email"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )
    await asyncio.sleep(0.01)  # let scheduled tasks run before end() cancels
    await speculator.end()
    # Now the non-idempotent send_email was dispatched.
    assert log == ["send_email(hi)"]


# ---------------------------------------------------------------------------
# Hit / miss / telemetry


async def test_hit_returns_cached_result_with_call_id_patched() -> None:
    dispatcher = _dispatcher(idempotent=["search"])
    speculator = Speculator(LastCallPredictor(history_window=1), max_speculations=1)

    history = _history_with_call("search", {"q": "x"})
    await speculator.begin(
        history=history,
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )

    # The model's actual call has the same shape but a fresh id.
    actual = ToolCall(name="search", arguments={"q": "x"}, id="model-emitted-id")
    result = await speculator.try_resolve(actual)
    await speculator.end()

    assert result is not None
    assert result.id == "model-emitted-id"
    assert result.content == "search-result-x"


async def test_miss_returns_none_and_keeps_other_pending_speculations() -> None:
    log: list[str] = []
    dispatcher = _dispatcher(idempotent=["search", "parse"], handler_log=log)
    speculator = Speculator(LastCallPredictor(history_window=2), max_speculations=2)

    history = [
        Message(role="user", content=[ContentBlock(type="text", text="hi")]),
        Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name="search", arguments={"q": "x"}, id="a"),
                )
            ],
        ),
        Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name="parse", arguments={"q": "x"}, id="b"),
                )
            ],
        ),
    ]

    await speculator.begin(
        history=history,
        agent=_agent(["search", "parse"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )

    # Model calls something neither speculation predicted: miss.
    miss_result = await speculator.try_resolve(
        ToolCall(name="search", arguments={"q": "DIFFERENT"}, id="m1")
    )
    assert miss_result is None

    # The other speculation is still pending — model could still hit it.
    hit_result = await speculator.try_resolve(
        ToolCall(name="parse", arguments={"q": "x"}, id="m2")
    )
    assert hit_result is not None
    assert hit_result.content == "parse-result-x"

    await speculator.end()


async def test_telemetry_emits_launched_hit_and_miss_events() -> None:
    dispatcher = _dispatcher(idempotent=["search"])
    sink = MemorySink()
    speculator = Speculator(
        LastCallPredictor(history_window=1),
        max_speculations=1,
        telemetry=Telemetry(sink=sink),
    )

    history = _history_with_call("search", {"q": "x"})
    await speculator.begin(
        history=history,
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )
    # Hit
    await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m"))
    # Miss (no other speculation pending)
    await speculator.try_resolve(ToolCall(name="other", arguments={}, id="m2"))
    await speculator.end()

    kinds = [type(e).__name__ for e in sink.events]
    assert "SpeculationLaunched" in kinds
    assert "SpeculationHit" in kinds
    assert "SpeculationMiss" in kinds


# ---------------------------------------------------------------------------
# Hook participation + concurrency + cleanup


async def test_speculative_dispatch_fires_pre_and_post_tool_hooks() -> None:
    """A BlockingPolicy hook must see speculative calls. The doc explicitly
    pegs this behavior — speculations go through the same PreToolUse flow."""
    dispatcher = _dispatcher(idempotent=["search"])
    seen_pre: list[ToolCall] = []
    hooks = HookRunner()
    hooks.register(PreToolUse, lambda e: seen_pre.append(e.call) or None)

    speculator = Speculator(LastCallPredictor(history_window=1), max_speculations=1)
    await speculator.begin(
        history=_history_with_call("search", {"q": "x"}),
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=hooks,
    )
    await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m"))
    await speculator.end()

    # Speculative dispatch fired PreToolUse exactly once.
    assert len(seen_pre) == 1
    assert seen_pre[0].name == "search"


async def test_speculation_runs_in_parallel_with_caller_work() -> None:
    """Wall-clock proof of concurrency: a speculation that takes 200ms,
    awaited concurrently with another 200ms task, completes in
    significantly less than 400ms. This is the latency win speculation
    exists for.

    The discriminating assertion is the *relative* one: parallel must be
    at least 30% faster than serial would have been. The absolute bound
    is a sanity check; the relative bound is the actual claim.
    """
    sleep_per = 0.20
    serial_baseline = sleep_per * 2  # what running sequentially would take
    dispatcher = _dispatcher(
        idempotent=["search"],
        slow_handlers={"search": sleep_per},
    )
    speculator = Speculator(LastCallPredictor(history_window=1), max_speculations=1)

    start = time.perf_counter()
    await speculator.begin(
        history=_history_with_call("search", {"q": "x"}),
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )
    # Caller does the same amount of "model wait" work in parallel.
    await asyncio.sleep(sleep_per)
    # By the time we ask for the result, speculation has already finished
    # — try_resolve returns near-instantly.
    await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m"))
    await speculator.end()
    elapsed = time.perf_counter() - start

    # Relative: parallel must be at least 30% faster than serial. This is
    # what proves the work overlapped, independent of CI scheduler jitter.
    assert elapsed < serial_baseline * 0.7, (
        f"parallel ({elapsed * 1000:.0f}ms) was not "
        f"30% faster than serial ({serial_baseline * 1000:.0f}ms) — "
        f"speculation does not appear to be running concurrently"
    )
    # Absolute sanity: ~one sleep_per plus task-scheduling overhead.
    # 350ms is generous slack for loaded CI machines.
    assert elapsed < 0.35, (
        f"expected ~{sleep_per * 1000:.0f}ms parallel, got {elapsed * 1000:.0f}ms"
    )


async def test_end_cancels_pending_unmatched_speculations() -> None:
    """Unmatched speculations don't leak past the iteration boundary."""
    log: list[str] = []
    # Slow speculation — 100ms — so it's still pending at end() time.
    dispatcher = _dispatcher(
        idempotent=["search", "parse"],
        handler_log=log,
        slow_handlers={"parse": 0.10},
    )
    speculator = Speculator(LastCallPredictor(history_window=2), max_speculations=2)

    history = [
        Message(role="user", content=[ContentBlock(type="text", text="hi")]),
        Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name="search", arguments={"q": "x"}, id="a"),
                )
            ],
        ),
        Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_use=ToolCall(name="parse", arguments={"q": "x"}, id="b"),
                )
            ],
        ),
    ]
    await speculator.begin(
        history=history,
        agent=_agent(["search", "parse"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )

    # Resolve search instantly (its handler returns immediately).
    await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m"))

    # parse is still in flight (100ms sleep) when we end.
    start = time.perf_counter()
    await speculator.end()
    end_duration = time.perf_counter() - start

    # End should NOT have to wait the full 100ms — the cancel + drain
    # should be fast (the handler's sleep gets interrupted).
    assert end_duration < 0.10, f"end took {end_duration * 1000:.0f}ms — cancel slow"


async def test_predictor_returning_unknown_tool_is_silently_dropped() -> None:
    log: list[str] = []
    dispatcher = _dispatcher(idempotent=["real"], handler_log=log)

    class BadPredictor:
        def predict(
            self,
            history: list[Message],
            idempotent_tools: dict[str, Tool],
            max_predictions: int,
        ) -> list[ToolCall]:
            return [
                ToolCall(name="ghost", arguments={}),
                ToolCall(name="real", arguments={"q": "ok"}),
            ]

    speculator = Speculator(BadPredictor(), max_speculations=2)
    await speculator.begin(
        history=[],
        agent=_agent(["real"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )
    await asyncio.sleep(0.01)  # let scheduled tasks run before end() cancels
    await speculator.end()

    # Only the real tool was dispatched; ghost was filtered.
    assert log == ["real(ok)"]


# ---------------------------------------------------------------------------
# Predictor protocol / external strategies


def test_predictor_protocol_accepts_external_strategy() -> None:
    """Custom predictors satisfy the Protocol structurally — no inheritance."""

    class MyPredictor:
        def predict(
            self,
            history: list[Message],
            idempotent_tools: dict[str, Tool],
            max_predictions: int,
        ) -> list[ToolCall]:
            return []

    p: Predictor = MyPredictor()  # type: ignore[assignment]
    assert p.predict([], {}, 1) == []


async def test_hook_exception_during_speculation_does_not_crash_runner() -> None:
    """A buggy hook handler in the speculative path must not propagate
    out through `try_resolve` — wrong predictions are supposed to be
    cheap, and a whole-turn crash because the hook misbehaved during
    speculation is not cheap. The speculator wraps the dispatch in a
    try/except and surfaces the failure as an is_error=True ToolResult,
    which the runner can pass back to the model."""
    dispatcher = _dispatcher(idempotent=["search"])

    def buggy_hook(_event: PreToolUse) -> None:
        raise RuntimeError("simulated hook bug")

    hooks = HookRunner()
    hooks.register(PreToolUse, buggy_hook)

    speculator = Speculator(LastCallPredictor(history_window=1), max_speculations=1)
    await speculator.begin(
        history=_history_with_call("search", {"q": "x"}),
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=hooks,
    )

    # The speculation task ended in an exception. try_resolve must NOT
    # propagate it — it should return a ToolResult(is_error=True).
    result = await speculator.try_resolve(
        ToolCall(name="search", arguments={"q": "x"}, id="m")
    )
    await speculator.end()

    assert result is not None
    assert result.is_error is True
    assert "speculation error" in str(result.content)
    assert "simulated hook bug" in str(result.content)
    # And the result's id was patched to the model's call id.
    assert result.id == "m"
