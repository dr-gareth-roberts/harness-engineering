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
    hit_result = await speculator.try_resolve(ToolCall(name="parse", arguments={"q": "x"}, id="m2"))
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
    hooks.register(PreToolUse, lambda e: seen_pre.append(e.call))

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

    p: Predictor = MyPredictor()
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
    result = await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m"))
    await speculator.end()

    assert result is not None
    assert result.is_error is True
    assert "speculation error" in str(result.content)
    assert "simulated hook bug" in str(result.content)
    # And the result's id was patched to the model's call id.
    assert result.id == "m"


# ---------------------------------------------------------------------------
# Wave 6: per-event observe + cancel_unobserved


async def test_observe_marks_first_unobserved_matching_pending_spec() -> None:
    """`observe(call)` is a hint from the runner that the model emitted a
    tool_use matching `(call.name, call.arguments)`. The speculator marks
    the first unobserved pending entry that matches. Subsequent
    `cancel_unobserved` must leave that entry alone."""
    dispatcher = _dispatcher(idempotent=["search", "parse"], slow_handlers={"search": 0.20})
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

    # Observe a call that matches the slow `search` speculation. This
    # marks it as observed; cancel_unobserved should NOT cancel it.
    await speculator.observe(ToolCall(name="search", arguments={"q": "x"}, id="m1"))
    await speculator.cancel_unobserved()

    # `search` was observed → still resolvable.
    hit = await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m2"))
    assert hit is not None
    assert hit.content == "search-result-x"

    # `parse` was unobserved → cancelled by cancel_unobserved → no longer
    # in the pending list, so try_resolve returns None.
    miss = await speculator.try_resolve(ToolCall(name="parse", arguments={"q": "x"}, id="m3"))
    assert miss is None

    await speculator.end()


async def test_observe_with_no_match_is_a_noop() -> None:
    """If the model emits a tool_use that no speculation predicted,
    `observe` records nothing — and `cancel_unobserved` cancels everything."""
    log: list[str] = []
    dispatcher = _dispatcher(
        idempotent=["search"],
        handler_log=log,
        slow_handlers={"search": 0.20},
    )
    speculator = Speculator(LastCallPredictor(history_window=1), max_speculations=1)

    await speculator.begin(
        history=_history_with_call("search", {"q": "x"}),
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )

    # Observe a call that doesn't match the prediction. Should not raise
    # or otherwise affect state — just a no-op claim attempt.
    await speculator.observe(ToolCall(name="other", arguments={}, id="m1"))
    await speculator.cancel_unobserved()

    # The unmatched speculation is gone — try_resolve returns None.
    result = await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m2"))
    assert result is None

    await speculator.end()


async def test_cancel_unobserved_with_no_pending_is_noop() -> None:
    """Calling cancel_unobserved with nothing pending is harmless — used
    by the runner on iterations where `begin` returned without launching
    any speculation (no eligible idempotent tools)."""
    speculator = Speculator(LastCallPredictor(history_window=1), max_speculations=1)
    # Don't call begin — pending is empty.
    await speculator.cancel_unobserved()  # must not raise
    # Sanity: end() is also fine on empty pending.
    await speculator.end()


async def test_cancel_unobserved_runs_fast_when_handler_is_slow() -> None:
    """The performance claim of Wave 6: an unobserved spec gets cancelled
    immediately rather than running until `end`. Pin this with a slow
    handler — cancel_unobserved must complete in well under the
    handler's natural runtime."""
    dispatcher = _dispatcher(
        idempotent=["search"],
        slow_handlers={"search": 0.50},
    )
    speculator = Speculator(LastCallPredictor(history_window=1), max_speculations=1)

    await speculator.begin(
        history=_history_with_call("search", {"q": "x"}),
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )

    # Don't observe anything — model "emitted" nothing matching.
    start = time.perf_counter()
    await speculator.cancel_unobserved()
    elapsed = time.perf_counter() - start

    # Cancellation drain should be fast (~ms), nowhere near the 500ms handler.
    assert elapsed < 0.10, f"cancel_unobserved took {elapsed * 1000:.0f}ms — cancel slow"

    await speculator.end()


async def test_observe_then_try_resolve_resolves_observed_spec() -> None:
    """Happy path through Wave 6's lifecycle: observe during stream,
    cancel_unobserved after stream, try_resolve at dispatch time."""
    dispatcher = _dispatcher(idempotent=["search"])
    speculator = Speculator(LastCallPredictor(history_window=1), max_speculations=1)

    await speculator.begin(
        history=_history_with_call("search", {"q": "x"}),
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )

    # Stream emitted the predicted call.
    await speculator.observe(ToolCall(name="search", arguments={"q": "x"}, id="m"))
    # Stream ended; nothing to cancel because observation matched.
    await speculator.cancel_unobserved()
    # Now dispatch phase asks for the result.
    result = await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m"))
    await speculator.end()

    assert result is not None
    assert result.id == "m"
    assert result.content == "search-result-x"


async def test_observe_claims_separate_entries_for_duplicate_calls() -> None:
    """When the speculator launched two specs for the same `(name, args)`
    shape (rare but allowed), two `observe` calls with that shape claim
    separate entries. The second `try_resolve` for that shape must still
    find the second observed entry."""
    dispatcher = _dispatcher(idempotent=["search"])

    class TwicePredictor:
        """Returns the same call twice — the speculator launches two
        identical-shape pending tasks."""

        def predict(
            self,
            history: list[Message],
            idempotent_tools: dict[str, Tool],
            max_predictions: int,
        ) -> list[ToolCall]:
            return [
                ToolCall(name="search", arguments={"q": "x"}),
                ToolCall(name="search", arguments={"q": "x"}),
            ]

    speculator = Speculator(TwicePredictor(), max_speculations=2)
    await speculator.begin(
        history=[],
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )

    # Both observations claim distinct entries.
    await speculator.observe(ToolCall(name="search", arguments={"q": "x"}, id="m1"))
    await speculator.observe(ToolCall(name="search", arguments={"q": "x"}, id="m2"))
    await speculator.cancel_unobserved()

    # Both try_resolves succeed.
    r1 = await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m1"))
    r2 = await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m2"))
    await speculator.end()

    assert r1 is not None and r1.id == "m1"
    assert r2 is not None and r2.id == "m2"


# ---------------------------------------------------------------------------
# Wave 13b #2: eager per-block cancellation


async def test_observe_eagerly_cancels_lone_speculation_on_miss() -> None:
    """When max_speculations == 1 and the observed call doesn't match
    the lone pending spec, observe cancels it immediately. The
    motivating case is a slow handler that would otherwise keep
    burning runtime until cancel_unobserved at stream-end.

    The pin: a speculation with a 10-second sleep is cancelled within
    milliseconds of the first non-matching observe."""
    log: list[str] = []
    cancelled = asyncio.Event()

    async def slow_handler(args: _Args) -> str:
        try:
            log.append("start")
            await asyncio.sleep(10.0)
            log.append("done")
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "should-not-finish"

    dispatcher = Dispatcher(
        [
            Tool(
                name="search",
                description="",
                input_model=_Args,
                handler=slow_handler,
                idempotent=True,
            ),
        ]
    )

    speculator = Speculator(LastCallPredictor(history_window=1), max_speculations=1)
    await speculator.begin(
        history=_history_with_call("search", {"q": "x"}),
        agent=_agent(["search"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )

    # Give the speculation a tick to start.
    await asyncio.sleep(0.01)

    # Observe a different call — eager cancellation should fire.
    start = time.perf_counter()
    await speculator.observe(ToolCall(name="other_tool", arguments={}, id="m1"))
    elapsed = time.perf_counter() - start

    # observe drains the cancel synchronously (within itself), so
    # elapsed is dominated by cancellation work, not the 10s sleep.
    assert elapsed < 0.5, (
        f"observe should cancel within ~ms, took {elapsed * 1000:.0f}ms — "
        "eager cancellation did not fire"
    )
    # The handler observed cancellation (or never started — both are
    # valid wins).
    assert cancelled.is_set() or "start" not in log

    # Pending is now empty; subsequent try_resolve returns None.
    result = await speculator.try_resolve(ToolCall(name="search", arguments={"q": "x"}, id="m2"))
    assert result is None

    await speculator.end()


async def test_observe_does_not_eagerly_cancel_when_max_speculations_is_two() -> None:
    """With max_speculations > 1, the eager-cancel policy is *not*
    applied — an unmatched first observe might still see a matching
    second observe. Stream-end cancel_unobserved handles the cleanup."""
    log: list[str] = []
    dispatcher = _dispatcher(
        idempotent=["search", "parse"],
        handler_log=log,
        slow_handlers={"search": 0.05, "parse": 0.05},
    )

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

    speculator = Speculator(LastCallPredictor(history_window=2), max_speculations=2)
    await speculator.begin(
        history=history,
        agent=_agent(["search", "parse"]),
        dispatcher=dispatcher,
        hooks=HookRunner(),
    )

    # Observe a non-matching call. With max_speculations=2, eager
    # cancel doesn't fire — both pending specs survive.
    await speculator.observe(ToolCall(name="other_tool", arguments={}, id="m1"))

    # The matching observe arrives next; it claims `parse`.
    await speculator.observe(ToolCall(name="parse", arguments={"q": "x"}, id="m2"))

    # `search` is still pending and unobserved.
    assert len(speculator._pending) == 2  # noqa: SLF001 - test internals
    observed_count = sum(1 for e in speculator._pending if e.observed)  # noqa: SLF001
    assert observed_count == 1

    await speculator.cancel_unobserved()
    await speculator.end()
