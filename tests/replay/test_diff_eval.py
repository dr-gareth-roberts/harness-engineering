"""Tests for `harness.replay.diff_eval` (the differential cross-provider runner).

Covers the 8 scenarios from the spec:

1. one case x three runners -> three sessions
2. five cases x three runners -> 15 sessions
3. ``unanimous()`` returns all-agree cases
4. ``outliers()`` returns disagreeing cases
5. cluster detection: 2 agree, 1 dissents -> dissenter identified
6. HTML report renders without errors
7. concurrency: wall time ~ slowest runner * cases (gathered, not serialised)
8. telemetry: every session emits an OrchestratorTurn
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from harness.agents.definition import SubAgent
from harness.memory.record import SessionRecord
from harness.prompts.messages import Message, text
from harness.replay.diff_eval import DiffMatrix, DiffOutlier, diff_eval
from harness.replay.harness import EvalCase
from harness.runner.demo import CannedRunner, EchoRunner
from harness.telemetry.events import OrchestratorTurn
from harness.telemetry.recorder import Telemetry
from harness.telemetry.sinks import MemorySink

Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]


def make_agent(name: str = "t") -> SubAgent:
    return SubAgent(name=name, system_prompt="", model="test-model")


# ---------------------------------------------------------------------------
# Test 1: one case, three runners


async def test_diff_eval_runs_one_case_against_three_runners() -> None:
    cases = [EvalCase(name="hello", prompts=["hi"])]
    runners: dict[str, Runner] = {
        "echo-a": EchoRunner(prefix="A:"),
        "echo-b": EchoRunner(prefix="B:"),
        "echo-c": EchoRunner(prefix="C:"),
    }

    matrix = await diff_eval(cases, agent=make_agent(), runners=runners)

    assert isinstance(matrix, DiffMatrix)
    assert matrix.runner_names == ["echo-a", "echo-b", "echo-c"]
    assert [c.name for c in matrix.cases] == ["hello"]
    assert len(matrix.records) == 3
    for runner_name in runners:
        record = matrix.records[("hello", runner_name)]
        assert isinstance(record, SessionRecord)


# ---------------------------------------------------------------------------
# Test 2: 5x3 product


async def test_diff_eval_produces_full_cartesian_matrix() -> None:
    cases = [EvalCase(name=f"case-{i}", prompts=[f"q{i}"]) for i in range(5)]
    runners: dict[str, Runner] = {
        "r1": EchoRunner(prefix="1:"),
        "r2": EchoRunner(prefix="2:"),
        "r3": EchoRunner(prefix="3:"),
    }

    matrix = await diff_eval(cases, agent=make_agent(), runners=runners)

    assert len(matrix.cases) == 5
    assert len(matrix.runner_names) == 3
    assert len(matrix.records) == 15
    for case in cases:
        for runner_name in runners:
            record = matrix.records[(case.name, runner_name)]
            assert isinstance(record, SessionRecord)


# ---------------------------------------------------------------------------
# Test 3: unanimous() returns all-agree cases


async def test_unanimous_returns_cases_where_all_runners_agree() -> None:
    cases = [
        EvalCase(name="agree", prompts=["hi"]),
        EvalCase(name="disagree", prompts=["hi"]),
    ]
    # Three identical EchoRunners agree on every case.
    # For the "disagree" case, swap in a different runner via per-runner
    # canned replies — but we want consistency across cases per runner.
    # Easiest: use two runs with EchoRunner (same behaviour) for "agree",
    # and one runner that diverges on "disagree" only. That requires
    # per-case behaviour, which a CannedRunner gives us.
    runners: dict[str, Runner] = {
        "echo-1": EchoRunner(),
        "echo-2": EchoRunner(),
        # Different reply on second case ("disagree"): "DIFFERENT" vs the
        # echoed "hi" the other two produce.
        "rogue": CannedRunner(replies=["hi", "DIFFERENT"]),
    }

    matrix = await diff_eval(cases, agent=make_agent(), runners=runners)

    unanimous_cases = matrix.unanimous()
    assert [c.name for c in unanimous_cases] == ["agree"]


# ---------------------------------------------------------------------------
# Test 4: outliers(threshold=1.0) returns disagreeing cases


async def test_outliers_returns_cases_with_at_least_one_dissent() -> None:
    cases = [
        EvalCase(name="agree", prompts=["hi"]),
        EvalCase(name="disagree", prompts=["hi"]),
    ]
    runners: dict[str, Runner] = {
        "echo-1": EchoRunner(),
        "echo-2": EchoRunner(),
        "rogue": CannedRunner(replies=["hi", "DIFFERENT"]),
    }

    matrix = await diff_eval(cases, agent=make_agent(), runners=runners)

    outliers = matrix.outliers(threshold=1.0)
    outlier_case_names = {o.case.name for o in outliers}
    assert outlier_case_names == {"disagree"}


# ---------------------------------------------------------------------------
# Test 5: cluster detection — 2 agree, 1 dissents


async def test_cluster_detection_identifies_dissenter_by_name() -> None:
    case = EvalCase(name="probe", prompts=["hi"])
    runners: dict[str, Runner] = {
        "alpha": EchoRunner(),
        "beta": EchoRunner(),
        "gamma": CannedRunner(replies=["DIFFERENT"]),
    }

    matrix = await diff_eval([case], agent=make_agent(), runners=runners)

    outliers = matrix.outliers()
    assert len(outliers) == 1
    out = outliers[0]
    assert isinstance(out, DiffOutlier)
    assert out.dissenting_runner == "gamma"
    assert set(out.consensus_runners) == {"alpha", "beta"}
    # And the diff records the actual disagreement on the assistant turn.
    assert out.diff.matches is False


# ---------------------------------------------------------------------------
# Test 6: HTML report renders without errors


async def test_report_html_renders_with_case_and_runner_names(tmp_path: Path) -> None:
    cases = [
        EvalCase(name="alpha-case", prompts=["hi"]),
        EvalCase(name="beta-case", prompts=["hi"]),
    ]
    runners: dict[str, Runner] = {
        "echo-1": EchoRunner(),
        "echo-2": EchoRunner(),
        "rogue": CannedRunner(replies=["hi", "DIFFERENT"]),
    }

    matrix = await diff_eval(cases, agent=make_agent(), runners=runners)

    out = tmp_path / "diff.html"
    matrix.report_html(str(out))

    assert out.exists()
    rendered = out.read_text(encoding="utf-8")
    # Case names and runner names appear in the rendered file.
    for case in cases:
        assert case.name in rendered
    for runner_name in runners:
        assert runner_name in rendered
    # Should be valid-ish HTML — has <html, </html>, and at least one <table>.
    assert "<html" in rendered.lower()
    assert "</html>" in rendered.lower()
    assert "<table" in rendered.lower()


# ---------------------------------------------------------------------------
# Test 7: concurrency — wall time ~ slowest runner * cases


class _SlowRunner:
    """Sleeps for ``delay`` seconds, then returns the user text echoed back."""

    def __init__(self, delay: float, prefix: str = "") -> None:
        self._delay = delay
        self._prefix = prefix

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        await asyncio.sleep(self._delay)
        last_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                for block in msg.content:
                    if block.type == "text" and block.text:
                        last_user = block.text
                        break
                break
        return text("assistant", self._prefix + last_user)


async def test_runners_execute_concurrently_not_serially() -> None:
    cases = [EvalCase(name=f"c{i}", prompts=["hi"]) for i in range(3)]
    delay = 0.05  # 50ms per call
    runners: dict[str, Runner] = {
        "r1": _SlowRunner(delay, prefix="1:"),
        "r2": _SlowRunner(delay, prefix="2:"),
        "r3": _SlowRunner(delay, prefix="3:"),
    }

    start = time.perf_counter()
    matrix = await diff_eval(cases, agent=make_agent(), runners=runners)
    elapsed = time.perf_counter() - start

    # Serial: 3 runners x 3 cases x 0.05s = 0.45s.
    # Parallel (gathered at runner level): 3 cases x 0.05s = 0.15s plus overhead.
    # Allow generous overhead for slow CI machines but still strictly less
    # than the serial time.
    serial_estimate = len(runners) * len(cases) * delay
    parallel_estimate = len(cases) * delay
    assert elapsed < serial_estimate * 0.75, (
        f"wall time {elapsed:.3f}s suggests serialised execution; "
        f"expected near {parallel_estimate:.3f}s, serial would be {serial_estimate:.3f}s"
    )
    # Sanity check: matrix is fully populated.
    assert len(matrix.records) == 9


# ---------------------------------------------------------------------------
# Test 8: telemetry — every session emits an OrchestratorTurn


async def test_every_session_emits_orchestrator_turn_telemetry() -> None:
    cases = [
        EvalCase(name="a", prompts=["hi"]),
        EvalCase(name="b", prompts=["hi"]),
    ]
    runners: dict[str, Runner] = {
        "r1": EchoRunner(),
        "r2": EchoRunner(),
        "r3": EchoRunner(),
    }
    sink = MemorySink()
    telemetry = Telemetry(sink)

    matrix = await diff_eval(
        cases,
        agent=make_agent(name="probe"),
        runners=runners,
        telemetry=telemetry,
    )

    turn_events = [e for e in sink.events if isinstance(e, OrchestratorTurn)]
    # 2 cases x 3 runners = 6 sessions = 6 OrchestratorTurn events.
    assert len(turn_events) == 6
    # Every event names the agent we passed in.
    assert all(e.agent_name == "probe" for e in turn_events)
    # And the matrix is consistent.
    assert len(matrix.records) == 6


# ---------------------------------------------------------------------------
# Bonus: failure isolation — one runner raising does not kill the whole matrix.


class _ExplodingRunner:
    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        raise RuntimeError("provider down")


async def test_one_runner_failing_does_not_kill_the_others() -> None:
    cases = [EvalCase(name="probe", prompts=["hi"])]
    runners: dict[str, Runner] = {
        "alpha": EchoRunner(),
        "beta": _ExplodingRunner(),
        "gamma": EchoRunner(),
    }

    matrix = await diff_eval(cases, agent=make_agent(), runners=runners)

    assert isinstance(matrix.records[("probe", "alpha")], SessionRecord)
    assert isinstance(matrix.records[("probe", "gamma")], SessionRecord)
    failure = matrix.records[("probe", "beta")]
    assert isinstance(failure, RuntimeError)
    assert "provider down" in str(failure)

    # The errored runner shows up in outliers.
    outliers = matrix.outliers()
    dissenters = {o.dissenting_runner for o in outliers}
    assert "beta" in dissenters
