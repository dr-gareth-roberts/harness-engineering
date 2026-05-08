"""Tests for the ``harness_property`` pytest helper."""

from __future__ import annotations

import sys

import pytest
from pydantic import BaseModel

pytest.importorskip("hypothesis")

from harness.fuzz.decorators import harness_property  # noqa: E402
from harness.tools import Dispatcher, Tool, ToolCall  # noqa: E402


class _StringIn(BaseModel):
    raw: str


def _identity(args: _StringIn) -> str:
    return args.raw


async def test_decorator_runs_test_n_times_for_each_payload() -> None:
    dispatcher = Dispatcher(
        [
            Tool(
                name="echo",
                description="Echo a string.",
                input_model=_StringIn,
                handler=_identity,
            )
        ]
    )

    seen: list[_StringIn] = []

    @harness_property(dispatcher=dispatcher, tool="echo", n=15, seed=0)
    async def each_payload(payload: _StringIn) -> None:
        seen.append(payload)
        result = await dispatcher.dispatch(ToolCall(name="echo", arguments=payload.model_dump()))
        assert result.is_error is False

    await each_payload()
    assert len(seen) == 15
    assert all(isinstance(p, _StringIn) for p in seen)


async def test_decorator_surfaces_assertion_errors() -> None:
    dispatcher = Dispatcher(
        [
            Tool(
                name="echo",
                description="Echo.",
                input_model=_StringIn,
                handler=_identity,
            )
        ]
    )

    @harness_property(dispatcher=dispatcher, tool="echo", n=5, seed=0)
    async def always_fails(payload: _StringIn) -> None:
        msg = "intentional"
        raise AssertionError(msg)

    with pytest.raises(AssertionError, match="intentional"):
        await always_fails()


async def test_decorator_unknown_tool_raises_keyerror() -> None:
    dispatcher = Dispatcher()

    @harness_property(dispatcher=dispatcher, tool="missing", n=2, seed=0)
    async def never_runs(payload: _StringIn) -> None:  # pragma: no cover
        pass

    with pytest.raises(KeyError, match="not registered"):
        await never_runs()


async def test_decorator_skips_when_hypothesis_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the [fuzz] extra is missing the wrapper raises ``pytest.skip``.

    We simulate that by knocking ``hypothesis`` out of ``sys.modules``
    via ``monkeypatch.setitem(..., None)``, which makes the in-wrapper
    import fail. The wrapper should swallow the ImportError and call
    ``pytest.skip`` instead, leaving the surrounding test harness
    responsible for marking the test skipped.
    """

    dispatcher = Dispatcher(
        [
            Tool(
                name="echo",
                description="Echo.",
                input_model=_StringIn,
                handler=_identity,
            )
        ]
    )

    @harness_property(dispatcher=dispatcher, tool="echo", n=2, seed=0)
    async def runs_only_with_hypothesis(payload: _StringIn) -> None:  # pragma: no cover
        pass

    for name in list(sys.modules):
        if name == "hypothesis" or name.startswith("hypothesis."):
            monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(pytest.skip.Exception):
        await runs_only_with_hypothesis()
