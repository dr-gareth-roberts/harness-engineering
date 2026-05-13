"""Pytest helper that wires :func:`fuzz_tool` into a test function.

The wrapped coroutine is treated as the per-input invariant: if it
raises (including ``AssertionError``), the input is recorded as a
failure; otherwise the input passes.

If Hypothesis is not installed, the wrapped test is skipped via
``pytest.skip`` instead of erroring at import time, so a developer who
hasn't installed the ``[fuzz]`` extra still gets a clean signal.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.tools.dispatcher import Dispatcher


PayloadFn = Callable[[Any], Awaitable[None]]


def harness_property(
    *,
    dispatcher: Dispatcher,
    tool: str,
    n: int = 100,
    seed: int = 0,
) -> Callable[[PayloadFn], Callable[[], Awaitable[None]]]:
    """Deterministic input enumeration over a Hypothesis strategy.

    **This is NOT a Hypothesis property test.** Despite the name, the
    decorator runs ``Phase.generate`` only — it **does not shrink failing
    examples** (no ``Phase.shrink``) and **reports the first failure
    directly** by propagating the raised exception. There is no
    Hypothesis-style ``Falsifying example: ...`` summary, no shrunk
    minimal counterexample, and no on-disk reproducer database.

    For property-based testing with shrinking, use Hypothesis's
    ``@given`` directly.

    What the decorator does provide:

    * Validated Pydantic payloads drawn from the tool's ``input_model``
      via :func:`harness.fuzz.strategies.pydantic_strategy`.
    * Deterministic enumeration: same ``seed`` → same payload sequence
      → same first failing payload.
    * Graceful degradation: if the ``[fuzz]`` extra is not installed,
      the wrapped test ``pytest.skip``\\ s rather than erroring at import.

    The wrapped function receives one validated payload (a Pydantic
    instance for the tool's ``input_model``) per generated example and
    is expected to perform whatever assertions matter to the test. A
    raised exception fails the run on the first failing payload.

    Usage::

        @harness_property(dispatcher=disp, tool="parse_csv", n=50)
        async def parser_never_crashes(payload):
            result = await disp.dispatch(
                ToolCall(name="parse_csv", arguments=payload.model_dump())
            )
            assert isinstance(result.is_error, bool)
    """

    def decorate(test_fn: PayloadFn) -> Callable[[], Awaitable[None]]:
        @functools.wraps(test_fn)
        async def wrapper() -> None:
            try:
                import hypothesis  # noqa: F401
            except ImportError:
                import pytest

                pytest.skip(
                    "harness.fuzz requires the optional [fuzz] extra "
                    "(install with `pip install 'harness-engineering-toolkit[fuzz]'`)"
                )

            from harness.fuzz.runner import _generate_examples

            # Deferred: opt-in Phase.shrink — see audit/RELEASE-TODO.md M1.21.
            # Today this is a deterministic enumerator, not a shrinking property test.
            tool_obj = dispatcher.tools.get(tool)
            if tool_obj is None:
                raise KeyError(f"tool {tool!r} is not registered on the dispatcher")

            # We deliberately do not call `fuzz_tool` here: the user's
            # function *is* the invariant, and we want their assertions
            # to surface as test failures, not silent FuzzReport entries.
            import asyncio

            examples = await asyncio.to_thread(
                _generate_examples, tool_obj.input_model, n, seed, None
            )
            for example in examples:
                await test_fn(example)

        # Expose internals for white-box tests; not part of the public API.
        wrapper.__harness_fuzz__ = {  # type: ignore[attr-defined]
            "dispatcher": dispatcher,
            "tool": tool,
            "n": n,
            "seed": seed,
        }
        return wrapper

    return decorate


__all__ = ["harness_property"]
