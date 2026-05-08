"""Tool and agent fuzzers built on top of :mod:`harness.fuzz.strategies`.

Two entry points:

* :func:`fuzz_tool` — drives generated inputs through ``Dispatcher.dispatch``.
  A "failure" is any input that produced an unhandled exception or a
  ``ToolResult`` with ``is_error=True``. ``Dispatcher.dispatch`` already
  catches handler exceptions and turns them into error results, so most
  surface-level handler bugs surface as the latter.
* :func:`fuzz_agent` — drives generated inputs through a full
  ``Orchestrator`` turn and tests an invariant on the resulting
  assistant message.

Hypothesis is imported lazily, the same way the strategy bridge does, so
the package stays importable without the ``[fuzz]`` extra.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from harness.fuzz.strategies import _require_hypothesis, pydantic_strategy
from harness.prompts.messages import Message, text
from harness.tools.schema import ToolCall, ToolResult

if TYPE_CHECKING:
    from harness.agents.definition import SubAgent
    from harness.agents.orchestrator import Orchestrator
    from harness.tools.dispatcher import Dispatcher


Invariant = Callable[[Message], Awaitable[bool] | bool]


@dataclass(frozen=True)
class FuzzFailure:
    """A single failing input from a fuzz run.

    Either ``result`` or ``exception`` is populated, never both. The
    ``reproducer_path`` attribute carries Hypothesis' on-disk reproducer
    file path when available; today we let Hypothesis manage its own
    cache and surface that path is left as ``None``.
    """

    input: dict[str, Any]
    result: ToolResult | Message | None = None
    exception: BaseException | None = None
    reproducer_path: str | None = None


@dataclass
class FuzzReport:
    """Aggregate report for a fuzz run.

    The truthiness of the report mirrors success — a green run is falsy,
    so callers can ``if report: ...`` to branch on failures.
    """

    total: int = 0
    failures: list[FuzzFailure] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return self.total - len(self.failures)

    def __bool__(self) -> bool:  # truthy when any failures exist
        return bool(self.failures)


def _generate_examples(
    model_cls: type[BaseModel],
    n: int,
    seed: int,
    overrides: dict[str, Any] | None,
) -> list[BaseModel]:
    """Generate ``n`` validated Pydantic instances deterministically.

    Hypothesis is sync; we run a small ``@given`` collector to build the
    list, then return it for the async caller to iterate. ``derandomize``
    + ``database=None`` make the run stable across invocations.
    """

    _require_hypothesis()
    from hypothesis import HealthCheck, Phase, given, settings
    from hypothesis import seed as hyp_seed

    strategy = pydantic_strategy(model_cls, overrides=overrides)
    collected: list[BaseModel] = []

    @hyp_seed(seed)
    @settings(
        max_examples=n,
        database=None,
        derandomize=True,
        deadline=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(value=strategy)
    def _collect(value: BaseModel) -> None:
        collected.append(value)

    _collect()
    # Hypothesis may produce slightly fewer than n examples in some
    # configurations; trim defensively so the report `total` is exact.
    return collected[:n]


async def fuzz_tool(
    dispatcher: Dispatcher,
    tool_name: str,
    n: int = 20,
    *,
    seed: int = 0,
    overrides: dict[str, Any] | None = None,
) -> FuzzReport:
    """Fuzz a single tool through the dispatcher.

    For each generated example, build a :class:`ToolCall` and await
    ``dispatcher.dispatch``. A failure is any input that:

    * produced an unhandled exception (the dispatcher catches handler
      exceptions internally, so this is rare), or
    * produced a ``ToolResult`` with ``is_error=True``.
    """

    tool = dispatcher._tools.get(tool_name)
    if tool is None:
        raise KeyError(f"tool {tool_name!r} is not registered on the dispatcher")

    examples = await asyncio.to_thread(_generate_examples, tool.input_model, n, seed, overrides)

    report = FuzzReport(total=len(examples))
    for example in examples:
        payload = example.model_dump()
        try:
            result = await dispatcher.dispatch(ToolCall(name=tool_name, arguments=payload))
        except Exception as exc:  # noqa: BLE001 - the whole point of fuzzing
            report.failures.append(FuzzFailure(input=payload, exception=exc))
            continue
        if result.is_error:
            report.failures.append(FuzzFailure(input=payload, result=result))
    return report


async def fuzz_agent(
    orchestrator: Orchestrator,
    agent: SubAgent,
    tool_name: str,
    n: int = 20,
    *,
    invariant: Invariant,
    seed: int = 0,
    overrides: dict[str, Any] | None = None,
    prompt_template: Callable[[Any], str] | None = None,
) -> FuzzReport:
    """Fuzz a full agent turn for inputs derived from a tool's input model.

    For each generated input the orchestrator is driven with a synthetic
    user message that embeds the input. The user-supplied ``invariant``
    is applied to the resulting assistant message; ``False`` or a raised
    exception counts as a failure.

    The ``prompt_template`` lets callers control how the input is
    embedded into the user message. The default renders the
    ``model_dump`` dict, which keeps tests self-contained.
    """

    tool = orchestrator.dispatcher._tools.get(tool_name)
    if tool is None:
        raise KeyError(f"tool {tool_name!r} is not registered on the dispatcher")

    examples = await asyncio.to_thread(_generate_examples, tool.input_model, n, seed, overrides)

    report = FuzzReport(total=len(examples))
    for example in examples:
        payload = example.model_dump()
        if prompt_template is None:
            prompt = f"Use {tool_name} with arguments {payload}."
        else:
            prompt = prompt_template(example)
        messages = [text("user", prompt)]
        try:
            response = await orchestrator.run(agent, messages)
        except Exception as exc:  # noqa: BLE001
            report.failures.append(FuzzFailure(input=payload, exception=exc))
            continue
        try:
            outcome = invariant(response)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except Exception as exc:  # noqa: BLE001
            report.failures.append(FuzzFailure(input=payload, result=response, exception=exc))
            continue
        if not outcome:
            report.failures.append(FuzzFailure(input=payload, result=response))
    return report
