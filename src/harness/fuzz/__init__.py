"""Hypothesis-based fuzzing for tool surfaces and agent flows.

Optional `[fuzz]` extra. The base install does not depend on Hypothesis;
imports of `hypothesis` are lazy so importing `harness.fuzz` succeeds even
when the extra is missing. The first call into a function that needs
Hypothesis (`pydantic_strategy`, `fuzz_tool`, `fuzz_agent`,
`harness_property`) raises a structured `ImportError` naming the extra.

Two modes:

* :func:`fuzz_tool` drives generated inputs through ``Dispatcher.dispatch``
  and reports any input that produced an unhandled exception or an error
  ``ToolResult``.
* :func:`fuzz_agent` drives generated inputs through a full
  ``Orchestrator`` turn (using a runner the caller supplies) and applies
  a user-defined invariant to the resulting assistant ``Message``.

For pytest integration, see :func:`harness_property`.
"""

from harness.fuzz.decorators import harness_property
from harness.fuzz.runner import (
    FuzzFailure,
    FuzzReport,
    fuzz_agent,
    fuzz_tool,
)
from harness.fuzz.strategies import (
    FuzzStrategyUnsupported,
    pydantic_strategy,
)

__all__ = [
    "FuzzFailure",
    "FuzzReport",
    "FuzzStrategyUnsupported",
    "fuzz_agent",
    "fuzz_tool",
    "harness_property",
    "pydantic_strategy",
]
