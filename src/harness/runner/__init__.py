"""Concrete `Runner` implementations.

Every runner in this package satisfies the same protocol:

    Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]

defined in `harness.agents.orchestrator`. Vendor-specific runners
(`anthropic`, `openai_compat`) are gated behind their own optional extras
and lazy-loaded — `import harness.runner` does not require any vendor SDK.

The base install ships `EchoRunner` and `CannedRunner` from
`harness.runner.demo` for tests and smoke checks.

To add another vendor: create `harness/runner/<vendor>.py` with a class
satisfying the protocol, register it in this `__init__`, and add an
optional extra in `pyproject.toml`. Nothing in `harness.agents`,
`harness.tools`, etc. needs to change.
"""

from typing import TYPE_CHECKING, Any

from harness.runner.demo import CannedRunner, EchoRunner

if TYPE_CHECKING:
    from harness.runner.anthropic import AnthropicRunner
    from harness.runner.openai_compat import OpenAICompatRunner

__all__ = [
    "AnthropicRunner",
    "CannedRunner",
    "EchoRunner",
    "OpenAICompatRunner",
]


def __getattr__(name: str) -> Any:
    if name == "AnthropicRunner":
        from harness.runner.anthropic import AnthropicRunner

        return AnthropicRunner
    if name == "OpenAICompatRunner":
        from harness.runner.openai_compat import OpenAICompatRunner

        return OpenAICompatRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
