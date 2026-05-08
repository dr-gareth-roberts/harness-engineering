"""Vendor-specific Orchestrator runners.

Each submodule is gated on its own optional extra. Nothing in this package
is imported eagerly by `harness.<core module>`, so the base install has no
runtime dependency on any model SDK.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.runner.anthropic import AnthropicRunner

__all__ = ["AnthropicRunner"]


def __getattr__(name: str) -> Any:
    if name == "AnthropicRunner":
        from harness.runner.anthropic import AnthropicRunner

        return AnthropicRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
