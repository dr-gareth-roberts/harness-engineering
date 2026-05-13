"""Top-level shared fixtures (M4.6).

Keeps the consolidation deliberately small: only the fixtures that are
actually consumed by more than one test file. Each addition here pays
for itself by removing a real duplicate elsewhere in the suite.

Vendor-shaped fakes (Anthropic / OpenAI SDK mimicry) stay in their
existing module-local `fakes.py` / `fakes_openai.py` because they bleed
SDK-specific surface and are not generic. The integration suite under
``tests/integration`` is the primary consumer of what's here.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from harness.agents.definition import SubAgent


class NoArgs(BaseModel):
    """Empty Pydantic input model for tools that take no arguments.

    Used by tests that register a placeholder tool just to satisfy the
    ``Dispatcher`` API without exercising the input-validation path.
    """


@pytest.fixture
def make_agent() -> Callable[..., SubAgent]:
    """Factory that builds a `SubAgent` with sensible test defaults.

    Usage::

        agent = make_agent()
        agent = make_agent(name="planner", allowed_tools=["search"])

    The `model` field defaults to ``"test-model"`` — a string the demo
    / fake runners ignore. Tests that care about a specific model
    identifier pass it explicitly.
    """

    def _make(
        *,
        name: str = "test-agent",
        system_prompt: str = "",
        model: str = "test-model",
        allowed_tools: list[str] | None = None,
    ) -> SubAgent:
        return SubAgent(
            name=name,
            system_prompt=system_prompt,
            model=model,
            allowed_tools=allowed_tools or [],
        )

    return _make
