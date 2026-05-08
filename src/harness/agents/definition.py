from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SubAgent(BaseModel):
    """Vendor-neutral agent description.

    `model` is a free-form string the runner interprets — `"claude-opus-4-7"`
    for `AnthropicRunner`, `"gpt-5"` for an OpenAI-compatible runner, etc.
    Required, with no default, so the choice is always explicit and no vendor
    is privileged at the type level.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    system_prompt: str
    model: str
    allowed_tools: list[str] = []
