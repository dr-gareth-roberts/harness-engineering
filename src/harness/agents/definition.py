from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SubAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    system_prompt: str
    allowed_tools: list[str] = []
    model: str = "claude-opus-4-7"
