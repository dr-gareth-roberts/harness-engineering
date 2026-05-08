"""harness-engineering — reusable building blocks for the layer around an LLM."""

from typing import TYPE_CHECKING, Any

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.memory import FileStore, InMemoryStore, Session, SessionRecord
from harness.policy import AllowList, DenyList
from harness.prompts import Message
from harness.telemetry import JSONLSink, MemorySink, Telemetry
from harness.tools import Dispatcher, Tool

if TYPE_CHECKING:
    from harness.runner.anthropic import AnthropicRunner

__version__ = "0.0.1"

__all__ = [
    "AllowList",
    "AnthropicRunner",
    "DenyList",
    "Dispatcher",
    "FileStore",
    "HookRunner",
    "InMemoryStore",
    "JSONLSink",
    "MemorySink",
    "Message",
    "Orchestrator",
    "Session",
    "SessionRecord",
    "SubAgent",
    "Telemetry",
    "Tool",
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name == "AnthropicRunner":
        from harness.runner import AnthropicRunner

        return AnthropicRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
