"""harness-engineering — reusable building blocks for the layer around an LLM."""

from typing import TYPE_CHECKING, Any

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.memory import FileStore, InMemoryStore, Session, SessionRecord
from harness.policy import AllowList, DenyList
from harness.prompts import Message
from harness.replay import ReplayRunner, compare_sessions, run_eval
from harness.sandbox import PathPolicy, PathScope, safe_subprocess_run, scrub_env
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
    "PathPolicy",
    "PathScope",
    "ReplayRunner",
    "Session",
    "SessionRecord",
    "SubAgent",
    "Telemetry",
    "Tool",
    "__version__",
    "compare_sessions",
    "run_eval",
    "safe_subprocess_run",
    "scrub_env",
]


def __getattr__(name: str) -> Any:
    if name == "AnthropicRunner":
        from harness.runner import AnthropicRunner

        return AnthropicRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
