"""harness-engineering — reusable building blocks for the layer around an LLM."""

from typing import TYPE_CHECKING, Any

from harness.agents import Orchestrator, SubAgent
from harness.attribute import (
    AttributionChunk,
    AttributionResult,
    JaccardSimilarity,
    LengthRatio,
    attribute,
)
from harness.contracts import (
    Contract,
    ContractViolation,
    Violation,
    attach_contracts,
    check,
)
from harness.fuzz import FuzzReport, fuzz_agent, fuzz_tool, harness_property
from harness.hooks import HookRunner
from harness.memory import FileStore, InMemoryStore, Session, SessionRecord
from harness.policy import AllowList, DenyList
from harness.prompts import Message
from harness.replay import (
    DeleteTurn,
    DiffMatrix,
    DiffOutlier,
    InsertTurn,
    Mutation,
    ReplaceToolResult,
    ReplayRunner,
    RewriteTurn,
    compare_sessions,
    counterfactual,
    diff_eval,
    run_eval,
)
from harness.runner import CannedRunner, EchoRunner
from harness.sandbox import PathPolicy, PathScope, safe_subprocess_run, scrub_env
from harness.telemetry import JSONLSink, MemorySink, Telemetry
from harness.tools import Dispatcher, Tool

if TYPE_CHECKING:
    from harness.runner.anthropic import AnthropicRunner
    from harness.runner.openai_compat import OpenAICompatRunner

__version__ = "0.0.1"

__all__ = [
    "AllowList",
    "AnthropicRunner",
    "AttributionChunk",
    "AttributionResult",
    "CannedRunner",
    "Contract",
    "ContractViolation",
    "DeleteTurn",
    "DenyList",
    "DiffMatrix",
    "DiffOutlier",
    "Dispatcher",
    "EchoRunner",
    "FileStore",
    "FuzzReport",
    "HookRunner",
    "InMemoryStore",
    "InsertTurn",
    "JSONLSink",
    "JaccardSimilarity",
    "LengthRatio",
    "MemorySink",
    "Message",
    "Mutation",
    "OpenAICompatRunner",
    "Orchestrator",
    "PathPolicy",
    "PathScope",
    "ReplaceToolResult",
    "ReplayRunner",
    "RewriteTurn",
    "Session",
    "SessionRecord",
    "SubAgent",
    "Telemetry",
    "Tool",
    "Violation",
    "__version__",
    "attach_contracts",
    "attribute",
    "check",
    "compare_sessions",
    "counterfactual",
    "diff_eval",
    "fuzz_agent",
    "fuzz_tool",
    "harness_property",
    "run_eval",
    "safe_subprocess_run",
    "scrub_env",
]


def __getattr__(name: str) -> Any:
    if name == "AnthropicRunner":
        from harness.runner import AnthropicRunner

        return AnthropicRunner
    if name == "OpenAICompatRunner":
        from harness.runner import OpenAICompatRunner

        return OpenAICompatRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
