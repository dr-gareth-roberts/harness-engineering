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
from harness.cache import (
    DriftEvent,
    DriftReport,
    FileFingerprintStore,
    PrefixWatcher,
)
from harness.contracts import (
    Contract,
    ContractViolation,
    Never,
    RoleIs,
    TextMatches,
    Violation,
    attach_contracts,
    check,
)
from harness.debug import (
    DapAdapter,
    DapProtocolError,
    DebugAborted,
    DebugContext,
    DebugRunner,
)
from harness.fuzz import FuzzReport, fuzz_agent, fuzz_tool, harness_property
from harness.hooks import (
    Event,
    HookDecision,
    HookRunner,
    PauseTurn,
    PostAssistantMessage,
    PostToolUse,
    PreToolUse,
    PromptSubmit,
    Refusal,
    SessionEnd,
    SessionStart,
    Stop,
)
from harness.memory import FileStore, InMemoryStore, PromptBlocked, Session, SessionRecord
from harness.plan import (
    Plan,
    PlanGuardedRunner,
    PlannedToolCall,
    PlanViolation,
    infer_plan_from_records,
)
from harness.policy import (
    AllowList,
    ArgumentMatcher,
    DenyList,
    Policy,
    attach_pre_tool_policies,
)
from harness.privacy import (
    HIPAA_PACK,
    PII_PACK,
    SECRET_PACK,
    EntropyDetector,
    PresidioDetector,
    PrivacyBoundary,
    PrivacyViolation,
    RegexDetector,
    build_pii_pack,
)
from harness.prompts import (
    Message,
    assistant_tool_use,
    attach_file,
    attach_image,
    compact,
    text,
    user_tool_result,
)
from harness.replay import (
    DeleteTurn,
    DiffMatrix,
    DiffOutlier,
    EvalCase,
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
from harness.speculate import (
    CrossSessionPredictor,
    LastCallPredictor,
    SequencePredictor,
    Speculator,
)
from harness.streaming import (
    MessageEnd,
    StreamEvent,
    StreamingRunner,
    TextDelta,
    ToolUseEnd,
    ToolUseStart,
)
from harness.telemetry import JSONLSink, MemorySink, MultiSink, NullSink, Sink, Telemetry
from harness.telemetry.events import TelemetryEvent
from harness.tools import Dispatcher, Tool

if TYPE_CHECKING:
    from harness.runner.anthropic import AnthropicRunner
    from harness.runner.openai_compat import OpenAICompatRunner
    from harness.telemetry.otel import OpenTelemetrySink

__version__ = "1.3.0"

__all__ = [
    "AllowList",
    "AnthropicRunner",
    "ArgumentMatcher",
    "AttributionChunk",
    "AttributionResult",
    "CannedRunner",
    "Contract",
    "ContractViolation",
    "CrossSessionPredictor",
    "DapAdapter",
    "DapProtocolError",
    "DebugAborted",
    "DebugContext",
    "DebugRunner",
    "DeleteTurn",
    "DenyList",
    "DiffMatrix",
    "DiffOutlier",
    "Dispatcher",
    "DriftEvent",
    "DriftReport",
    "EchoRunner",
    "EntropyDetector",
    "EvalCase",
    "Event",
    "FileFingerprintStore",
    "FileStore",
    "FuzzReport",
    "HIPAA_PACK",
    "HookDecision",
    "HookRunner",
    "InMemoryStore",
    "InsertTurn",
    "JSONLSink",
    "JaccardSimilarity",
    "LastCallPredictor",
    "LengthRatio",
    "MemorySink",
    "Message",
    "MessageEnd",
    "MultiSink",
    "Mutation",
    "Never",
    "NullSink",
    "OpenAICompatRunner",
    "OpenTelemetrySink",
    "Orchestrator",
    "PII_PACK",
    "PathPolicy",
    "PathScope",
    "PauseTurn",
    "Plan",
    "PlanGuardedRunner",
    "PlanViolation",
    "PlannedToolCall",
    "Policy",
    "PostAssistantMessage",
    "PostToolUse",
    "PreToolUse",
    "PrefixWatcher",
    "PresidioDetector",
    "PrivacyBoundary",
    "PrivacyViolation",
    "PromptBlocked",
    "PromptSubmit",
    "Refusal",
    "RegexDetector",
    "ReplaceToolResult",
    "ReplayRunner",
    "RewriteTurn",
    "RoleIs",
    "SECRET_PACK",
    "SequencePredictor",
    "Session",
    "SessionEnd",
    "SessionRecord",
    "SessionStart",
    "Sink",
    "Speculator",
    "Stop",
    "StreamEvent",
    "StreamingRunner",
    "SubAgent",
    "Telemetry",
    "TelemetryEvent",
    "TextDelta",
    "TextMatches",
    "Tool",
    "ToolUseEnd",
    "ToolUseStart",
    "Violation",
    "__version__",
    "assistant_tool_use",
    "attach_contracts",
    "attach_file",
    "attach_image",
    "attach_pre_tool_policies",
    "attribute",
    "build_pii_pack",
    "check",
    "compact",
    "compare_sessions",
    "counterfactual",
    "diff_eval",
    "fuzz_agent",
    "fuzz_tool",
    "harness_property",
    "infer_plan_from_records",
    "run_eval",
    "safe_subprocess_run",
    "scrub_env",
    "text",
    "user_tool_result",
]


def __getattr__(name: str) -> Any:
    if name == "AnthropicRunner":
        from harness.runner import AnthropicRunner

        return AnthropicRunner
    if name == "OpenAICompatRunner":
        from harness.runner import OpenAICompatRunner

        return OpenAICompatRunner
    if name == "OpenTelemetrySink":
        from harness.telemetry.otel import OpenTelemetrySink

        return OpenTelemetrySink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
