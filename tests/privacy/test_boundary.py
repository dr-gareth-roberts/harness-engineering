"""Boundary integration tests.

Covers the remaining spec tests from `designs/standout.md` §6:

3. `redact` replaces match with `[REDACTED:name]`.
4. `block` raises `PrivacyViolation` and the inner runner is never called.
5. `audit` passes through unchanged; emits an event.
6. `direction="outbound"` only triggers on outbound text.
7. `direction="inbound"` only triggers on the returned message.
8. Audit event does NOT contain the detected value (exhaustive check).
11. End-to-end: a fake runner that would have received a SSN sees `[REDACTED:us_ssn]`.
12. `wrap` returns an object satisfying the `Runner` protocol (drives an `Orchestrator`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.agents import Orchestrator, SubAgent
from harness.hooks import HookRunner
from harness.privacy import (
    AuditSink,
    DetectionEvent,
    EntropyDetector,
    PrivacyBoundary,
    PrivacyViolation,
    RegexDetector,
)
from harness.prompts.messages import Message, text
from harness.runner.demo import EchoRunner
from harness.telemetry import JSONLSink
from harness.tools.dispatcher import Dispatcher

# ---------------------------------------------------------------------------
# Helpers


def make_agent() -> SubAgent:
    return SubAgent(name="t", system_prompt="", model="test-model")


class RecordingRunner:
    """Captures the messages it was called with for inspection."""

    def __init__(self, reply_text: str = "ok") -> None:
        self.calls: list[list[Message]] = []
        self._reply_text = reply_text

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        # Defensive copy so the boundary's later in-place edits (none, but
        # belt-and-braces) don't corrupt the record.
        self.calls.append([m.model_copy(deep=True) for m in messages])
        return text("assistant", self._reply_text)


def make_event_capture() -> tuple[list[DetectionEvent], AuditSink]:
    """Build a list-backed audit sink for assertions on captured events."""
    captured: list[DetectionEvent] = []

    async def sink(event: DetectionEvent) -> None:
        captured.append(event)

    return captured, sink


# ---------------------------------------------------------------------------
# Action: redact


async def test_redact_replaces_match_with_placeholder() -> None:
    """Spec test 3."""
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", action="redact"),
        ],
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    await wrapped(
        make_agent(),
        [text("user", "the ssn is 123-45-6789, please process")],
    )

    sent = inner.calls[0][0].content[0].text or ""
    assert "[REDACTED:us_ssn]" in sent
    assert "123-45-6789" not in sent


async def test_redact_handles_multiple_matches_in_one_fragment() -> None:
    boundary = PrivacyBoundary(
        detectors=[RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b")],
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    await wrapped(
        make_agent(),
        [text("user", "first 111-22-3333 second 444-55-6666 done")],
    )

    sent = inner.calls[0][0].content[0].text or ""
    assert sent.count("[REDACTED:us_ssn]") == 2
    assert "111-22-3333" not in sent
    assert "444-55-6666" not in sent


# ---------------------------------------------------------------------------
# Action: block


async def test_block_raises_privacy_violation_and_skips_inner_runner() -> None:
    """Spec test 4."""
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector(
                "aws_access_key",
                r"\bAKIA[A-Z0-9]{16}\b",
                action="block",
            ),
        ],
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    with pytest.raises(PrivacyViolation) as excinfo:
        await wrapped(
            make_agent(),
            [text("user", "key is AKIAIOSFODNN7EXAMPLE here")],
        )

    # Inner runner must never have been invoked once block fires.
    assert inner.calls == []
    # The exception's stringified form must not echo the secret.
    assert "AKIAIOSFODNN7EXAMPLE" not in str(excinfo.value)
    assert excinfo.value.detection.name == "aws_access_key"


async def test_block_per_detector_overrides_boundary_default() -> None:
    """Two detectors, two different actions: AWS blocks, SSN redacts."""
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector("aws_access_key", r"\bAKIA[A-Z0-9]{16}\b", action="block"),
            RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", action="redact"),
        ],
        on_detect="audit",
    )

    # SSN-only path: redact action wins despite boundary's `audit` default.
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)
    await wrapped(make_agent(), [text("user", "ssn 123-45-6789")])
    sent = inner.calls[0][0].content[0].text or ""
    assert "[REDACTED:us_ssn]" in sent

    # AWS path on a fresh wrapper: block action wins.
    inner2 = RecordingRunner()
    wrapped2 = boundary.wrap(inner2)
    with pytest.raises(PrivacyViolation):
        await wrapped2(make_agent(), [text("user", "AKIAIOSFODNN7EXAMPLE")])
    assert inner2.calls == []


# ---------------------------------------------------------------------------
# Action: audit


async def test_audit_passes_through_unchanged_and_emits_event() -> None:
    """Spec test 5."""
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[RegexDetector("us_phone", r"\b\d{3}-\d{3}-\d{4}\b", action="audit")],
        audit_sink=sink,
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    await wrapped(make_agent(), [text("user", "call me 555-867-5309")])

    # Pass-through: the inner runner saw the original payload.
    sent = inner.calls[0][0].content[0].text or ""
    assert sent == "call me 555-867-5309"
    # And the sink fired one DetectionEvent.
    assert len(captured) == 1
    assert captured[0].action == "audit"
    assert captured[0].name == "us_phone"


# ---------------------------------------------------------------------------
# Direction filtering


async def test_outbound_only_detector_does_not_fire_inbound() -> None:
    """Spec test 6."""
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector(
                "us_ssn",
                r"\b\d{3}-\d{2}-\d{4}\b",
                direction="outbound",
                action="audit",
            ),
        ],
        audit_sink=sink,
    )

    class LeakingRunner:
        async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
            return text("assistant", "here is one for you: 222-33-4444")

    wrapped = boundary.wrap(LeakingRunner())
    reply = await wrapped(make_agent(), [text("user", "no leaks here")])

    # No detection on inbound text, even though the SSN is there.
    assert reply.content[0].text == "here is one for you: 222-33-4444"
    assert captured == []


async def test_inbound_only_detector_does_not_fire_outbound() -> None:
    """Spec test 7."""
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector(
                "leak",
                r"INTERNAL_[A-Z0-9]{6,}",
                direction="inbound",
                action="redact",
            ),
        ],
        audit_sink=sink,
    )

    class ExfilRunner:
        async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
            return text("assistant", "trying to send INTERNAL_ABCDEF1 outwards")

    wrapped = boundary.wrap(ExfilRunner())
    # Outbound contains the same shape — must not trigger.
    reply = await wrapped(
        make_agent(),
        [text("user", "INTERNAL_ABCDEF1 should pass through outbound")],
    )

    # Outbound: no event.
    outbound_events = [e for e in captured if e.direction == "outbound"]
    assert outbound_events == []
    # Inbound: event fired and content redacted.
    inbound_events = [e for e in captured if e.direction == "inbound"]
    assert len(inbound_events) == 1
    assert "[REDACTED:leak]" in (reply.content[0].text or "")


# ---------------------------------------------------------------------------
# Audit-trail privacy guarantee (spec test 8)


async def test_audit_event_never_contains_detected_value() -> None:
    """Spec test 8 — the load-bearing privacy guarantee.

    Strategy: feed in a synthetic high-entropy secret; serialize the
    `DetectionEvent` to JSON; assert the secret string does not appear
    anywhere in the serialized form. This catches both per-field leakage
    *and* accidental string interpolation in any future field.
    """
    secret = "9aF3qZ7kP2vB8nC4xS6tR1dE5wY0uM7jL9hG3bN8oI4cV2pKsynth"
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            EntropyDetector(min_entropy=4.0, min_length=24, action="audit"),
        ],
        audit_sink=sink,
    )

    class PassThroughRunner:
        async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
            return text("assistant", "ok")

    wrapped = boundary.wrap(PassThroughRunner())
    await wrapped(make_agent(), [text("user", f"secret={secret}")])

    assert len(captured) >= 1
    for event in captured:
        # Pydantic round-trip catches every reachable string field.
        serialized = event.model_dump_json()
        assert secret not in serialized, f"audit event leaked secret: {serialized!r}"
        # Defensive: check explicit fields too — a future schema change
        # that adds a `value` field would silently regress without this.
        for field_name, value in event.model_dump().items():
            assert secret != value, f"DetectionEvent.{field_name} equals the secret value"
            if isinstance(value, str):
                assert secret not in value, f"DetectionEvent.{field_name} contains the secret value"


async def test_audit_event_is_emitted_for_block_actions(tmp_path: Path) -> None:
    """Block events still hit the audit sink — blocked attempts must remain visible."""
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector("aws_access_key", r"\bAKIA[A-Z0-9]{16}\b", action="block"),
        ],
        audit_sink=sink,
    )
    wrapped = boundary.wrap(RecordingRunner())

    with pytest.raises(PrivacyViolation):
        await wrapped(
            make_agent(),
            [text("user", "key=AKIAIOSFODNN7EXAMPLE")],
        )
    assert len(captured) == 1
    assert captured[0].action == "block"


# ---------------------------------------------------------------------------
# End-to-end (spec test 11)


async def test_end_to_end_ssn_is_redacted_before_inner_runner() -> None:
    """Spec test 11 — the inner runner sees `[REDACTED:us_ssn]`, not the SSN."""
    boundary = PrivacyBoundary(
        detectors=[RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", action="redact")],
    )
    inner = RecordingRunner(reply_text="processed")
    wrapped = boundary.wrap(inner)

    await wrapped(
        make_agent(),
        [text("user", "process ssn 123-45-6789 now")],
    )

    sent_text = inner.calls[0][0].content[0].text or ""
    assert "[REDACTED:us_ssn]" in sent_text
    assert "123-45-6789" not in sent_text


# ---------------------------------------------------------------------------
# Runner protocol satisfaction (spec test 12)


async def test_wrapped_runner_drives_an_orchestrator() -> None:
    """Spec test 12 — wrap returns a `Runner`; Orchestrator drives it cleanly."""
    boundary = PrivacyBoundary(
        detectors=[RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b")],
    )
    wrapped = boundary.wrap(EchoRunner())

    orchestrator = Orchestrator(Dispatcher(), HookRunner(), runner=wrapped)
    reply = await orchestrator.run(
        make_agent(),
        [text("user", "echo this 123-45-6789 back")],
    )

    assert reply.role == "assistant"
    text_out = reply.content[0].text or ""
    # The user-side SSN was redacted before EchoRunner saw it; the echoed
    # reply therefore carries the redaction marker, not the SSN.
    assert "[REDACTED:us_ssn]" in text_out
    assert "123-45-6789" not in text_out


# ---------------------------------------------------------------------------
# JSONL sink integration (smoke test for the doc'd usage)


async def test_jsonl_sink_emit_works_as_audit_sink(tmp_path: Path) -> None:
    """The doc shows `JSONLSink("./privacy.jsonl")` — pass `.emit` as the sink."""
    path = tmp_path / "privacy.jsonl"
    boundary = PrivacyBoundary(
        detectors=[RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", action="redact")],
        audit_sink=JSONLSink(path).emit,
    )
    wrapped = boundary.wrap(RecordingRunner())
    await wrapped(make_agent(), [text("user", "ssn 123-45-6789 here")])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["name"] == "us_ssn"
    assert record["action"] == "redact"
    assert record["direction"] == "outbound"
    # No value field — privacy guarantee survives the JSONL serialization.
    assert "value" not in record
    assert "123-45-6789" not in lines[0]


# ---------------------------------------------------------------------------
# Extended scope: tool_use.arguments and tool_result.content
# (Wave 2 follow-up — boundary v1 scanned text blocks only.)


def _tool_use_msg(name: str, arguments: dict[str, object]) -> Message:
    """Build an assistant message containing a single tool_use block."""
    from harness.prompts.messages import ContentBlock
    from harness.tools.schema import ToolCall

    return Message(
        role="assistant",
        content=[
            ContentBlock(
                type="tool_use",
                tool_use=ToolCall(name=name, arguments=arguments, id="tu-1"),
            )
        ],
    )


def _tool_result_msg(content: object) -> Message:
    """Build a user message containing a single tool_result block."""
    from harness.prompts.messages import ContentBlock
    from harness.tools.schema import ToolResult

    return Message(
        role="user",
        content=[
            ContentBlock(
                type="tool_result",
                tool_result=ToolResult(id="tu-1", content=content),
            )
        ],
    )


async def test_tool_use_arguments_string_value_is_redacted() -> None:
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", action="redact")],
        audit_sink=sink,
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    msg = _tool_use_msg("save_user", {"name": "Alex", "ssn": "123-45-6789"})
    await wrapped(make_agent(), [msg])

    # The inner runner saw a redacted argument value, never the raw SSN.
    seen = inner.calls[0][0].content[0].tool_use
    assert seen is not None
    assert seen.arguments["ssn"] == "[REDACTED:us_ssn]"
    assert seen.arguments["name"] == "Alex"

    # Audit event carries the nested location path.
    assert len(captured) == 1
    assert captured[0].location == "messages[0].content[0].tool_use.arguments.ssn"
    assert captured[0].direction == "outbound"


async def test_tool_use_arguments_block_action_raises_before_inner_call() -> None:
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector("aws_key", r"\bAKIA[A-Z0-9]{16}\b", action="block"),
        ],
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    msg = _tool_use_msg("upload", {"key": "AKIAABCDEFGHIJKLMNOP", "bucket": "x"})

    with pytest.raises(PrivacyViolation) as exc_info:
        await wrapped(make_agent(), [msg])

    assert exc_info.value.detection.name == "aws_key"
    assert "tool_use.arguments.key" in exc_info.value.detection.location
    assert inner.calls == []


async def test_tool_result_string_content_is_redacted_inbound() -> None:
    """A tool result returning a SSN string is redacted before the caller sees it."""
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector(
                "us_ssn",
                r"\b\d{3}-\d{2}-\d{4}\b",
                direction="inbound",
                action="redact",
            ),
        ],
        audit_sink=sink,
    )

    class ToolResultRunner:
        async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
            return _tool_result_msg("Found user: SSN 123-45-6789")

    wrapped = boundary.wrap(ToolResultRunner())
    reply = await wrapped(make_agent(), [text("user", "lookup")])

    block = reply.content[0]
    assert block.tool_result is not None
    content = block.tool_result.content
    assert "[REDACTED:us_ssn]" in content
    assert "123-45-6789" not in content
    assert any(
        e.location == "messages[0].content[0].tool_result.content" for e in captured
    )


async def test_tool_result_nested_dict_redaction_uses_dotted_path() -> None:
    """A SSN nested inside a dict-shaped tool result is redacted; the audit
    event's location reflects the path that was walked."""
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector(
                "us_ssn",
                r"\b\d{3}-\d{2}-\d{4}\b",
                direction="inbound",
                action="redact",
            ),
        ],
        audit_sink=sink,
    )

    class NestedDictRunner:
        async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
            payload = {
                "user": {"name": "Alex", "identifiers": {"ssn": "123-45-6789"}},
                "ok": True,
            }
            return _tool_result_msg(payload)

    wrapped = boundary.wrap(NestedDictRunner())
    reply = await wrapped(make_agent(), [text("user", "lookup")])

    new = reply.content[0].tool_result
    assert new is not None
    assert new.content["user"]["identifiers"]["ssn"] == "[REDACTED:us_ssn]"
    assert new.content["user"]["name"] == "Alex"
    assert new.content["ok"] is True

    locations = [e.location for e in captured]
    assert (
        "messages[0].content[0].tool_result.content.user.identifiers.ssn" in locations
    )


async def test_tool_result_list_element_is_redacted_with_index_grammar() -> None:
    """A SSN buried in a list inside the tool result is redacted; the audit
    event's location uses `[n]` for list indices."""
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector(
                "us_ssn",
                r"\b\d{3}-\d{2}-\d{4}\b",
                direction="inbound",
                action="redact",
            ),
        ],
        audit_sink=sink,
    )

    class ListRunner:
        async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
            return _tool_result_msg(
                {"hits": ["clean", "ssn=123-45-6789", "also clean"]}
            )

    wrapped = boundary.wrap(ListRunner())
    reply = await wrapped(make_agent(), [text("user", "lookup")])
    content = reply.content[0].tool_result.content  # type: ignore[union-attr]
    assert content["hits"][1] == "ssn=[REDACTED:us_ssn]"
    assert content["hits"][0] == "clean"

    assert any(
        e.location == "messages[0].content[0].tool_result.content.hits[1]"
        for e in captured
    )


async def test_recursion_depth_cap_still_catches_deep_leaks() -> None:
    """Beyond `_MAX_RECURSION_DEPTH`, the subtree is stringified and scanned
    flat. Detection must still work even when nesting is pathological."""
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector(
                "us_ssn",
                r"\b\d{3}-\d{2}-\d{4}\b",
                direction="inbound",
                action="redact",
            ),
        ],
        audit_sink=sink,
    )

    # 10 levels deep — beyond the cap of 4.
    deep: dict[str, object] = {"ssn": "123-45-6789"}
    for _ in range(10):
        deep = {"nested": deep}

    class DeepRunner:
        async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
            return _tool_result_msg(deep)

    wrapped = boundary.wrap(DeepRunner())
    reply = await wrapped(make_agent(), [text("user", "lookup")])

    # The deep subtree was flat-scanned and the redacted serialization
    # replaced the inner subtree wholesale — at least one audit location
    # carries the [depth-cap] suffix.
    assert any("[depth-cap]" in e.location for e in captured)

    # The redaction marker reached the resulting payload via the flat-scan.
    payload_str = json.dumps(reply.content[0].tool_result.content)  # type: ignore[union-attr]
    assert "[REDACTED:us_ssn]" in payload_str
    assert "123-45-6789" not in payload_str
