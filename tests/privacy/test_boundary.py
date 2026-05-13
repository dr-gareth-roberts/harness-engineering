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
    )

    # SSN-only path: redact action wins.
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
        # JSONLSink.emit accepts any TelemetryEvent; PrivacyBoundary's
        # audit_sink is typed for the narrower DetectionEvent. Callable
        # arg-types are contravariant — the broader signature is
        # behaviorally fine, but mypy can't see that without a wrapper.
        audit_sink=JSONLSink(path).emit,  # type: ignore[arg-type]
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
    assert any(e.location == "messages[0].content[0].tool_result.content" for e in captured)


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
    assert "messages[0].content[0].tool_result.content.user.identifiers.ssn" in locations


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
            return _tool_result_msg({"hits": ["clean", "ssn=123-45-6789", "also clean"]})

    wrapped = boundary.wrap(ListRunner())
    reply = await wrapped(make_agent(), [text("user", "lookup")])
    content = reply.content[0].tool_result.content  # type: ignore[union-attr]
    assert content["hits"][1] == "ssn=[REDACTED:us_ssn]"
    assert content["hits"][0] == "clean"

    assert any(e.location == "messages[0].content[0].tool_result.content.hits[1]" for e in captured)


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


# ---------------------------------------------------------------------------
# Dict-key leak vector (M1.12)


async def test_dict_key_shaped_like_secret_is_redacted_in_audit_location() -> None:
    """An attacker (or careless caller) can put a secret-shaped string in a
    dict key. The matched-value invariant must also hold for keys: the audit
    event's `location` must not echo the key verbatim.

    Leak vector: a SSN value lives **under** a secret-shaped key. Without
    sanitization, the SSN detection's audit event carries the raw key in
    `location`, leaking it. With sanitization, the key is replaced by
    `<redacted>` in `location` while the inner runner still sees the
    original key in `arguments`.
    """
    captured, sink = make_event_capture()
    secret_key = "AKIAIOSFODNN7EXAMPLE"
    boundary = PrivacyBoundary(
        detectors=[
            # The key detector must run on the key string (and on the
            # arguments dict — but `_sanitize_key` is what matters for the
            # leak path).
            RegexDetector("aws_key", r"\bAKIA[A-Z0-9]{16}\b", action="redact"),
            RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", action="redact"),
        ],
        audit_sink=sink,
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    # SSN nested *under* the secret-shaped key — this is the leak vector.
    msg = _tool_use_msg("upload", {secret_key: "ssn=123-45-6789"})
    await wrapped(make_agent(), [msg])

    # The inner runner still sees the original key in arguments (only the
    # *value* under it was redacted) — sanitization is for audit metadata
    # only.
    seen = inner.calls[0][0].content[0].tool_use
    assert seen is not None
    assert secret_key in seen.arguments

    # The SSN detection's audit event has a location path that walks
    # through the key — and that key must appear as `<redacted>`, never
    # verbatim.
    ssn_events = [e for e in captured if e.name == "us_ssn"]
    assert ssn_events, "expected SSN detection on the keyed value"
    for event in ssn_events:
        assert secret_key not in event.location, (
            f"SSN audit event leaked secret key into location: {event.location!r}"
        )
        # Concretely the location is the sanitized path:
        assert event.location == ("messages[0].content[0].tool_use.arguments.<redacted>"), (
            f"unexpected SSN location shape: {event.location!r}"
        )

    # Full serialization round-trip: no event may carry the secret-shaped
    # key in any field. This catches PrivacyViolation.__str__ leakage too,
    # since the violation's message is built from `detection.location`.
    for event in captured:
        assert secret_key not in event.model_dump_json()


async def test_dict_key_shaped_like_secret_is_redacted_under_nested_dict_value() -> None:
    """Sibling regression for the key-sanitization leak vector, but with
    the secret-shaped key landing **above a nested dict** rather than a
    scalar.

    The recursion path: `arguments` -> dict -> `{secret_key: {"inner":
    "ssn=..."}}` -> walk into the inner dict and detect the SSN there.
    The audit `location` must:

    1. Replace the secret-shaped key with `<redacted>` in the composed
       path (the original key-leak vector this whole test pins), and
    2. Preserve the inner-dict path segment (`.inner`) so callers can
       still locate where the detection fired without needing the raw
       key. The redaction lands at the right level (the key) and the
       nested structure underneath survives.
    """
    captured, sink = make_event_capture()
    secret_key = "AKIAIOSFODNN7EXAMPLE"
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector("aws_key", r"\bAKIA[A-Z0-9]{16}\b", action="redact"),
            RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", action="redact"),
        ],
        audit_sink=sink,
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    # SSN nested *under* a dict that is itself *under* the secret-shaped
    # key. Same leak vector, deeper structure.
    msg = _tool_use_msg("upload", {secret_key: {"inner": "ssn=123-45-6789"}})
    await wrapped(make_agent(), [msg])

    # Inner runner still sees the original key (sanitization is for audit
    # metadata only) and the nested dict shape is preserved.
    seen = inner.calls[0][0].content[0].tool_use
    assert seen is not None
    assert secret_key in seen.arguments
    assert isinstance(seen.arguments[secret_key], dict)
    assert "inner" in seen.arguments[secret_key]

    # The SSN detection's audit location walks through the redacted key
    # *and* the inner-dict path segment — proving the redaction lands at
    # the right level (the key) without flattening the path below it.
    ssn_events = [e for e in captured if e.name == "us_ssn"]
    assert ssn_events, "expected SSN detection on the nested keyed value"
    for event in ssn_events:
        assert secret_key not in event.location, (
            f"SSN audit event leaked secret key into location: {event.location!r}"
        )
        assert event.location == ("messages[0].content[0].tool_use.arguments.<redacted>.inner"), (
            f"unexpected SSN location shape: {event.location!r}"
        )

    # Full serialization round-trip — same invariant as the scalar case.
    for event in captured:
        assert secret_key not in event.model_dump_json()


# ---------------------------------------------------------------------------
# Overlapping detection ranges (M1.13)


async def test_overlapping_redactions_merge_into_one_marker() -> None:
    """Two detectors firing on overlapping ranges must merge before splice.

    Without merging, the right-to-left splice corrupts text: the wider
    redaction's placeholder is inserted, then the narrow detection's
    start/end indices (computed against the *original* string) point into
    the placeholder and the second splice mangles it. Concretely, the
    buggy output for this fixture would be
    `"value=[REDACTED:wide]TED:narrow] end"` — a leaked fragment.

    Expectation after the fix: exactly one `[REDACTED:wide]` span replacing
    the merged range, with surrounding text untouched.
    """
    boundary = PrivacyBoundary(
        detectors=[
            # Wide detector: matches the full literal "secretXYZ123".
            RegexDetector("wide", r"secretXYZ123", action="redact"),
            # Narrow detector: matches the substring "XYZ123" (overlaps the
            # tail of `wide`).
            RegexDetector("narrow", r"XYZ123", action="redact"),
        ],
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    await wrapped(make_agent(), [text("user", "value=secretXYZ123 end")])

    sent = inner.calls[0][0].content[0].text or ""
    # Exact-match: any double-splice corruption changes the output shape.
    # The merged span keeps the earlier-and-wider detection's name.
    assert sent == "value=[REDACTED:wide] end", f"unexpected splice output: {sent!r}"


async def test_overlapping_redactions_wider_first_merges_narrow_inside() -> None:
    """Reverse-order overlap: a narrow match fully inside a wider match.

    Tests the case where one range fully contains another (vs. partial
    overlap above). The buggy output here would be
    `"value=[REDACTED:outer_phrase]er_word]-suffix done"` — again a leak
    fragment. The fix produces a single clean splice.
    """
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector("inner_word", r"inside", action="redact"),
            RegexDetector("outer_phrase", r"prefix-inside-suffix", action="redact"),
        ],
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    await wrapped(make_agent(), [text("user", "value=prefix-inside-suffix done")])

    sent = inner.calls[0][0].content[0].text or ""
    assert sent == "value=[REDACTED:outer_phrase] done", f"unexpected splice output: {sent!r}"


# ---------------------------------------------------------------------------
# Multimodal-aware metadata scanning (M2.9)
#
# Image and file blocks were previously passed through unchanged. The
# boundary now scans image *metadata* (URL when url-sourced, media_type)
# and file *metadata* (file_id, path). Base64 image content and file
# bodies remain out of scope — they require OCR / file fetch and are
# documented as a pre-pass concern.


def _image_block_url(url: str, *, media_type: str = "image/png") -> Message:
    """Build a user message with a URL-sourced image block."""
    from harness.prompts.messages import ContentBlock, ImageRef

    return Message(
        role="user",
        content=[
            ContentBlock(
                type="image",
                image=ImageRef(source="url", media_type=media_type, data=url),
            )
        ],
    )


def _image_block_base64(data: str, *, media_type: str = "image/png") -> Message:
    """Build a user message with a base64-sourced image block."""
    from harness.prompts.messages import ContentBlock, ImageRef

    return Message(
        role="user",
        content=[
            ContentBlock(
                type="image",
                image=ImageRef(source="base64", media_type=media_type, data=data),
            )
        ],
    )


def _file_block(*, file_id: str | None = None, path: str | None = None) -> Message:
    """Build a user message with a file block carrying optional metadata."""
    from harness.prompts.messages import ContentBlock

    return Message(
        role="user",
        content=[ContentBlock(type="file", file_id=file_id, path=path)],
    )


async def test_image_url_metadata_is_scanned_and_fires_entropy_detector() -> None:
    """A URL-sourced image whose URL embeds a high-entropy token is detected.

    The detector pipeline runs over the URL string (treated as a text
    fragment); the audit event records a structural location pointing
    at the image URL, never the URL itself.
    """
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            EntropyDetector(
                name="high_entropy",
                min_entropy=4.0,
                min_length=24,
                action="audit",
            ),
        ],
        audit_sink=sink,
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    # A signed-URL shape with a long high-entropy token tail.
    high_entropy = "9aF3qZ7kP2vB8nC4xS6tR1dE5wY0uM7jL9hG3bN8oI4cV2pK"
    url = f"https://bucket.example.com/img.png?token={high_entropy}"
    await wrapped(make_agent(), [_image_block_url(url)])

    # The pipeline fired on the URL — audit event location points at the
    # image.url field.
    matched = [e for e in captured if e.location.endswith(".image.url")]
    assert matched, f"no audit event for image.url; got locations={[e.location for e in captured]}"
    assert matched[0].name == "high_entropy"
    assert matched[0].direction == "outbound"

    # Audit-trail privacy guarantee: the secret token never appears in
    # any serialized event field.
    for event in captured:
        assert high_entropy not in event.model_dump_json()


async def test_image_media_type_metadata_is_scanned_by_regex_detector() -> None:
    """A regex detector matching the media_type string fires.

    This is a silly case in practice — media types don't carry secrets —
    but it pins the behavior that ``image.media_type`` is part of the
    scanned surface. A future refactor that accidentally skips it would
    be caught.
    """
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            # Match the literal "image/png" — silly, but exact.
            RegexDetector("media_type_marker", r"image/png", action="audit"),
        ],
        audit_sink=sink,
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    await wrapped(
        make_agent(),
        [_image_block_url("https://example.com/x", media_type="image/png")],
    )

    matched = [e for e in captured if e.location.endswith(".image.media_type")]
    assert matched, (
        f"no audit event for image.media_type; got locations={[e.location for e in captured]}"
    )
    assert matched[0].name == "media_type_marker"


async def test_file_block_file_id_is_scanned_by_regex_detector() -> None:
    """A file block whose file_id matches a regex detector fires.

    The audit event location points at the ``file_id`` field; the inner
    runner sees the file_id redacted (when the action is redact).
    """
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            RegexDetector("aws_key", r"\bAKIA[A-Z0-9]{16}\b", action="redact"),
        ],
        audit_sink=sink,
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    # An attacker — or careless caller — stuffs an AWS key into the
    # file_id slot. The boundary catches it.
    bad_file_id = "AKIAIOSFODNN7EXAMPLE"
    await wrapped(make_agent(), [_file_block(file_id=bad_file_id)])

    # Audit event points at file_id, never echoes the key.
    matched = [e for e in captured if e.location.endswith(".file_id")]
    assert matched, f"no audit event for file_id; got locations={[e.location for e in captured]}"
    assert matched[0].name == "aws_key"
    assert matched[0].action == "redact"
    for event in captured:
        assert bad_file_id not in event.model_dump_json()

    # And the inner runner saw the redacted file_id, not the raw key.
    seen_block = inner.calls[0][0].content[0]
    assert seen_block.file_id == "[REDACTED:aws_key]"


async def test_file_block_path_metadata_is_scanned() -> None:
    """The `path` metadata field on a file block is scanned and redacted."""
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[RegexDetector("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", action="redact")],
        audit_sink=sink,
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    # Pathological: an operator path that embeds an SSN-shaped run.
    await wrapped(make_agent(), [_file_block(path="/data/user-123-45-6789.csv")])

    matched = [e for e in captured if e.location.endswith(".path")]
    assert matched, f"no audit event for path; got locations={[e.location for e in captured]}"
    assert matched[0].name == "us_ssn"

    seen_block = inner.calls[0][0].content[0]
    assert seen_block.path is not None
    assert "[REDACTED:us_ssn]" in seen_block.path
    assert "123-45-6789" not in seen_block.path


async def test_image_base64_data_is_not_scanned() -> None:
    """Regression guard: base64 image payloads must not be scanned.

    OCR is out of scope for the boundary. If a future refactor
    accidentally runs detectors over base64-encoded image bytes, this
    test fails — the detector pipeline would have fired on the
    secret-shaped bytes inside the payload.
    """
    captured, sink = make_event_capture()
    boundary = PrivacyBoundary(
        detectors=[
            # An AWS-key-shaped substring that would match if the bytes
            # were scanned. The boundary must skip the base64 data.
            RegexDetector("aws_key", r"\bAKIA[A-Z0-9]{16}\b", action="redact"),
        ],
        audit_sink=sink,
    )
    inner = RecordingRunner()
    wrapped = boundary.wrap(inner)

    # Synthetic base64-looking blob that *embeds* an AWS-key-shaped run.
    # If the boundary scanned base64 payloads, this detector would fire.
    payload = "ZmFrZS1ieXRlcy0AKIAIOSFODNN7EXAMPLE-bW9yZS1ieXRlcw=="
    await wrapped(make_agent(), [_image_block_base64(payload)])

    # No detection events at all: the media_type is "image/png" (no match),
    # the URL path is skipped because source=="base64", and crucially the
    # base64 data must not be scanned. Any audit event here is a regression
    # — using the tight assertion catches scans emitted at a different
    # location path (e.g. `.image.data` or `.image.base64`) too.
    assert captured == [], (
        f"base64 image bytes / metadata were scanned; this is a regression — "
        f"OCR is out of scope for the boundary. Got events at locations: "
        f"{[e.location for e in captured]}"
    )
    # And the inner runner sees the original base64 payload untouched.
    seen_block = inner.calls[0][0].content[0]
    assert seen_block.image is not None
    assert seen_block.image.data == payload
    assert seen_block.image.source == "base64"


async def test_image_url_block_inbound_redaction() -> None:
    """Image-block metadata scanning applies inbound, mirroring outbound."""
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

    class LeakingImageRunner:
        async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
            # Assistant message replying with an image URL that embeds an SSN.
            from harness.prompts.messages import ContentBlock, ImageRef

            return Message(
                role="assistant",
                content=[
                    ContentBlock(
                        type="image",
                        image=ImageRef(
                            source="url",
                            media_type="image/png",
                            data="https://leaked.example.com/123-45-6789.png",
                        ),
                    )
                ],
            )

    wrapped = boundary.wrap(LeakingImageRunner())
    reply = await wrapped(make_agent(), [text("user", "fetch me an image")])

    matched = [e for e in captured if e.location.endswith(".image.url")]
    assert matched, "expected inbound redaction on image.url"
    assert matched[0].direction == "inbound"

    seen = reply.content[0].image
    assert seen is not None
    assert "[REDACTED:us_ssn]" in seen.data
    assert "123-45-6789" not in seen.data
