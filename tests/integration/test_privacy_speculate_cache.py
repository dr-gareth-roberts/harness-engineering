"""PrivacyBoundary wrapping a runner-protocol-conformant fake that uses speculate + cache (M4.5).

The contract this test pins:

1. **PrivacyBoundary redacts outbound text before the runner sees it.**
   A secret embedded in an assistant `tool_use` argument crossing the
   prompt boundary becomes ``[REDACTED:<name>]`` by the time the inner
   runner is invoked.

2. **Downstream consumers see only redacted state.** Inside the wrapped
   runner we:
   - drive a `Speculator` (its `begin()` reads the same history; its
     `LastCallPredictor` picks an idempotent tool to pre-dispatch),
   - fingerprint the request via `PrefixWatcher`,
   and assert that neither sees the raw secret. Speculator and watcher
   are *runner-internal* — they consume the post-redaction message
   stream and do not re-flow back through `PrivacyBoundary`. This test
   pins the implication of the architecture, not a separate
   re-scanning guarantee. See note at the end of this module about
   why "the speculator's dispatch goes through the boundary" is true
   *only via* outbound-redaction-of-history, not via a re-scan path.

3. **PrefixWatcher sees clean prompts.** The fingerprint store records
   no raw secret in its `full_prompt` capture.

The audit-sink wired into PrivacyBoundary collects `DetectionEvent`s so
the test can pin the structural facts of the detection (name, action,
location) without ever seeing the matched value.

This suite uses fake runner callables to exercise the
orchestrator/runner protocol surface. End-to-end coverage with
concrete vendor runners (`AnthropicRunner` / `OpenAICompatRunner`
against faked SDK boundaries) is tracked separately.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from harness.agents.definition import SubAgent
from harness.cache import InMemoryFingerprintStore, PrefixWatcher
from harness.hooks import HookRunner
from harness.privacy import (
    PrivacyBoundary,
    RegexDetector,
)
from harness.privacy.events import DetectionEvent
from harness.prompts import (
    Message,
    assistant_tool_use,
    text,
)
from harness.speculate import LastCallPredictor, Speculator
from harness.tools import Dispatcher, Tool, ToolCall

# A clearly-shaped fake secret. Format mimics the `SECRET_PACK`
# AWS access-key shape so we don't accidentally collide with real
# secret detectors in other tests. The detector below targets only
# this specific shape so the assertions can pin the redaction count.
_FAKE_SECRET = "AKIA" + "Z" * 16  # 20-char total, real AWS key shape
_DETECTOR_NAME = "test_aws_key"


class _SearchArgs(BaseModel):
    query: str


def _detector() -> RegexDetector:
    """Single redactor for the fake AWS-key shape; outbound only."""
    return RegexDetector(
        _DETECTOR_NAME,
        r"\bAKIA[A-Z0-9]{16}\b",
        direction="outbound",
        action="redact",
    )


def _build_dispatcher(dispatch_log: list[tuple[str, dict[str, Any]]]) -> Dispatcher:
    async def search(args: _SearchArgs) -> str:
        dispatch_log.append(("search", {"query": args.query}))
        return f"hits for {args.query}"

    return Dispatcher(
        [
            Tool(
                name="search",
                description="idempotent search",
                input_model=_SearchArgs,
                handler=search,
                idempotent=True,
            )
        ]
    )


def _serialize_history_for_fingerprint(messages: list[Message]) -> dict[str, Any]:
    """Mimic what a vendor runner builds for the fingerprint call —
    the runner usually translates the harness messages into the
    vendor-shaped dict; for this test we use the harness messages
    directly (the watcher hashes whatever dict it gets).
    """
    return {
        "model": "test-model",
        "messages": [m.model_dump() for m in messages],
        "tools": [
            {"name": "search", "input_schema": _SearchArgs.model_json_schema()},
        ],
    }


async def test_privacy_boundary_wraps_runner_using_speculator_and_watcher(
    make_agent: Callable[..., SubAgent],
) -> None:
    """End-to-end: a secret hidden in an assistant `tool_use.arguments`
    is redacted before the wrapped runner runs. Inside the runner, a
    speculator (LastCallPredictor) and a PrefixWatcher both consume the
    redacted message stream — neither sees the raw secret.
    """
    audit_events: list[DetectionEvent] = []

    async def audit_sink(event: DetectionEvent) -> None:
        audit_events.append(event)

    boundary = PrivacyBoundary([_detector()], audit_sink=audit_sink)

    dispatch_log: list[tuple[str, dict[str, Any]]] = []
    dispatcher = _build_dispatcher(dispatch_log)
    hooks = HookRunner()

    speculator = Speculator(
        LastCallPredictor(history_window=1),
        max_speculations=1,
        only_idempotent=True,
    )
    watcher_store = InMemoryFingerprintStore()
    watcher = PrefixWatcher(watcher_store, full_capture="always")

    # State captured by the fake runner, asserted by the test.
    runner_observations: dict[str, Any] = {}

    async def fake_runner(agent: SubAgent, messages: list[Message]) -> Message:
        """Simulate what a real vendor runner does internally:

        - Reads the history (already redacted by the outer boundary).
        - Asks the speculator to predict + pre-execute the next call.
        - Calls the prefix watcher with the request body.
        - Drives one tool-use round-trip with the model's call.
        - Resolves the speculation against the model's call.
        """
        runner_observations["history"] = messages

        # 1) Speculator begin — predictor reads `messages`. The
        #    predictor's LastCallPredictor walks assistant tool_use
        #    blocks; the redacted args land here.
        await speculator.begin(
            history=messages,
            agent=agent,
            dispatcher=dispatcher,
            hooks=hooks,
        )

        # 2) Fingerprint via the watcher. The serialized request is
        #    the post-redaction message stream.
        request_body = _serialize_history_for_fingerprint(messages)
        await watcher.fingerprint(request_body)
        runner_observations["fingerprint_body"] = request_body

        # 3) Model's actual call — same name+args as the previous turn
        #    so the speculator's prediction matches. Real Anthropic
        #    runner does this in response to a `tool_use` block from
        #    the model; here we synthesize it directly.
        # Use the SAME arguments the predictor would have predicted —
        # which is the redacted form from the history.
        prev_tool_use = next(
            block.tool_use
            for msg in reversed(messages)
            if msg.role == "assistant"
            for block in msg.content
            if block.type == "tool_use" and block.tool_use is not None
        )
        model_call = ToolCall(
            id="model-id",
            name=prev_tool_use.name,
            arguments=dict(prev_tool_use.arguments),
        )

        # Observe + resolve the call against the speculator. If the
        # speculation hits, the speculator returns the result it
        # pre-dispatched; otherwise the runner would fall through to
        # dispatcher.dispatch(model_call).
        await speculator.observe(model_call)
        resolved = await speculator.try_resolve(model_call)
        if resolved is None:
            resolved = await dispatcher.dispatch(model_call)
        await speculator.end()

        runner_observations["resolved"] = resolved
        runner_observations["model_call_args"] = model_call.arguments

        # Return a clean assistant reply (no secrets — boundary will
        # scan inbound too).
        return text("assistant", "all done")

    wrapped = boundary.wrap(fake_runner)

    # Build the input history. The assistant `tool_use.arguments`
    # contains the secret inside the `query` field; PrivacyBoundary
    # must redact it on the way through.
    previous_call = ToolCall(
        id="prev",
        name="search",
        arguments={"query": f"please look up {_FAKE_SECRET}"},
    )
    history = [
        text("user", "find info"),
        Message(
            role="assistant",
            content=[
                *text("assistant", "I'll search.").content,
                *assistant_tool_use(previous_call).content,
            ],
        ),
        text("user", "anything else?"),
    ]

    reply = await wrapped(make_agent(allowed_tools=["search"]), history)

    # 1) The runner's view of history: the secret was redacted.
    seen_history = runner_observations["history"]
    redacted_assistant = seen_history[1]
    redacted_args = next(
        b.tool_use.arguments
        for b in redacted_assistant.content
        if b.type == "tool_use" and b.tool_use is not None
    )
    assert _FAKE_SECRET not in redacted_args["query"], (
        "PrivacyBoundary failed to redact the secret in tool_use.arguments: "
        f"runner saw {redacted_args!r}"
    )
    assert f"[REDACTED:{_DETECTOR_NAME}]" in redacted_args["query"]

    # 2) The fingerprint body: serialized request must not contain the
    # raw secret either. The watcher hashed only redacted content.
    fingerprint_body = runner_observations["fingerprint_body"]
    serialized = json.dumps(fingerprint_body)
    assert _FAKE_SECRET not in serialized, (
        "PrefixWatcher hashed a request body containing the raw secret"
    )

    # 3) The fingerprint store should have one record per breakpoint;
    # since we used a vanilla OpenAI-shaped request (no `cache_control`
    # markers), the watcher records one segment. `full_capture="always"`
    # means the prompt bytes are stored — pin that no record's
    # `full_prompt` contains the raw secret.
    from datetime import UTC, datetime, timedelta

    since = datetime.now(UTC) - timedelta(seconds=60)
    records = [r async for r in watcher_store.iter_recent(since=since)]
    assert records, "PrefixWatcher did not append any record"
    for record in records:
        if record.full_prompt is not None:
            assert _FAKE_SECRET not in record.full_prompt, (
                f"FingerprintRecord.full_prompt leaked the secret: "
                f"breakpoint_index={record.breakpoint_index}, "
                f"hash={record.hash}"
            )

    # 4) Speculator behaviour: it saw the redacted previous call and
    # speculated on `search` with the redacted query. The dispatcher
    # log captures every dispatch (speculative + model); none of them
    # carry the raw secret.
    #
    # Pin non-empty so a silent regression where the speculator no longer
    # dispatches (predictor stops firing, `_dispatch_via_hooks` short-circuits)
    # fails this test instead of passing vacuously.
    assert dispatch_log, (
        "expected at least one dispatch via the speculator's pre-execute path "
        "(LastCallPredictor with history_window=1 on a tool_use history should "
        "have predicted + dispatched 'search'); got empty dispatch_log"
    )
    for _, args in dispatch_log:
        assert _FAKE_SECRET not in str(args), f"Dispatcher handler received raw secret: {args!r}"

    # The runner's resolve step must have surfaced a non-None ToolResult —
    # either a hit from the speculator or a fallback dispatch. Either way
    # the path that consumes the redacted message stream produced output.
    assert runner_observations.get("resolved") is not None, (
        "fake runner never resolved a ToolResult (speculator try_resolve + "
        "fallback dispatch both returned nothing)"
    )

    # 5) Audit-sink events: at least one detection fired, with the
    # expected detector name and outbound direction.
    assert audit_events, "PrivacyBoundary emitted no audit event for the matched secret"
    detector_events = [e for e in audit_events if e.name == _DETECTOR_NAME]
    assert detector_events, (
        f"no audit event for detector {_DETECTOR_NAME!r}; "
        f"saw names={[e.name for e in audit_events]}"
    )
    assert all(e.direction == "outbound" for e in detector_events)
    assert all(e.action == "redact" for e in detector_events)
    # The location must point inside the tool_use.arguments path —
    # PrivacyBoundary's grammar uses `tool_use.arguments.<key>`.
    detected_locations = [e.location for e in detector_events]
    assert any("tool_use.arguments.query" in loc for loc in detected_locations), (
        f"no detection located inside tool_use.arguments.query; locations={detected_locations}"
    )

    # 6) The reply is clean (no secret to inbound-scan, but the path
    # is still exercised).
    assert reply.role == "assistant"
    assert _FAKE_SECRET not in (reply.content[0].text or "")


async def test_privacy_boundary_audit_event_never_carries_matched_value(
    make_agent: Callable[..., SubAgent],
) -> None:
    """The load-bearing audit contract: `DetectionEvent` never exposes
    the matched secret. Pin that the event surface is structural
    (name / location / direction / action / match_length) and the
    matched string is nowhere on the event.
    """
    audit_events: list[DetectionEvent] = []

    async def audit_sink(event: DetectionEvent) -> None:
        audit_events.append(event)

    boundary = PrivacyBoundary([_detector()], audit_sink=audit_sink)

    async def inner_runner(agent: SubAgent, messages: list[Message]) -> Message:
        return text("assistant", "ok")

    wrapped = boundary.wrap(inner_runner)
    await wrapped(
        make_agent(allowed_tools=["search"]),
        [text("user", f"my key is {_FAKE_SECRET}")],
    )

    assert audit_events, "no audit event was emitted for the in-text secret"
    for event in audit_events:
        # Convert to JSON via the same path a real sink would use —
        # if any field leaked the secret it would show up here.
        as_json = event.model_dump_json()
        assert _FAKE_SECRET not in as_json, (
            f"DetectionEvent serialization leaked the matched value: {as_json!r}"
        )
        # And the structural fields are populated correctly.
        assert event.match_length == len(_FAKE_SECRET)


# ---------------------------------------------------------------------------
# Architectural note (intentionally not a test):
#
# Test scope clarification — "the speculator's speculative dispatch goes
# through the privacy boundary" can only mean "the speculator predicts on
# args the boundary has already redacted in the input history." The
# speculator's `_dispatch_via_hooks` calls `dispatcher.dispatch(call)`
# *directly*; that dispatch does NOT re-flow through
# `PrivacyBoundary.scan_outbound`. The boundary's contract is "wrap a
# Runner, scan messages once before invocation"; speculative dispatches
# are runner-internal and inherit the redacted history but are not
# independently re-scanned. The asymmetry is documented in
# `harness.privacy.boundary` and intentional — re-scanning a structured
# argument that was already scanned during outbound would double-emit
# audit events for the same span. If a future requirement needs
# argument-level re-scanning at the Dispatcher boundary, that's a new
# layer, not a tweak to PrivacyBoundary.
