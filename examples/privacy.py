"""Privacy boundary: redact PII before it crosses the runner boundary.

Run with: `uv run python examples/privacy.py`

`harness.privacy.PrivacyBoundary` wraps any runner and treats the prompt
boundary as a compliance surface. Detectors scan every text fragment
crossing in either direction. Three actions:

* `redact` — replace the match with `[REDACTED:<name>]`.
* `block`  — raise `PrivacyViolation` before the wrapped runner is called.
* `audit`  — pass through but log a `DetectionEvent` to the audit sink.

This example demonstrates two flows:

1. **PII redaction**: a user message containing a SSN is sanitized before
   the inner `EchoRunner` ever sees it. The audit event records that a
   detection happened — but never the matched value itself.
2. **Secret blocking**: an outbound message containing an AWS-key-shaped
   string raises `PrivacyViolation`. The wrapped runner is never invoked.
"""

from __future__ import annotations

import asyncio

from harness.agents import SubAgent
from harness.privacy import (
    PII_PACK,
    DetectionEvent,
    PrivacyBoundary,
    PrivacyViolation,
    RegexDetector,
)
from harness.prompts import text
from harness.runner import EchoRunner


def _agent() -> SubAgent:
    return SubAgent(
        name="privacy-demo",
        system_prompt="",
        model="demo-model",
        allowed_tools=[],
    )


async def main() -> int:
    transcript: list[str] = []

    # The audit sink is just a list-appending callable. In production
    # you'd typically wire `JSONLSink(...).emit` from `harness.telemetry`.
    captured: list[DetectionEvent] = []

    async def audit_sink(event: DetectionEvent) -> None:
        captured.append(event)

    # ----- Flow 1: PII redaction (PII_PACK includes us_ssn) ------------
    transcript.append("--- flow 1: SSN redacted before reaching the runner ---")
    inner = EchoRunner()
    boundary_redact = PrivacyBoundary(detectors=list(PII_PACK), audit_sink=audit_sink)
    wrapped = boundary_redact.wrap(inner)

    user_message = text("user", "My SSN is 123-45-6789. Please confirm.")
    reply = await wrapped(_agent(), [user_message])

    # The reply is what the inner runner saw — and the runner only saw
    # `[REDACTED:us_ssn]`, not the original digits.
    reply_text = next((b.text for b in reply.content if b.type == "text"), "")
    transcript.append(f"  inner runner echoed: {reply_text!r}")
    transcript.append(f"  audit events captured: {len(captured)}")
    for event in captured:
        transcript.append(
            f"    detected name={event.name!r} "
            f"direction={event.direction} action={event.action} "
            f"location={event.location}"
        )
    # The privacy guarantee: no audit event carries the raw secret value.
    assert all("123-45-6789" not in str(e.model_dump()) for e in captured)
    transcript.append("  raw SSN never appears in any audit event ✓")

    # ----- Flow 2: AWS key blocked outright ----------------------------
    transcript.append("--- flow 2: AWS key blocks the outbound call ---")
    captured.clear()
    boundary_block = PrivacyBoundary(
        detectors=[
            RegexDetector(
                "aws_access_key",
                r"\bAKIA[A-Z0-9]{16}\b",
                direction="outbound",
                action="block",
            ),
        ],
        audit_sink=audit_sink,
    )
    wrapped = boundary_block.wrap(inner)

    bad = text("user", "Please use my key: AKIAABCDEFGHIJKLMNOP for the call.")
    blocked = False
    try:
        await wrapped(_agent(), [bad])
    except PrivacyViolation as exc:
        blocked = True
        transcript.append(
            f"  PrivacyViolation raised: {exc.detection.name!r} at {exc.detection.location}"
        )
    transcript.append(f"  inner runner invoked: {not blocked}")
    transcript.append(f"  audit events captured (block path): {len(captured)}")

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
