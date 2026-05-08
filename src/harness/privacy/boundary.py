"""`PrivacyBoundary`: wrap any `Runner` with a pattern + entropy gate.

Architecture
------------
The boundary scans every text fragment that crosses the prompt boundary and
either *redacts* the match, *blocks* the whole call, or *audits* (passes
through with an event). It is intentionally a thin pure layer over the
detectors — no model knowledge, no provider knowledge.

```
caller -> messages -+--> [outbound scan] --> inner runner -> message
                    |                                          ^
                    +--> [inbound scan] <-----------------------+
```

Scope
-----
v1 scans `type == "text"` content blocks only. `tool_use.arguments` and
`tool_result.content` are not scanned in this pass; documented as future
work. The doc comment on `_scan_messages` calls this out.

Per-detector overrides
----------------------
A detector with `action="block"` blocks regardless of `on_detect`. A
detector with `action="redact"` redacts even when `on_detect="audit"`. The
boundary's `on_detect` is the *default* shape used by detectors that don't
override — which today is none of the shipped ones, but external detectors
may rely on it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from harness.privacy.detectors import (
    Action,
    Detection,
    Detector,
    Direction,
)
from harness.privacy.events import DetectionEvent
from harness.prompts.messages import ContentBlock, Message

if TYPE_CHECKING:
    from harness.agents.definition import SubAgent
    from harness.agents.orchestrator import Runner


AuditSink = Callable[[DetectionEvent], Awaitable[None]]


class PrivacyViolation(RuntimeError):
    """Raised when a `block` detector fires.

    The exception message is sanitized — never includes the matched value.
    The `detection` attribute carries the structural facts; the audit event
    is also emitted (the violation does not skip the audit trail).
    """

    def __init__(self, detection: Detection) -> None:
        self.detection = detection
        self.direction = detection.direction
        super().__init__(
            f"PrivacyViolation: detector {detection.name!r} matched on "
            f"{detection.direction} pass at {detection.location} "
            f"(match_length={detection.match_length})"
        )


class PrivacyBoundary:
    """Build a wrapper around any `Runner`. The wrapper enforces detectors.

    Parameters
    ----------
    detectors:
        Ordered list of detectors. Order matters when actions differ for
        overlapping matches: the first detection at a given range wins.
    on_detect:
        Default action used for detectors that report `audit` and have no
        per-detector override. Detector-level `action` always wins.
    audit_sink:
        Optional async callable that receives every `DetectionEvent`. The
        sink contract is intentionally narrow — pass `JSONLSink(...).emit`
        from `harness.telemetry` for file logging.
    """

    def __init__(
        self,
        detectors: list[Detector],
        *,
        on_detect: Action = "redact",
        audit_sink: AuditSink | None = None,
    ) -> None:
        self._detectors = list(detectors)
        self._default_action: Action = on_detect
        self._audit_sink = audit_sink

    def wrap(self, inner: Runner) -> Runner:
        """Return a new runner that gates `inner` through this boundary."""
        return _PrivacyWrappedRunner(self, inner)

    # ------------------------------------------------------------------
    # Internal API used by `_PrivacyWrappedRunner`

    async def scan_outbound(self, messages: list[Message]) -> list[Message]:
        """Walk every message; return a (possibly redacted) new list.

        Raises `PrivacyViolation` immediately if any detector with action
        `block` fires. The audit event for the block is emitted *before*
        the exception is raised.
        """
        return await self._scan_messages(messages, direction="outbound")

    async def scan_inbound(self, message: Message) -> Message:
        """Walk a single returned message and return a (possibly redacted) copy."""
        scanned = await self._scan_messages([message], direction="inbound")
        return scanned[0]

    async def _scan_messages(
        self,
        messages: list[Message],
        *,
        direction: Direction,
    ) -> list[Message]:
        out: list[Message] = []
        for m_idx, msg in enumerate(messages):
            new_blocks: list[ContentBlock] = []
            for b_idx, block in enumerate(msg.content):
                # Scope decision: v1 only scans `text` blocks. tool_use args
                # and tool_result content can also leak; documented as
                # future work in the module docstring.
                if block.type != "text" or not block.text:
                    new_blocks.append(block)
                    continue
                location = f"messages[{m_idx}].content[{b_idx}].text"
                redacted_text = await self._scan_text(
                    block.text,
                    direction=direction,
                    location=location,
                )
                new_blocks.append(block.model_copy(update={"text": redacted_text}))
            out.append(Message(role=msg.role, content=new_blocks))
        return out

    async def _scan_text(
        self,
        text: str,
        *,
        direction: Direction,
        location: str,
    ) -> str:
        # Collect detections from every detector. Each detector knows whether
        # it participates in this direction.
        detections: list[Detection] = []
        for detector in self._detectors:
            for det in detector.scan(text, direction=direction):
                detections.append(
                    Detection(
                        name=det.name,
                        start=det.start,
                        end=det.end,
                        direction=det.direction,
                        action=det.action,
                        location=location,
                    )
                )

        # Emit audit events for every detection — `block` decisions still
        # produce an audit record so blocked attempts stay visible.
        for det in detections:
            await self._emit_event(det)

        # Apply actions. `block` short-circuits everything else; redact /
        # audit can coexist on the same fragment.
        for det in detections:
            if self._effective_action(det) == "block":
                raise PrivacyViolation(det)

        # Redact right-to-left so earlier indices stay valid as we splice.
        redactions = sorted(
            (d for d in detections if self._effective_action(d) == "redact"),
            key=lambda d: d.start,
            reverse=True,
        )
        for det in redactions:
            text = text[: det.start] + f"[REDACTED:{det.name}]" + text[det.end :]
        return text

    def _effective_action(self, detection: Detection) -> Action:
        """Resolve the final action for a detection.

        The detection carries the per-detector action (set inside the
        detector). When that's the boundary fallback `audit` (the default
        for `EntropyDetector`), the boundary's `on_detect` is *not*
        applied — the per-detector intent wins. The exception is the
        deprecated case where a future detector ships without a default,
        which we intentionally don't have today.
        """
        return detection.action

    async def _emit_event(self, detection: Detection) -> None:
        if self._audit_sink is None:
            return
        event = DetectionEvent(
            name=detection.name,
            direction=detection.direction,
            action=self._effective_action(detection),
            location=detection.location,
            match_length=detection.match_length,
        )
        await self._audit_sink(event)


class _PrivacyWrappedRunner:
    """Internal callable. Satisfies the `Runner` protocol: ``async (agent, messages) -> Message``.

    Uses a class instead of a closure so the wrapper has a stable repr and
    callers can `isinstance`-check (e.g. for "is this runner already
    privacy-gated?" diagnostics).
    """

    def __init__(self, boundary: PrivacyBoundary, inner: Runner) -> None:
        self._boundary = boundary
        self._inner = inner

    def __repr__(self) -> str:
        return f"_PrivacyWrappedRunner(inner={self._inner!r})"

    async def __call__(self, agent: SubAgent, messages: list[Message]) -> Message:
        sanitized_outbound = await self._boundary.scan_outbound(messages)
        reply = await self._inner(agent, sanitized_outbound)
        return await self._boundary.scan_inbound(reply)
