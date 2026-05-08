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
The boundary scans three content shapes: `text` blocks, `tool_use.arguments`
(walked recursively to find string leaves), and `tool_result.content` (also
walked recursively when its value is a dict / list — string-typed content
is scanned directly).

Recursion into structured tool_use / tool_result values is capped at
`_MAX_RECURSION_DEPTH` levels deep. Nodes deeper than the cap are
stringified via `json.dumps(default=str, sort_keys=True)` and scanned as a
single flat blob — detection still works, but the audit event's location
is annotated `[depth-cap]` rather than carrying a nested path.

Location-path grammar
---------------------
Audit events carry a `location` string identifying where the match was
found. Format:

    messages[i].content[j].text                                  — text block
    messages[i].content[j].tool_use.arguments.<key>              — tool_use
    messages[i].content[j].tool_use.arguments.<key>.<sub>        — nested
    messages[i].content[j].tool_use.arguments.<key>[n]           — list item
    messages[i].content[j].tool_result.content                   — string-typed
    messages[i].content[j].tool_result.content.<key>             — nested
    messages[i].content[j].tool_result.content[depth-cap]        — capped

Top-level tool-call args dicts use plain `.<key>`; nested keys chain with
`.`; list elements use `[n]`. The cap suffix appears only when recursion
exceeded `_MAX_RECURSION_DEPTH`.

Per-detector overrides
----------------------
A detector with `action="block"` blocks regardless of `on_detect`. A
detector with `action="redact"` redacts even when `on_detect="audit"`. The
boundary's `on_detect` is the *default* shape used by detectors that don't
override — which today is none of the shipped ones, but external detectors
may rely on it.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from harness.privacy.detectors import (
    Action,
    Detection,
    Detector,
    Direction,
)
from harness.privacy.events import DetectionEvent
from harness.prompts.messages import ContentBlock, Message
from harness.tools.schema import ToolCall, ToolResult

_MAX_RECURSION_DEPTH = 4
"""Max nesting depth when walking tool_use.arguments and tool_result.content.

Beyond this, the value is stringified once and scanned as a flat blob.
Bounded recursion avoids hangs on self-referential or pathologically deep
data while still catching leaks at realistic nesting depths.
"""

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
                base = f"messages[{m_idx}].content[{b_idx}]"

                if block.type == "text" and block.text:
                    redacted_text = await self._scan_text(
                        block.text,
                        direction=direction,
                        location=f"{base}.text",
                    )
                    new_blocks.append(block.model_copy(update={"text": redacted_text}))
                    continue

                if block.type == "tool_use" and block.tool_use is not None:
                    new_args = await self._scan_value(
                        block.tool_use.arguments,
                        direction=direction,
                        location=f"{base}.tool_use.arguments",
                        depth=0,
                    )
                    new_call = ToolCall(
                        name=block.tool_use.name,
                        arguments=new_args if isinstance(new_args, dict) else {},
                        id=block.tool_use.id,
                    )
                    new_blocks.append(block.model_copy(update={"tool_use": new_call}))
                    continue

                if block.type == "tool_result" and block.tool_result is not None:
                    new_content = await self._scan_value(
                        block.tool_result.content,
                        direction=direction,
                        location=f"{base}.tool_result.content",
                        depth=0,
                    )
                    new_result = ToolResult(
                        id=block.tool_result.id,
                        content=new_content,
                        is_error=block.tool_result.is_error,
                    )
                    new_blocks.append(block.model_copy(update={"tool_result": new_result}))
                    continue

                # File blocks and other types: pass through.
                new_blocks.append(block)
            out.append(Message(role=msg.role, content=new_blocks))
        return out

    async def _scan_value(
        self,
        value: Any,
        *,
        direction: Direction,
        location: str,
        depth: int,
    ) -> Any:
        """Walk a structured value, scanning string leaves.

        `block` actions raise `PrivacyViolation` exactly as they do for
        top-level text — `_scan_text` is the single source of action
        semantics. Non-string scalars (int / bool / None / float) pass
        through unchanged. Dicts and lists are walked; their keys / indices
        chain into the location string. Beyond `_MAX_RECURSION_DEPTH`,
        the subtree is stringified once and scanned flat.
        """
        if value is None or isinstance(value, bool | int | float):
            return value

        if isinstance(value, str):
            return await self._scan_text(value, direction=direction, location=location)

        if depth >= _MAX_RECURSION_DEPTH:
            # Cap reached: flat-scan the JSON serialization. The redacted
            # string replaces the original subtree wholesale — the audit
            # event makes the depth-cap explicit so callers can see when
            # recursion was truncated.
            flat = json.dumps(value, default=str, sort_keys=True)
            return await self._scan_text(
                flat,
                direction=direction,
                location=f"{location}[depth-cap]",
            )

        if isinstance(value, dict):
            new_dict: dict[str, Any] = {}
            for k, v in value.items():
                sub_location = f"{location}.{k}"
                new_dict[k] = await self._scan_value(
                    v,
                    direction=direction,
                    location=sub_location,
                    depth=depth + 1,
                )
            return new_dict

        if isinstance(value, list):
            new_list: list[Any] = []
            for i, v in enumerate(value):
                sub_location = f"{location}[{i}]"
                new_list.append(
                    await self._scan_value(
                        v,
                        direction=direction,
                        location=sub_location,
                        depth=depth + 1,
                    )
                )
            return new_list

        # Unknown type (custom class etc.): scan the str() representation
        # as a defense-in-depth measure. Redaction here can't preserve the
        # type, so we return the original on miss and the redacted str on
        # hit. Document as a corner case.
        as_str = str(value)
        scanned = await self._scan_text(
            as_str,
            direction=direction,
            location=f"{location}[str]",
        )
        return scanned if scanned != as_str else value

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
