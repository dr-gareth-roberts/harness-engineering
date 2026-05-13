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
The boundary scans the following content shapes:

- `text` blocks — the `text` field.
- `tool_use.arguments` — walked recursively to find string leaves.
- `tool_result.content` — walked recursively when its value is a dict /
  list; string-typed content is scanned directly.
- `image` blocks — image *metadata* only: the URL string (when
  ``image.source == "url"``) and the ``media_type`` field. The base64
  payload of an inline image is **not** scanned (that would require
  OCR and is out of scope; see "Multimodal limits" below).
- `file` blocks — file *metadata* only: the ``file_id`` and ``path``
  fields. The file body fetched by the model later (via the vendor's
  Files API) crosses the boundary outside this layer and is **not**
  scanned here.

Recursion into structured tool_use / tool_result values is capped at
`_MAX_RECURSION_DEPTH` levels deep. Nodes deeper than the cap are
stringified via `json.dumps(default=str, sort_keys=True)` and scanned as a
single flat blob — detection still works, but the audit event's location
is annotated `[depth-cap]` rather than carrying a nested path.

Multimodal limits
-----------------
The boundary is a string-level detector pipeline. It does **not** decode
image bytes or read file bodies. Callers that need image-text scanning
should run an OCR pre-pass over the image *before* the message reaches the
boundary (e.g. construct the image block with
:func:`harness.prompts.attach_image` *and* materialize OCR-extracted text
into a sibling `text` block; the boundary then scans the OCR output as
normal text). Callers that need file-body scanning should fetch and
inline the file's text into a `text` block before the boundary, rather
than rely on the runner's later Files API resolution.

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
    messages[i].content[j].image.url                             — image URL (url-sourced)
    messages[i].content[j].image.media_type                      — image mime type
    messages[i].content[j].file_id                               — file block id
    messages[i].content[j].path                                  — file/image path metadata

Top-level tool-call args dicts use plain `.<key>`; nested keys chain with
`.`; list elements use `[n]`. The cap suffix appears only when recursion
exceeded `_MAX_RECURSION_DEPTH`.

Dict-key sanitization
---------------------
Dict keys themselves can carry sensitive strings (e.g. an API-key-shaped
string used as a key by a careless caller). Before a key is interpolated
into the audit-event `location`, it is run through the configured
detectors and any matching range is replaced with `<redacted>`. The
inner runner still sees the original key in the arguments dict — only
the audit-event location is sanitized.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from harness.privacy.detectors import (
    Detection,
    Detector,
    Direction,
)
from harness.privacy.events import DetectionEvent
from harness.prompts.messages import ContentBlock, ImageRef, Message
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


def _merge_ranges(detections: list[Detection]) -> list[Detection]:
    """Greedy-merge overlapping ranges; return one Detection per merged span.

    Sort by `(start, -end)` so when two ranges share a start the wider one
    sorts first. Then walk left-to-right, merging any range whose `start`
    is `<= prev.end`. The merged Detection keeps the earlier-and-wider
    detection's `name`, `direction`, `action`, and `location` — those
    fields are not used past splice-time for the wider boundary semantics,
    and the audit events for each individual detection have already been
    emitted upstream.
    """
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: (d.start, -d.end))
    merged: list[Detection] = [ordered[0]]
    for det in ordered[1:]:
        prev = merged[-1]
        if det.start <= prev.end:
            if det.end > prev.end:
                merged[-1] = Detection(
                    name=prev.name,
                    start=prev.start,
                    end=det.end,
                    direction=prev.direction,
                    action=prev.action,
                    location=prev.location,
                )
            # else: prev fully covers det — drop det.
        else:
            merged.append(det)
    return merged


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
        Ordered list of detectors. Detector-level `action` is honored
        verbatim; when two detectors fire on overlapping ranges, the
        ranges are merged before redaction so the splice produces clean
        output (see `_merge_ranges`).
    audit_sink:
        Optional async callable that receives every `DetectionEvent`. The
        sink contract is intentionally narrow — pass `JSONLSink(...).emit`
        from `harness.telemetry` for file logging.
    """

    def __init__(
        self,
        detectors: list[Detector],
        *,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self._detectors = list(detectors)
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

                if block.type == "image" and block.image is not None:
                    new_blocks.append(
                        await self._scan_image_block(block, base=base, direction=direction)
                    )
                    continue

                if block.type == "file":
                    new_blocks.append(
                        await self._scan_file_block(block, base=base, direction=direction)
                    )
                    continue

                # Unknown block type: pass through unchanged. (All currently
                # defined types are handled above; this guard exists so that
                # adding a new `BlockType` literal doesn't silently bypass
                # the boundary if the new branch is missed in review.)
                new_blocks.append(block)
            out.append(Message(role=msg.role, content=new_blocks))
        return out

    async def _scan_image_block(
        self,
        block: ContentBlock,
        *,
        base: str,
        direction: Direction,
    ) -> ContentBlock:
        """Scan image *metadata* and return a (possibly redacted) copy.

        Scans the URL string when ``image.source == "url"`` and always
        scans ``image.media_type``. The base64 payload of an inline image
        is **not** decoded or scanned here — that's an OCR problem and is
        out of scope. Callers needing image-text scanning should run an
        OCR pre-pass before the message reaches the boundary.

        ``block.path`` (if set) is scanned as a sibling metadata field;
        its location is ``{base}.path`` to keep the grammar uniform with
        file blocks.
        """
        # `block.image is not None` is the caller's precondition.
        assert block.image is not None  # noqa: S101 — narrow type for mypy

        new_media_type = await self._scan_text(
            block.image.media_type,
            direction=direction,
            location=f"{base}.image.media_type",
        )

        # URL-sourced images: scan the URL string. Base64-sourced images
        # carry their bytes in `data`; we leave that untouched (OCR is
        # out of scope per the module docstring).
        if block.image.source == "url":
            new_data = await self._scan_text(
                block.image.data,
                direction=direction,
                location=f"{base}.image.url",
            )
        else:
            new_data = block.image.data

        new_image = ImageRef(
            source=block.image.source,
            media_type=new_media_type,
            data=new_data,
        )
        update: dict[str, Any] = {"image": new_image}

        if block.path is not None:
            update["path"] = await self._scan_text(
                block.path,
                direction=direction,
                location=f"{base}.path",
            )

        return block.model_copy(update=update)

    async def _scan_file_block(
        self,
        block: ContentBlock,
        *,
        base: str,
        direction: Direction,
    ) -> ContentBlock:
        """Scan file *metadata* and return a (possibly redacted) copy.

        Scans ``block.file_id`` and ``block.path``. The file body that
        the model fetches later (via the vendor's Files API) is **not**
        read here — that crosses the boundary outside this layer.
        Callers needing file-body scanning should inline the file's
        text into a `text` block before the message reaches the boundary.
        """
        update: dict[str, Any] = {}

        if block.file_id is not None:
            update["file_id"] = await self._scan_text(
                block.file_id,
                direction=direction,
                location=f"{base}.file_id",
            )

        if block.path is not None:
            update["path"] = await self._scan_text(
                block.path,
                direction=direction,
                location=f"{base}.path",
            )

        if not update:
            return block
        return block.model_copy(update=update)

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
                # Sanitize the key before composing it into the location
                # path — a key shaped like a secret would otherwise leak
                # into `DetectionEvent.location` / `PrivacyViolation.__str__`.
                # `k` itself is preserved as-is in `new_dict` so the inner
                # runner sees the original arguments shape.
                safe_key = self._sanitize_key(str(k), direction=direction)
                sub_location = f"{location}.{safe_key}"
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
            if det.action == "block":
                raise PrivacyViolation(det)

        # Merge overlapping redaction ranges before splicing. Without this,
        # two detectors firing on overlapping spans corrupt the output
        # (the right-to-left splice would double-redact). The merged range
        # keeps the earlier-and-wider detection's name for `[REDACTED:<name>]`.
        redactions = _merge_ranges([d for d in detections if d.action == "redact"])

        # Redact right-to-left so earlier indices stay valid as we splice.
        for det in sorted(redactions, key=lambda d: d.start, reverse=True):
            text = text[: det.start] + f"[REDACTED:{det.name}]" + text[det.end :]
        return text

    def _sanitize_key(self, key: str, *, direction: Direction) -> str:
        """Run a dict key through the configured detectors; redact matches.

        Used before composing key strings into the audit-event `location`.
        The inner runner still sees the original key in the arguments dict
        — only the structural path that lands in `DetectionEvent.location`
        and `PrivacyViolation.__str__` is sanitized.

        Block-action detectors do **not** raise here: a dict key that
        merely looks like a secret is not the same as a value being sent
        outbound, and raising would surface a different surprise (a block
        on a key shape with no audit trail for the value beside it). The
        key is treated as audit-context only.
        """
        if not key:
            return key
        detections: list[Detection] = []
        for detector in self._detectors:
            for det in detector.scan(key, direction=direction):
                detections.append(det)
        if not detections:
            return key
        merged = _merge_ranges(detections)
        # Right-to-left splice keeps earlier indices valid.
        result = key
        for det in sorted(merged, key=lambda d: d.start, reverse=True):
            result = result[: det.start] + "<redacted>" + result[det.end :]
        return result

    async def _emit_event(self, detection: Detection) -> None:
        if self._audit_sink is None:
            return
        event = DetectionEvent(
            name=detection.name,
            direction=detection.direction,
            action=detection.action,
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
