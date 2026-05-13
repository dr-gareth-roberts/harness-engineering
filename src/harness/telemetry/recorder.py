"""Telemetry recorder — fans events out to a sink and threads correlation IDs.

`Telemetry` wraps a single `Sink` and isolates failures (sink errors are
logged at WARNING and swallowed; a misbehaving sink can never crash an
orchestrator turn or a tool dispatch). Pair with `MultiSink` to fan out
to multiple backends.

**Correlation IDs** (Wave 11 #11): the recorder uses `contextvars` to
thread a `trace_id` (per session) and a hierarchical `span_id` /
`parent_span_id` (per turn / dispatch / speculation) through async work
without explicit threading. Two context-manager APIs:

```python
async with telemetry.session_scope():
    # Inside this block, `_current_trace_id` is set; emitted events
    # carry it. Nested span_scope() calls inherit it.
    async with telemetry.span_scope():
        await telemetry.emit(...)  # event.trace_id and span_id populated
```

Events emitted inside a scope inherit `trace_id` / `span_id` /
`parent_span_id` from the current `contextvars` state. Events emitted
outside any scope (no `session_scope` open) keep their default `None`
IDs — `OpenTelemetrySink` then falls back to flat-event behavior, the
JSONL / Memory sinks just record the IDs as null.

Scopes are designed for `async with` use so nested orchestrator turns
and concurrent tool dispatches each get their own `span_id` without
clobbering each other's context.
"""

from __future__ import annotations

import contextvars
import logging
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from harness.telemetry.events import TelemetryEvent
from harness.telemetry.sinks import NullSink, Sink

logger = logging.getLogger(__name__)


Redactor = Callable[[TelemetryEvent], TelemetryEvent]
"""Pure-data scrubber applied at the `Telemetry.emit` boundary.

A redactor takes a `TelemetryEvent` and returns a (possibly new) event of
the same type, applied *before* fan-out to any sink. The contract is
pure-data: callers should return a new instance (typically via
`event.model_copy(update={...})`) rather than mutating the input — the
recorder doesn't snapshot the event before handing it off, and a sink
that retains references (e.g. `MemorySink`) would otherwise observe the
mutation.

`Telemetry` does *not* catch exceptions raised by the redactor: a bug in
the scrubber is a configuration error, not a runtime curiosity, and
silently dropping events on a redactor crash would be worse than a loud
failure. Wrap your redactor's body in `try/except` if you want soft
failure modes.

This is a *telemetry-boundary* primitive — sinks are still not
audit-grade by default. For audit-grade redaction of model I/O across
the runner boundary, use `harness.privacy.PrivacyBoundary`; that's a
different boundary with stronger guarantees.
"""


_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "harness_trace_id", default=None
)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "harness_span_id", default=None
)
# parent_span_id snapshots the previous span_id at the moment a new
# span_scope opens. Events emitted inside that scope record both the
# new span_id and this parent. Reset together when the scope exits.
_current_parent_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "harness_parent_span_id", default=None
)


class Telemetry:
    """Central recorder. Wraps a single `Sink` and isolates failures.

    The recorder never raises on sink failure — sink errors are logged
    at WARNING and swallowed so a misbehaving sink can never crash an
    orchestrator turn or a tool dispatch. Pair with `MultiSink` to fan
    out to multiple backends.

    Pass `redactor=` to scrub events at the telemetry boundary before
    they reach any sink. The redactor runs *after* correlation IDs are
    threaded in (so the redactor sees the populated event) and *before*
    `self._sink.emit(...)` (so every sink in a `MultiSink` sees the same
    redacted view). See the `Redactor` type alias for the contract;
    sinks are not audit-grade by default — for runner-boundary
    redaction, use `harness.privacy.PrivacyBoundary`.
    """

    def __init__(
        self,
        sink: Sink | None = None,
        *,
        redactor: Redactor | None = None,
    ) -> None:
        self._sink: Sink = sink if sink is not None else NullSink()
        self._redactor: Redactor | None = redactor

    async def emit(self, event: TelemetryEvent) -> None:
        # Pick up correlation IDs from the current context if the
        # caller hasn't already filled them in. Existing IDs on the
        # event are respected — letting tests construct events with
        # explicit IDs without going through a scope.
        if event.trace_id is None:
            ctx_trace = _current_trace_id.get()
            if ctx_trace is not None:
                event.trace_id = ctx_trace
        if event.span_id is None:
            ctx_span = _current_span_id.get()
            if ctx_span is not None:
                event.span_id = ctx_span
        if event.parent_span_id is None:
            ctx_parent = _current_parent_span_id.get()
            if ctx_parent is not None:
                event.parent_span_id = ctx_parent

        # Apply the boundary redactor (if any) *before* fan-out so every
        # sink — JSONL, OTel, MultiSink fan-out — sees the same scrubbed
        # event. A redactor crash is not caught here: a buggy scrubber
        # is a configuration error and silently dropping events would be
        # worse than a loud failure.
        if self._redactor is not None:
            event = self._redactor(event)

        try:
            await self._sink.emit(event)
        except Exception:
            logger.warning("telemetry sink %r failed", self._sink, exc_info=True)

    @asynccontextmanager
    async def session_scope(self, trace_id: str | None = None) -> AsyncIterator[str]:
        """Set a fresh `trace_id` for the duration of the block.

        Pass `trace_id` to use a caller-supplied value (e.g., propagate
        from an upstream system); otherwise a fresh 32-hex (128-bit)
        ID is minted — this length matches OpenTelemetry's trace_id
        format so `OpenTelemetrySink` can use the value as-is when it
        synthesizes spans.

        Yields the trace_id so callers can record it externally if
        needed.
        """
        new_trace = trace_id if trace_id is not None else uuid4().hex
        token = _current_trace_id.set(new_trace)
        try:
            yield new_trace
        finally:
            _current_trace_id.reset(token)

    @asynccontextmanager
    async def span_scope(self, span_id: str | None = None) -> AsyncIterator[str]:
        """Open a nested span: the current `span_id` becomes the parent
        of the new one. Emitted events inside the block carry the new
        `span_id` and the previous one as `parent_span_id`.

        Pass `span_id` to use a caller-supplied value (e.g., when
        synthesizing IDs from an external trace context); otherwise a
        fresh 16-hex (64-bit) ID is minted — this length matches
        OpenTelemetry's span_id format so `OpenTelemetrySink` can use
        the value as-is when it synthesizes spans.

        Yields the new span_id.
        """
        new_span = span_id if span_id is not None else secrets.token_hex(8)
        previous_span = _current_span_id.get()
        span_token = _current_span_id.set(new_span)
        # The new span's parent is whatever span (if any) was current
        # when we opened. Concurrent span_scope() calls in different
        # tasks won't collide because contextvars copy on `asyncio.create_task`.
        parent_token = _current_parent_span_id.set(previous_span)
        try:
            yield new_span
        finally:
            _current_span_id.reset(span_token)
            _current_parent_span_id.reset(parent_token)

    @staticmethod
    def current_trace_id() -> str | None:
        """The trace_id of the active `session_scope`, or `None`."""
        return _current_trace_id.get()

    @staticmethod
    def current_span_id() -> str | None:
        """The span_id of the active `span_scope`, or `None`."""
        return _current_span_id.get()
