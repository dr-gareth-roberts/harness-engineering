"""Typed DAP message shapes — only the subset the harness adapter speaks.

DAP defines hundreds of fields across dozens of message types; we model
only what the adapter sends, receives, or surfaces to a tester. Each
field's optionality follows the DAP spec — required-by-spec fields are
non-optional here, optional ones are `... | None = None`.

Message taxonomy (from the spec):

- **Request**: editor → adapter. Has `seq`, `type="request"`, `command`,
  `arguments?`. The adapter must respond.
- **Response**: adapter → editor. Has `seq`, `type="response"`,
  `request_seq`, `success`, `command`, `body?`, optional `message` on
  failure.
- **Event**: adapter → editor. Has `seq`, `type="event"`, `event`, `body?`.
  Asynchronous; no response expected.

Wire-level dispatch is by `command` (for requests) or `event` (for
events) — we keep those as plain strings to avoid a closed-set Literal
that would force the adapter to reject anything new without code change.
The Pydantic models still validate the *shape* of the requests we
implement.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class _DapBase(BaseModel):
    """Permissive base — DAP fields use camelCase and we'd rather not
    fight that. `populate_by_name` lets us alias snake_case ↔ camelCase
    where it makes the Python side cleaner (`request_seq` → `requestSeq`).
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ---------------------------------------------------------------------------
# Envelope types


class Request(_DapBase):
    seq: int
    type: Literal["request"] = "request"
    command: str
    arguments: dict[str, Any] | None = None


class Response(_DapBase):
    seq: int
    type: Literal["response"] = "response"
    # request_seq is one of the few snake_case fields in the DAP spec —
    # keep both validation and serialization in snake_case.
    request_seq: int
    success: bool
    command: str
    message: str | None = None
    body: dict[str, Any] | None = None


class Event(_DapBase):
    seq: int
    type: Literal["event"] = "event"
    event: str
    body: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Request `arguments` shapes — decoded ad-hoc per command in dap.py to
# avoid coupling every command's schema to a model class. Helpers here
# are for the responses + events the adapter emits, where shape stability
# matters more.


class Capabilities(_DapBase):
    """Subset of `Capabilities` fields that we set or default explicitly.

    DAP defines ~50 capability flags. Most can be omitted — DAP treats
    absent capabilities as `false`. We surface only the ones whose answer
    is non-trivial for this adapter.
    """

    supports_configuration_done_request: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "supports_configuration_done_request",
            "supportsConfigurationDoneRequest",
        ),
        serialization_alias="supportsConfigurationDoneRequest",
    )
    supports_evaluate_for_hovers: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "supports_evaluate_for_hovers",
            "supportsEvaluateForHovers",
        ),
        serialization_alias="supportsEvaluateForHovers",
    )
    supports_terminate_request: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "supports_terminate_request",
            "supportsTerminateRequest",
        ),
        serialization_alias="supportsTerminateRequest",
    )
    # We accept setBreakpoints requests but only honor them as turn-index
    # equality (DAP line N → ctx.turn_index == N - 1). Per-source
    # breakpoints are how editors communicate "stop at trajectory point N".
    supports_breakpoint_locations_request: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "supports_breakpoint_locations_request",
            "supportsBreakpointLocationsRequest",
        ),
        serialization_alias="supportsBreakpointLocationsRequest",
    )


class Source(_DapBase):
    """A pseudo-source representing the trajectory.

    The adapter synthesizes a one-line-per-turn transcript and reports it
    as `name="trajectory"`, `path=None`, `sourceReference=N`. Editors
    fetch the body via the `source` request. Line numbers in `Source` are
    1-based per DAP — line 1 = turn 0.
    """

    name: str
    path: str | None = None
    source_reference: int | None = Field(
        default=None,
        validation_alias=AliasChoices("source_reference", "sourceReference"),
        serialization_alias="sourceReference",
    )
    presentation_hint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("presentation_hint", "presentationHint"),
        serialization_alias="presentationHint",
    )


class StackFrame(_DapBase):
    id: int
    name: str
    line: int
    column: int = 1
    source: Source | None = None


class Scope(_DapBase):
    name: str
    variables_reference: int = Field(
        validation_alias=AliasChoices("variables_reference", "variablesReference"),
        serialization_alias="variablesReference",
    )
    expensive: bool = False
    presentation_hint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("presentation_hint", "presentationHint"),
        serialization_alias="presentationHint",
    )


class Variable(_DapBase):
    name: str
    value: str
    type: str | None = None
    variables_reference: int = Field(
        default=0,
        validation_alias=AliasChoices("variables_reference", "variablesReference"),
        serialization_alias="variablesReference",
    )
    # DAP also defines `presentationHint`, `evaluateName`, `memoryReference`
    # — we don't surface them, so they're not modeled.


class Breakpoint(_DapBase):
    """A breakpoint as seen by DAP. The adapter returns `verified=True`
    for any line in the synthesized trajectory and `False` otherwise.
    """

    verified: bool
    line: int | None = None
    message: str | None = None
    source: Source | None = None
