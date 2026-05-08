"""Telemetry events for the speculative-execution loop.

Three events fire over the lifetime of a tool-use loop:

* `SpeculationLaunched` — when `Speculator.begin` kicks off a background
  task to dispatch a predicted tool call. One per launched task.
* `SpeculationHit` — when `Speculator.try_resolve` finds a matching
  pending speculation and returns its cached result.
* `SpeculationMiss` — when `Speculator.try_resolve` finds no match for
  the model's actual call.

Hit-rate and latency analysis use `(launched, hit, miss)` counts. Wire
the speculator's `telemetry=` kwarg to a `MemorySink` to inspect events
in tests, or to a `JSONLSink` for production accounting.
"""

from __future__ import annotations

from typing import Literal

from harness.telemetry.events import TelemetryEvent


class SpeculationLaunched(TelemetryEvent):
    """A predicted tool call was dispatched as a background task."""

    kind: Literal["speculation.launched"] = "speculation.launched"
    tool_name: str


class SpeculationHit(TelemetryEvent):
    """The model's actual tool call matched a pending speculation."""

    kind: Literal["speculation.hit"] = "speculation.hit"
    tool_name: str


class SpeculationMiss(TelemetryEvent):
    """The model's actual tool call did NOT match any pending speculation."""

    kind: Literal["speculation.miss"] = "speculation.miss"
    tool_name: str
