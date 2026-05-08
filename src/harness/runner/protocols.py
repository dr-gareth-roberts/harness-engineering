"""Structural protocols runners accept as constructor kwargs.

These live here (not inside the vendor-specific runner files) so each
vendor runner can import them without having to redeclare the same shape,
and so feature modules (`harness.cache`, `harness.speculate`) can satisfy
the protocol without taking a runtime dependency on any one vendor SDK.
"""

from __future__ import annotations

from typing import Any, Protocol


class PrefixWatcherProtocol(Protocol):
    """Anything callable as `await watcher.fingerprint(request)`.

    The `harness.cache.PrefixWatcher` (Wave-2 feature #3) implements this.
    Runners call `await prefix_watcher.fingerprint(request)` once per
    iteration of their tool-use loop, immediately before the model call.

    The `request` dict is whatever the runner is about to send to its SDK
    — its exact shape is vendor-specific, but the watcher only needs it
    to compute a stable byte-level fingerprint, so it treats the value
    as opaque JSON-serializable data.
    """

    async def fingerprint(self, request: dict[str, Any]) -> None: ...
