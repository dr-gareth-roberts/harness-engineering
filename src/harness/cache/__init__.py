"""Prefix-drift watcher (#3) — fingerprints rendered prompt prefixes per cache
breakpoint, surfaces silent invalidators.

Public API:

    from harness.cache import (
        PrefixWatcher, FingerprintStore,
        InMemoryFingerprintStore, FileFingerprintStore,
        DriftReport, DriftEvent,
    )

The `PrefixWatcher` satisfies `harness.runner.protocols.PrefixWatcherProtocol`
structurally — pass it as `prefix_watcher=` to `AnthropicRunner` /
`OpenAICompatRunner`.

The `cli` submodule registers a `cache-audit` subcommand on the top-level
`harness` CLI; `harness.cli` discovers it automatically.
"""

from harness.cache.audit import DriftEvent, DriftReport, audit
from harness.cache.store import (
    FileFingerprintStore,
    FingerprintRecord,
    FingerprintStore,
    InMemoryFingerprintStore,
)
from harness.cache.watcher import PrefixWatcher

__all__ = [
    "DriftEvent",
    "DriftReport",
    "FileFingerprintStore",
    "FingerprintRecord",
    "FingerprintStore",
    "InMemoryFingerprintStore",
    "PrefixWatcher",
    "audit",
]
