# `harness.cache`

Prompt-prefix-drift watcher. `PrefixWatcher` satisfies the runner's
structural `prefix_watcher=` protocol and fingerprints each cache
breakpoint per request. `audit(store, window_hours)` surfaces silent
invalidators in unified-diff form. Ships the `harness cache-audit`
CLI subcommand.

## When to reach for this

- You're using Anthropic's prompt cache (or any provider cache) and
  cache hits are inconsistent across runs.
- You suspect a system-prompt timestamp leak, a randomly-ordered
  tool list, or whitespace drift is invalidating the cache.
- You want a unified-diff audit so you can see exactly which
  characters of your "stable" prefix changed.

## Quick example

<!-- reason: illustrative; references undefined dispatcher / hooks and needs the [anthropic] extra -->
<!--pytest.mark.skip-->
```python
from harness import (
    AnthropicRunner, FileFingerprintStore, PrefixWatcher,
)

watcher = PrefixWatcher(store=FileFingerprintStore("./cache-prints"))
runner = AnthropicRunner(dispatcher, hooks, prefix_watcher=watcher)

# Run normally. Drift is recorded; query later via:
#   uv run harness cache-audit ./cache-prints --window-hours 24
```

The audit CLI prints unified diffs per drift event:

```
breakpoint 0 drifted at 2026-05-09T14:22:11
--- old prefix (2026-05-09T13:55:02)
+++ new prefix (2026-05-09T14:22:11)
@@ -3,7 +3,7 @@
 You are a helpful assistant.
-Today is 2026-05-09T13:55:02.123456+00:00
+Today is 2026-05-09T14:22:11.987654+00:00
```

## Gotchas

- **The watcher only audits drift; it doesn't fix it.** The diff
  shows what changed; you fix the source.
- **Fingerprints are per-cache-breakpoint.** Anthropic caps at 4
  breakpoints per request; the runner enforces that cap client-side
  (Wave 10 #12). Fingerprint store size scales with breakpoint
  count × distinct content.
- **`FileFingerprintStore` is one file per session_id × breakpoint
  index.** Fine for personal / small-scale; consider a
  custom backing store for high-volume production.
- **Provider cache invalidation isn't observable from outside** —
  the watcher's diffs are *inputs* that *would* invalidate the
  cache, not confirmation that it actually did.

## Related

- [Cookbook: Cache + speculate](../cookbook/cache-and-speculate.md) — extended walkthrough.
- [`examples/cache.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/cache.py)
- [CLI reference](../cli.md#harness-cache-audit) — full flag list.

## API reference

::: harness.cache
