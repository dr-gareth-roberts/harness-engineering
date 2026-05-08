# `harness.cache`

Prompt-prefix-drift watcher. `PrefixWatcher` satisfies the runner's
structural `prefix_watcher=` protocol and fingerprints each cache
breakpoint per request. `audit(store, window_hours)` surfaces silent
invalidators in unified-diff form. Ships the `harness cache-audit`
CLI subcommand.

::: harness.cache
