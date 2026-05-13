# Cache + speculate for latency wins

## Problem

Your agent loop has visible latency. Two concrete sources you can
attack:

1. **Prompt cache misses** — Anthropic's prompt cache (and similar
   provider caches) saves you re-sending the same prefix every turn.
   But the cache invalidates on any prefix change. A timestamp in
   your system prompt, a randomly-ordered tool list, even whitespace
   drift — all silently invalidate the cache and you wonder why the
   first token took 8s today and 800ms yesterday.
2. **Sequential tool dispatch** — when the model emits a `tool_use`,
   the next thing your runner does is execute the handler, then
   send the result back. The model is idle during dispatch.

`harness.cache` and `harness.speculate` address each.

## Cache drift audit (`PrefixWatcher`)

`PrefixWatcher` plugs into the runner's `prefix_watcher=` slot. It
fingerprints each Anthropic `cache_control` breakpoint per request
and stores the fingerprints in a `FingerprintStore`. Every subsequent
request, it compares the live fingerprint against the stored one and
emits a `DriftEvent` if they differ.

<!-- reason: illustrative; AnthropicRunner needs Dispatcher with tools and the [anthropic] extra -->
<!--pytest.mark.skip-->
```python
from harness import (
    AnthropicRunner,
    Dispatcher,
    FileFingerprintStore,
    HookRunner,
    PrefixWatcher,
)

watcher = PrefixWatcher(store=FileFingerprintStore("./cache-prints"))
runner = AnthropicRunner(
    Dispatcher(),
    HookRunner(),
    prefix_watcher=watcher,
)
```

Drift events fire silently. To audit them in batch:

```bash
uv run harness cache-audit --store ./cache-prints --since 24h
```

`--since` accepts `24h`, `7d`, `30m`, `2w` forms. `--store` may be
either the JSONL file or the directory it lives in.

Output: a unified-diff per drift, so you can see exactly which
characters of your "stable" system prompt changed:

```
breakpoint 0 drifted at 2026-05-09T14:22:11
--- old prefix (2026-05-09T13:55:02)
+++ new prefix (2026-05-09T14:22:11)
@@ -3,7 +3,7 @@
 You are a helpful assistant.
-Today is 2026-05-09T13:55:02.123456+00:00
+Today is 2026-05-09T14:22:11.987654+00:00
 Do not exceed 50 tokens.
```

There's the timestamp leak. Strip it from your prompt prefix; the
cache hit comes back.

## Speculative tool execution

When your tool-use loop is dominated by handler latency (DB query,
API call, file read), pre-execute the likely next call in parallel
with the model's generation. On hit, you skip the handler runtime
entirely. On miss, the speculation is cancelled — at stream-end at
the latest, eagerly inside `observe()` for the simple
`max_speculations=1` case.

<!-- reason: illustrative; references undefined dispatcher / hooks and needs the [anthropic] extra -->
<!--pytest.mark.skip-->
```python
from harness import LastCallPredictor, Speculator

speculator = Speculator(
    predictor=LastCallPredictor(history_window=1),
    max_speculations=1,
)
runner = AnthropicRunner(dispatcher, hooks, speculator=speculator)
```

`LastCallPredictor` simply re-runs whatever tool the model called
last turn — useful when the model is iterating on a search/refine
loop. `SequencePredictor` builds a bigram model over the call
sequence for richer prediction. `CrossSessionPredictor` pulls the
top-K most-recent SessionRecords from a `MemoryStore` and predicts
from the union, so a fresh agent benefits from past patterns.

### Custom predictor

Anything satisfying the `Predictor` Protocol works:

<!-- reason: illustrative; class body uses `...` and references undefined Message / Tool -->
<!--pytest.mark.skip-->
```python
from harness.speculate import Predictor
from harness.tools.schema import ToolCall

class MyMLPredictor:
    def predict(
        self,
        history: list[Message],
        idempotent_tools: dict[str, Tool],
        max_predictions: int,
    ) -> list[ToolCall]:
        # Your logic. Return the calls you'd like pre-executed.
        ...

speculator = Speculator(MyMLPredictor(), max_speculations=2)
```

## Combine them

<!-- reason: illustrative; references undefined dispatcher / hooks and needs the [anthropic] extra -->
<!--pytest.mark.skip-->
```python
runner = AnthropicRunner(
    dispatcher,
    hooks,
    prefix_watcher=PrefixWatcher(store=FileFingerprintStore("./prints")),
    speculator=Speculator(LastCallPredictor(), max_speculations=2),
)
```

Both kwargs are structurally typed so neither module imports the
other; the runner has zero runtime dependency on either.

## Gotchas

- **Idempotency is a tool-author promise.** The speculator pre-executes
  tools marked `Tool.idempotent=True`. If you mark a tool idempotent
  but it has side effects (sending email, writing to a DB),
  speculative miss runs cause silent duplicate side effects. Mark a
  tool idempotent only if re-running with the same args is observably
  equivalent to running it once.
- **Cache breakpoints cap at 4** in Anthropic. The runner enforces
  this client-side, so you get a typed
  `CacheBreakpointLimitExceeded` instead of an opaque API 400.
- **`harness cache-audit` only surfaces drift, not "why."** The
  unified diff shows you what changed; you still have to fix the
  source.
- **Speculator + streaming** — the speculator API is honored in
  both `__call__` and `run_stream`. The "before dispatch"
  cancellation timing is preserved.

## Related

- [`harness.cache`](../modules/cache.md), [`harness.speculate`](../modules/speculate.md) — module references.
- [`examples/cache.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/cache.py),
  [`examples/speculate.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/speculate.py),
  [`examples/cross_session.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/cross_session.py)
  — runnable demos.
- [CLI reference](../cli.md#harness-cache-audit) — `harness cache-audit` flags.
