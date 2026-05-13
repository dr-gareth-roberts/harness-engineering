# `harness.speculate`

Pre-execute likely tool calls in `asyncio` tasks while the model
generates. `Speculator` satisfies the runner's structural
`speculator=` protocol; hits skip the runner's
`PreToolUse` / dispatch / `PostToolUse` cycle. The lifecycle is
per-iteration: `begin` launches predictions, `observe` (per emitted
`tool_use` block) marks matches, `cancel_unobserved` (after
stream-end) cancels unmatched, `try_resolve` returns hits, `end`
is a final safety net.

Ships `LastCallPredictor`, `SequencePredictor` (bigram), and
`CrossSessionPredictor` (loads recent session records from a
`MemoryStore`). Idempotency is a tool-author promise — speculator
only fires for `Tool.idempotent=True` by default.

## When to reach for this

- Your tool handlers have non-trivial latency (DB query, external
  API, file I/O) and the next call is predictable from history.
- You want to overlap handler runtime with model generation rather
  than running them sequentially.
- You want a custom predictor (ML-based, business-rule-based) — the
  `Predictor` Protocol is one method, plug in any class.

## Quick example

<!-- reason: illustrative; AnthropicRunner needs the [anthropic] extra and references undefined dispatcher / hooks -->
<!--pytest.mark.skip-->
```python
from harness import AnthropicRunner, LastCallPredictor, Speculator

speculator = Speculator(
    predictor=LastCallPredictor(history_window=1),
    max_speculations=2,
    only_idempotent=True,   # default; safety net
)

runner = AnthropicRunner(dispatcher, hooks, speculator=speculator)
```

`SequencePredictor` for richer prediction (bigram model over the
call sequence). `CrossSessionPredictor.from_store(store, K=5)` pulls
the last 5 sessions for cross-session learning.

## Gotchas

- **Idempotency is a tool-author promise.** Speculator runs
  `Tool.idempotent=True` tools whether the model would have called
  them or not. Marking a side-effecting tool idempotent produces
  silent duplicate side effects on miss. Mark only when re-running
  with the same args is observably equivalent to running once.
- **Eager per-block cancellation only fires when `max_speculations==1`.**
  With multiple pending, cancellation defers to stream-end via
  `cancel_unobserved`. Multi-pending mid-stream cancellation is a
  policy question that wasn't worth the complexity.
- **Speculation contends with the model's stream** for SDK / network
  resources. `max_speculations` defaults to 2 to bound the wall-clock
  cost of a miss.
- **Speculator + streaming** (Wave 13a) — the speculator API is
  honored in both `__call__` and `run_stream`. Cancellation timing
  is preserved.

## Related

- [Cookbook: Cache + speculate](../cookbook/cache-and-speculate.md) — extended walkthrough.
- [`examples/speculate.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/speculate.py),
  [`examples/cross_session.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/cross_session.py)
- [`harness.runner`](runner.md) — `speculator=` kwarg on the vendor runners.

## API reference

::: harness.speculate
