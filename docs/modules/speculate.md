# `harness.speculate`

Pre-execute likely tool calls in `asyncio` tasks while the model
generates. `Speculator` satisfies the runner's structural
`speculator=` protocol; hits skip the runner's
`PreToolUse` / dispatch / `PostToolUse` cycle. The lifecycle is
per-iteration: `begin` launches predictions, `observe` (per emitted
`tool_use` block) marks matches, `cancel_unobserved` (after stream-end)
cancels unmatched, `try_resolve` returns hits, `end` is a final safety
net.

Ships `LastCallPredictor`, `SequencePredictor` (bigram), and
`CrossSessionPredictor` (loads recent session records from a
`MemoryStore`). Idempotency is a tool-author promise — speculator
only fires for `Tool.idempotent=True` by default.

::: harness.speculate
