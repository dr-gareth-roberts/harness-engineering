# `harness.runner`

Pluggable runners. `EchoRunner` and `CannedRunner` ship with no
external deps; `AnthropicRunner` (extra `[anthropic]`) and
`OpenAICompatRunner` (extra `[openai-compat]`) drive real models.
Both vendor runners accept structural `prefix_watcher=` and
`speculator=` kwargs.

::: harness.runner
