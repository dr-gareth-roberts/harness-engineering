# `harness.telemetry`

Pluggable `Sink` protocol plus `JSONLSink`, `MemorySink`, and
`MultiSink`. `OpenTelemetrySink` lives behind the `[otel]` extra;
events ride as flat OTel events on the active span.

::: harness.telemetry
