# harness-engineering

Opensource toolbox for harness engineering — utilities, primitives, and patterns for building robust harnesses around LLM-powered agents and coding tools.

## Status

Greenfield. Scope and structure are being planned via `/ultraplan`.

## Scope (working draft)

The "harness" is everything around the model: prompt assembly, tool wiring, permission gating, hook execution, sub-agent dispatch, memory, retries, sandboxing, telemetry. This repo aims to collect reusable building blocks for that layer — independent of any one CLI or vendor — so harness authors can compose rather than rebuild.

Areas of interest:

- Prompt and context assembly (caching, compaction, file references)
- Tool schemas and dispatch
- Sub-agent orchestration and parallelism
- Hook lifecycles (pre/post tool, session start/end, prompt submit)
- Permission and sandbox primitives
- Memory and persistence patterns
- Telemetry, eval, and replay

## License

TBD — likely Apache-2.0 or MIT.
