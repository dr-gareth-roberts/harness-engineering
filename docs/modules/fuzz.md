# `harness.fuzz`

Hypothesis-driven tool and agent fuzzing (extra `[fuzz]`). `fuzz_tool`
drives Pydantic-typed inputs through `Dispatcher.dispatch` and
collects failures; `fuzz_agent` does the same through a full
`Orchestrator` turn. The `harness_property` pytest decorator wires
generated inputs into a property-based test.

::: harness.fuzz
