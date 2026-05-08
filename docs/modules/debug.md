# `harness.debug`

`pdb`-flavored debugger for orchestrator runs. `DebugRunner(real_runner, ...)`
wraps any runner and pauses on a configurable predicate, exposing a
`DebugContext` for inspect / mutate / fire / resume / abort. Three
modes: programmatic (callback), interactive REPL (`harness debug`),
and DAP server over stdio (`harness debug --dap`) — see [CLI](../cli.md).

::: harness.debug
