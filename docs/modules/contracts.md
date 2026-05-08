# `harness.contracts`

Declarative invariants over agent trajectories. Predicates compose
with `&` / `|`; patterns include `Always` / `Eventually` /
`Earlier(...).when(...)` / `Never`. A shared DFA backs both runtime
enforcement (`attach_contracts(hooks, ...)`) and offline auditing
(`check(record, ...)`) so a contract that passes online passes
offline and vice versa.

::: harness.contracts
