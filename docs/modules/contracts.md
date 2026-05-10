# `harness.contracts`

Declarative invariants over agent trajectories. Predicates compose
with `&` / `|`; patterns include `Always` / `Eventually` /
`Earlier(...).when(...)` / `Never`. A shared DFA backs both runtime
enforcement (`attach_contracts(hooks, ...)`) and offline auditing
(`check(record, ...)`) — a contract that passes online passes
offline and vice versa.

## When to reach for this

- You want to assert a property on every session: "the agent never
  calls `delete_user` before calling `confirm`."
- You want to test the property both at runtime (block / warn) and
  in CI against recorded sessions.
- You want declarative invariants instead of imperative `if`-ladders
  inside hooks.

## Quick example

```python
from harness.contracts import (
    Always, Earlier, Eventually, HasToolUse, Never,
    attach_contracts, check,
)

# Compose predicates with & and |.
delete = HasToolUse(name="delete_user")
confirm = HasToolUse(name="confirm")

# A `require` contract: every delete must follow a confirm.
contracts = [
    Earlier(confirm).when(delete).require("confirm-before-delete"),
    Never(HasToolUse(name="exfiltrate")).forbid("no-data-exfil"),
]

# Runtime: blocks (or warns, depending on action) at violation time.
hooks = HookRunner()
attach_contracts(hooks, contracts)

# Offline: check a recorded SessionRecord.
violations = check(record, contracts)
for v in violations:
    print(f"{v.contract.name}: {v.reason}")
```

Three actions: `forbid` (block), `warn` (record + continue),
`require` (must hold by SessionEnd).

## Gotchas

- **The same DFA backs both modes.** A contract that passes runtime
  passes offline, and vice versa. If they ever diverge, that's a
  bug in the DFA compiler.
- **`require` contracts only fire at SessionEnd**, by definition.
  If the session aborts mid-way, the require may not have had a
  chance to satisfy.
- **`Earlier(P).when(Q)`** means "for every Q, P must have happened
  earlier in the trajectory." Order matters; this isn't symmetric.
- **Predicate composition (`&` / `|`)** short-circuits the same way
  Python's `and` / `or` do.

## Related

- [`examples/contracts.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/contracts.py) — runtime + offline equivalence demo.
- [`harness.plan`](plan.md) — `PlanGuardedRunner` builds on contracts.
- [`harness.hooks`](hooks.md) — what `attach_contracts` registers under.

## API reference

::: harness.contracts
