# `harness.debug`

`pdb`-flavored debugger for orchestrator runs. `DebugRunner(real_runner, ...)`
wraps any runner and pauses on a configurable predicate, exposing
a `DebugContext` for inspect / mutate / fire / resume / abort.
Three modes: programmatic (callback), interactive REPL
(`harness debug`), and DAP server over stdio (`harness debug --dap`).

## When to reach for this

- A recorded session has a bad turn and you want to pause there to
  poke at state.
- You want a CLI debugger like `pdb` but for agent trajectories.
- You want to drive the same debug session from VS Code,
  neovim-dap, or Emacs dap-mode.

## Quick example

CLI:

```bash
uv run harness debug session.json --break turn=3
uv run harness debug session.json --break tool=delete_user
uv run harness debug session.json --dap         # speaks DAP over stdio
```

Programmatic:

```python
from harness import DebugContext, DebugRunner, Orchestrator

async def on_break(ctx: DebugContext) -> None:
    print(f"paused at turn {ctx.turn_index}")
    result = await ctx.fire("lookup", {"key": "config-flag"})
    ctx.mutate(text("assistant", "Override the next reply."))
    ctx.resume()

debug = DebugRunner(
    real_runner=runner,
    break_on=lambda c: c.turn_index == 3,
    breakpoint_callback=on_break,
    dispatcher=dispatcher,
)
orchestrator = Orchestrator(dispatcher, hooks, debug)
```

## Gotchas

- **Mutation short-circuits the runner.** Once `ctx.mutate(...)` is
  set, the wrapped runner is *not* called for that turn. The
  supplied message goes back to the orchestrator directly.
- **`abort()` raises `DebugAborted`** — catch it at the orchestrator
  level if you want graceful shutdown.
- **DAP `step_in` aliases `step_over` today** — the runner doesn't
  expose a one-shot pre-tool-use breakpoint surface yet. `next` and
  `stepOut` work as expected.
- **DAP `evaluate` is restricted by default** to a fixed set of
  variable names. Set `allowEvaluate: true` in the launch
  arguments to enable arbitrary-Python expressions over `ctx`
  (Wave 13b #17).
- **The CLI's REPL `inspect` command runs arbitrary Python** against
  the paused context. By design — it's a debugger; only opt-in.

## Related

- [Cookbook: Debug a trajectory](../cookbook/debug-a-trajectory.md) — extended walkthrough including DAP.
- [`examples/debug.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/debug.py) — programmatic-mode demo.
- [CLI reference](../cli.md#harness-debug) — full flag list.
- [`harness.replay`](replay.md) — record sessions before debugging them.

## API reference

::: harness.debug
