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

<!-- reason: shell example; refers to a non-existent session.json -->
<!--pytest.mark.skip-->
```bash
uv run harness debug session.json --break turn=3
uv run harness debug session.json --break tool=delete_user
uv run harness debug session.json --dap         # speaks DAP over stdio
```

Programmatic:

<!-- reason: illustrative; references undefined runner / dispatcher / hooks / text -->
<!--pytest.mark.skip-->
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
- **DAP `evaluate` is restricted by default** to a fixed set of
  variable names. Set `allowEvaluate: true` in the launch
  arguments to enable arbitrary-Python expressions over `ctx`
  (Wave 13b #17).
- **The CLI's REPL `inspect` command runs arbitrary Python** against
  the paused context. By design — it's a debugger; only opt-in.

## DAP step semantics

The DAP adapter distinguishes two execution frames so editor step
buttons behave like a developer expects, rather than aliasing to
"advance one turn":

- `orchestrator` — the natural state of the session: between tool
  dispatches, including while the assistant message that triggered
  them is being produced.
- `tool` — inside a single tool dispatch, between `PreToolUse` and
  `PostToolUse`.

Step requests map to these frames as follows:

| DAP request | Action |
|---|---|
| `next` (step_over) | Run to the next turn boundary, ignoring tool dispatches in between. |
| `stepIn` | Run until the next `PreToolUse` event, then pause inside the tool frame. |
| `stepOut` | From a tool frame: run past the current `PostToolUse`, pause at the next event (next `PreToolUse` or next turn boundary). From an orchestrator frame: same as `next`. |

Fallbacks (documented so an editor user never sees a silent
no-op):

- **`stepIn` with no follow-up tool dispatch** — if the session
  reaches the next turn boundary without firing another
  `PreToolUse`, `break_on_predicate` pauses at the turn boundary
  instead.
- **`stepOut` with no follow-up event** — same: if no further
  `PreToolUse` arrives before the next turn boundary,
  `break_on_predicate` pauses at the turn boundary.
- **`stepOut` from an orchestrator frame** — there is no outer
  frame to return to, so the request is promoted to `next`
  (step_over).

Frame-aware stepping needs the adapter to observe `PreToolUse` /
`PostToolUse` events directly. Wire that with
`adapter.attach_hooks(hooks)` before starting the orchestrator
session:

<!-- reason: illustrative; references undefined runner / dispatcher / record -->
<!--pytest.mark.skip-->
```python
from harness.debug.dap import DapAdapter
from harness.debug.runner import DebugRunner
from harness.hooks.runner import HookRunner

adapter = DapAdapter()
hooks = HookRunner()
adapter.attach_hooks(hooks)        # M3.6 — enables stepIn / stepOut

debug = DebugRunner(
    real_runner=runner,
    break_on=adapter.break_on_predicate,
    breakpoint_callback=adapter.breakpoint_callback,
    dispatcher=dispatcher,
)
# Pass `hooks` into the Orchestrator so the runner emits PreToolUse
# / PostToolUse through the same HookRunner the adapter listens on.
```

`harness debug --dap` (the CLI entry point) wires this for you. If
you call `DapAdapter` directly and omit `attach_hooks`, `stepIn` and
`stepOut` degrade to `next` so the editor's buttons still pause
*somewhere* — they just operate at coarser per-turn granularity.

### Pre-1.1.0 limitation

Before 1.1.0 (`audit/RELEASE-TODO.md` M3.6), all three step requests
hard-aliased to `step_over`: `stepIn` and `stepOut` each set the
same per-turn pause flag, so an editor user pressing "Step Into" got
"advance one turn" instead of "enter this tool dispatch." The fix
adds the hook-listener path described above; existing wiring keeps
working because the new behavior only kicks in when
`attach_hooks(...)` is called.

## Related

- [Cookbook: Debug a trajectory](../cookbook/debug-a-trajectory.md) — extended walkthrough including DAP.
- [`examples/debug.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/debug.py) — programmatic-mode demo.
- [CLI reference](../cli.md#harness-debug) — full flag list.
- [`harness.replay`](replay.md) — record sessions before debugging them.

## API reference

::: harness.debug
