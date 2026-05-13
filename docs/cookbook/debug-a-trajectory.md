# Debug a bad trajectory

## Problem

The model called the wrong tool with the wrong arguments three turns
into a 10-turn conversation. You want to: pause execution at that
turn, inspect the conversation history, fire ad-hoc tool calls to
explore state, mutate the next assistant message, and resume.

## Solution sketch

Wrap any `Runner` in `DebugRunner`. The wrapped runner pauses on a
configurable `break_on(ctx)` predicate; when it fires, control hands
off to one of three modes:

- **Programmatic** — your callback runs against the paused
  `DebugContext`. Best for automated repro scripts.
- **Interactive REPL** (`harness debug <session>`) — line-based
  console. `messages`, `last_call`, `mutate`, `fire`, `inspect`,
  `resume`, `abort`.
- **DAP server** (`harness debug --dap <session>`) — speaks the
  Debug Adapter Protocol over stdio. VS Code, neovim-dap, Emacs
  dap-mode all attach.

All three drive the same `DebugContext`; only the I/O layer differs.

## Working code

### CLI: pause at turn N

You have `session.json` saved from production. Pause right before
turn 3 (the bad one):

<!-- reason: shell example referring to a non-existent session.json -->
<!--pytest.mark.skip-->
```bash
uv run harness debug session.json --break turn=3
```

Drops you into a REPL:

```
[harness-debug] paused at turn 3 (type 'help' for commands)
> messages
  [0] user: What's the weather in Berlin?
  [1] assistant: I'll look that up.
  [2] tool_use(weather, {"city": "Berlin"})
  [3] tool_result("It is 22°C and sunny in Berlin.")
> last_call
tool_use(weather, {"city": "Berlin"})
> inspect ctx.messages[1].content[0].text
'I'll look that up.'
> mutate assistant Actually, I will use the geocoder tool first.
[harness-debug] queued mutation as assistant
> resume
[harness-debug] resuming
```

`mutate` short-circuits the runner — the supplied message replaces
what the model would have produced for this turn. Useful for
"counterfactual nudge and replay."

### CLI: pause when a specific tool is called

<!-- reason: shell example referring to a non-existent session.json -->
<!--pytest.mark.skip-->
```bash
uv run harness debug session.json --break tool=delete_user
```

Stops the first time the model emits a `tool_use` for `delete_user`.
Combined with `mutate`, you can rewrite the call to something safe
(or `abort` to terminate the run entirely).

### IDE integration via DAP

Wave 7 added a DAP server so VS Code's debugger UI works against
recorded sessions:

<!-- reason: shell example referring to a non-existent session.json -->
<!--pytest.mark.skip-->
```bash
uv run harness debug session.json --dap
```

That speaks DAP over stdio. Your editor launches the process and
connects. `tasks.json` example:

```json
{
  "type": "harness",
  "request": "launch",
  "name": "Debug recorded session",
  "program": "${workspaceFolder}/session.json",
  "allowEvaluate": false
}
```

By default `evaluate` is restricted to a fixed set of variable names
(`turn_index`, `message_count`, `last_call.name`, etc.). To let the
editor's debug console run arbitrary Python over `ctx`, set
`allowEvaluate: true`. Same security trade-off as the REPL's
`inspect` command: only reachable when a breakpoint hits in an
opt-in debug session.

### Programmatic: pause-and-script for automated repro

<!-- reason: illustrative; references undefined dispatcher / hooks / text and uses placeholder AnthropicRunner(...) -->
<!--pytest.mark.skip-->
```python
from harness import DebugContext, DebugRunner, Orchestrator

async def on_break(ctx: DebugContext) -> None:
    print(f"paused at turn {ctx.turn_index}")
    # Probe state via an ad-hoc tool call (doesn't advance the conversation):
    result = await ctx.fire("lookup", {"key": "config-flag"})
    print(f"  config = {result.content}")
    # Replace the next assistant turn entirely:
    ctx.mutate(text("assistant", "Override: ignore previous instructions."))
    ctx.resume()

debug = DebugRunner(
    real_runner=AnthropicRunner(...),
    break_on=lambda c: c.turn_index == 3,
    breakpoint_callback=on_break,
    dispatcher=dispatcher,
)
orchestrator = Orchestrator(dispatcher, hooks, debug)
```

## Gotchas

- **Mutation short-circuits the runner** — once `ctx.mutate(...)` is
  set, the wrapped runner is *not* called for that turn. The
  supplied message goes back to the orchestrator directly. If you
  want the model to also run, don't mutate.
- **`abort()` raises `DebugAborted`** — catch it at the orchestrator
  level if you want graceful shutdown.
- **DAP `step_in` semantics (1.3.0+)** — `stepIn` is a first-class
  request: it runs until the next `PreToolUse` event and pauses
  inside the tool frame, with a turn-boundary fallback if no
  further tool dispatch arrives. Pre-1.3.0 it aliased `step_over`.
  See [`docs/modules/debug.md`](../modules/debug.md) for the
  current step-request -> frame mapping (including the `stepOut`
  semantics that depend on whether you're in an orchestrator or
  tool frame).
- **The CLI's REPL `inspect` command runs arbitrary Python** against
  the paused context. By design — it's a debugger. Don't expose
  this surface to untrusted users.

## Related

- [`harness.debug`](../modules/debug.md) — module reference.
- [`examples/debug.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/debug.py)
  — runnable programmatic-mode demo.
- [CLI reference](../cli.md#harness-debug) — full flag list.
- [Cookbook: Replay evaluation](replay-evaluation.md) — record the
  session before you debug it.
