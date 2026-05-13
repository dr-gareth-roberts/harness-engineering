# CLI

```bash
$ harness --help
```

The `harness` console script exposes a small set of subcommands. Each
runs against artifacts the package itself produces (recorded sessions,
fingerprint stores) so the CLI is useful even without a live model.

## `harness debug`

Replay a recorded session through `DebugRunner` and pause on a
breakpoint. Two modes:

### Interactive REPL (default)

```bash
harness debug path/to/session.json --break turn=2
```

Pauses at `turn_index == 2` and drops into a line-based REPL:

```
[harness-debug] paused at turn 2 (type 'help' for commands)
> messages
  [0] user: hi
  [1] assistant: text(...)
  [2] user: ...
> last_call
tool_use(search, {"q": "rust"})
> mutate assistant rewritten reply
[harness-debug] queued mutation as assistant
> resume
[harness-debug] resuming
```

Break specs:

| Spec | Stops when |
|---|---|
| `turn=N` | `ctx.turn_index == N` |
| `tool=NAME` | the most recent `tool_use` had name `NAME` |

### DAP server (`--dap`)

```bash
harness debug path/to/session.json --dap
```

Speaks the [Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/)
over stdio. Editors that ship DAP clients (VS Code, neovim-dap, Emacs
dap-mode) launch the process, send DAP requests, and drive the same
replay-driven debug session. Setting a breakpoint at line N in the
synthesized trajectory source pauses right before producing the Nth
assistant turn.

A VS Code launch config (illustrative — adapt the `program` path):

```json
{
  "type": "harness",
  "request": "launch",
  "name": "Debug recorded session",
  "program": "${workspaceFolder}/sessions/example.json"
}
```

The DAP subset implemented covers initialize, launch, setBreakpoints,
configurationDone, threads, stackTrace, scopes, variables, evaluate
(limited to the variables-view names — arbitrary expressions stay in
the REPL), source, continue, next, stepIn, stepOut, pause, terminate,
disconnect; events: initialized, stopped, continued, output,
terminated, exited.

Stepping semantics (frame-aware since 1.3.0 / Wave 13b M3.6):

- `next` (step_over) runs to the next turn boundary, ignoring tool
  dispatches in between.
- `stepIn` (step_in) runs until the next `PreToolUse` event and
  pauses inside the tool frame; if no further tool dispatch arrives
  before the next turn boundary, it falls back to pausing there so
  the editor's step-in button is never silently unresponsive.
- `stepOut` (step_out) from inside a tool frame runs until the
  current dispatch's `PostToolUse` fires, then pauses at the next
  event (another `PreToolUse` in the same tool-use loop, or the
  next turn boundary). From outside a tool frame, `stepOut` has no
  outer frame to return to and falls back to step_over semantics.
- `pause` is honored mid-trajectory: the next break check fires
  unconditionally and the runner stops at the next opportunity.

Frame-aware stepping requires the adapter to observe `PreToolUse`
/ `PostToolUse` events. `harness debug --dap` wires this
automatically via `DapAdapter.attach_hooks(hooks)`. If a host
omits that call (legacy wiring), `stepIn` / `stepOut` degrade to
per-turn `step_over` rather than being silently ignored.

## `harness cache-audit`

Audit a fingerprint store for prefix-cache drift over a window of
recent fingerprints. Surfaces silent invalidators in unified-diff
form so you can tell *what* changed in the prompt prefix that
invalidated the prompt cache.

```bash
harness cache-audit --store path/to/fingerprint-store --since 24h
```

| Flag | Meaning |
|---|---|
| `--store PATH` | Required. Either the `fingerprints.jsonl` file or the directory containing it. |
| `--since SPEC` | Audit window (default `24h`). Accepts `24h`, `7d`, `30m`, `2w`. |

Backed by `harness.cache.PrefixWatcher` and a `FileFingerprintStore`.

## Help

```bash
harness --help
harness debug --help
harness cache-audit --help
```
