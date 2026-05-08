## Wave 7 — DAP for debug REPL

### Goal
Let editors (VS Code, neovim-dap, Emacs dap-mode, etc.) drive the same
replay-based debug session that `harness debug` already supports
interactively. The user picks an editor frame, sets breakpoints on
trajectory turns, sees `stopped` events, browses scopes/variables/the
synthesized source, and resumes — without leaving the IDE.

### Status
Shipped on `feature/dap-debug`. Single-coherent refactor in main, no
parallel agents.

### Architecture

`harness.debug.dap.DapAdapter` is the bridge: a long-lived state
holder that runs a DAP message loop (`serve`) on one asyncio task and
the orchestrator session on another, sharing the event loop. The
breakpoint pump (`_on_breakpoint`) is wired into `DebugRunner` as the
`breakpoint_callback`; on hit it parks on `_continue_event` while the
DAP read-loop keeps pumping inspect requests against the held
`DebugContext`.

This concurrency is the load-bearing property of the design. A
sequential implementation (read one DAP message → handle it → loop)
would deadlock the editor on every `evaluate`/`variables`/`scopes`
during a breakpoint, because the message loop would itself be parked
inside the breakpoint callback. The test
`test_inspect_requests_pump_during_breakpoint_hold` pins this — it
fires four interleaved inspect requests during a breakpoint hold and
asserts each gets a response *before* `continue` is sent.

### Source mapping

DAP frames carry `Source` + `line`, so an editor without a real source
file is hard to drive. The adapter synthesizes one line per assistant
turn (caller supplies `synthesize_source: () -> list[str]`). DAP line
N (1-based) maps to `ctx.turn_index == N - 1`; setting a breakpoint at
line 5 in the synthesized source pauses right before producing the
5th assistant turn. The CLI's `_trajectory_lines` summarizes each
assistant message (text blocks first, then `(tool_use <name>)`).

### DAP subset implemented

| Group | Surface |
|---|---|
| Requests (responded) | `initialize`, `launch`, `setBreakpoints`, `configurationDone`, `threads`, `stackTrace`, `scopes`, `variables`, `evaluate`, `source`, `continue`, `next`, `stepIn`, `stepOut`, `pause`, `terminate`, `disconnect` |
| Events (emitted) | `initialized`, `stopped`, `continued`, `output`, `terminated`, `exited` |
| Capabilities advertised | `supportsConfigurationDoneRequest`, `supportsEvaluateForHovers`, `supportsTerminateRequest` |

`next` / `stepIn` / `stepOut` / `pause` are accepted but treated as
`continue` — agent trajectories don't have a meaningful intra-turn
step granularity yet. The handlers exist so editors that rely on these
capabilities don't error out.

`evaluate` is limited to looking up the same names the `variables`
view exposes (`turn_index`, `message_count`, `last_call.name`,
`last_call.arguments`, `pending_mutation.role`). Arbitrary-expression
evaluation is intentionally out of scope for the DAP surface; the
interactive REPL (`harness debug` without `--dap`) is the place for
that. This rationale is documented in the module docstring.

### Files

| File | Lines | Purpose |
|---|---|---|
| `src/harness/debug/dap_protocol.py` | ~115 | Content-Length + JSON framing over `asyncio.StreamReader/Writer`. Distinguishes graceful EOF (caller treats as disconnect) from truncation/malformed input (raises `DapProtocolError`). |
| `src/harness/debug/dap_messages.py` | ~140 | Pydantic models for `Request`, `Response`, `Event`, `Capabilities`, `Source`, `StackFrame`, `Scope`, `Variable`, `Breakpoint`. snake_case ↔ camelCase aliasing follows the spec field-by-field via `validation_alias=AliasChoices(...)` + `serialization_alias`. |
| `src/harness/debug/dap.py` | ~440 | `DapAdapter` — the bridge. |
| `src/harness/debug/cli.py` | +85 | `--dap` flag on `harness debug`; runs the same replay-driven session under DAP control over stdio. |

### CLI

```
$ harness debug --help
usage: harness debug [-h] [--break BREAK_SPEC] [--dap] path

  --dap                 Speak the Debug Adapter Protocol over stdio
                        instead of running the interactive REPL.
```

VS Code launch config example (illustrative — not shipped in repo):

```json
{
  "type": "harness",
  "request": "launch",
  "name": "Debug recorded session",
  "program": "${workspaceFolder}/path/to/session.json"
}
```

### Tests added

| File | Test count | Coverage |
|---|---|---|
| `tests/debug/test_dap_protocol.py` | 16 | round-trip, header tolerance (case insensitive, extra headers), malformed input (missing/invalid/negative Content-Length, bad header lines, invalid JSON, non-object body), EOF semantics (clean vs mid-headers vs mid-body), back-to-back messages, UTF-8 framing. |
| `tests/debug/test_dap.py` | 13 | initialize → initialized event sequence, setBreakpoints validation against synthesized source length, full launch → break → continue → terminated flow, **concurrent inspect during breakpoint hold (the load-bearing test)**, evaluate (supported + unsupported), source request (known + unknown reference), disconnect mid-breakpoint aborts, unknown command error response, launch-without-run-session error, aborted session does not propagate. |

29 new tests, **494 total** (was 465).

### Design constraint that surfaced

The pre-tool-use security hook blocks `eval(` literal in new files —
correctly, since arbitrary-expression evaluation is high-risk surface.
The DAP `evaluate` handler resolves this by *not* offering arbitrary
expression evaluation; it looks up names from a fixed snapshot of the
`DebugContext`'s public scope (the same set the `variables` view
exposes). The interactive REPL, which already had a documented
exception for `eval`, keeps its arbitrary-expression power. Two
surfaces, two safety profiles, both documented.

### Verification gate

```
ruff check       — clean
ruff format     — 169 + 2 reformatted = 171 files
mypy --strict src/harness  — clean (82 source files)
pytest          — 494 passed
```

### Commits

```
2f40af6  chore: Wave 7 pre-step — DAP framing + message models
a00e3da  feat(debug): DapAdapter + harness debug --dap stdio mode
0183f08  docs: progress.md log of Wave 7
f70fadf  test(dap): drop overly broad warnings.warn monkeypatch
```
