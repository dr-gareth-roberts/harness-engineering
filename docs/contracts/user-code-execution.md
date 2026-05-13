# User-code execution discipline

This page is the contract for what happens when a caller-supplied
callable raises inside the harness. The four surfaces below behave
differently *on purpose*; this document is the canonical reference for
that asymmetry. If you're writing a handler, sink, or predictor and
want to know "what happens if my code crashes," you're in the right
place.

## Map of surfaces

| Surface | On `raise` | Visibility |
|---|---|---|
| `harness.hooks.HookRunner.emit` | **Propagates** | Aborts the turn |
| `harness.tools.Dispatcher.dispatch` | **Converts to `ToolResult(is_error=True)`** | Model sees it |
| `harness.telemetry.MultiSink.emit` | **Isolates per-sink** | Logged at WARNING |
| `harness.speculate.Speculator._dispatch_via_hooks` | **Converts to `ToolResult(is_error=True)`** | Visible via `try_resolve` hit |

### `harness.hooks.HookRunner.emit` — propagates

A hook handler that raises an exception aborts the turn. The exception
travels up through the orchestrator and is the caller's problem. There
is no per-handler `try/except` wrapping. If you want a hook to fail
soft, catch inside your handler and return a sentinel `HookDecision`
(or `None` to opt out of the decision).

### `harness.tools.Dispatcher.dispatch` — converts

A tool handler that raises produces
`ToolResult(id=call.id, content=str(exc), is_error=True)`. The model
sees the error string in its tool-result block and can self-correct on
the next turn. Validation errors from the input model are converted
the same way. The dispatcher itself never raises from a handler bug.

### `harness.telemetry.MultiSink.emit` — isolates

An exception inside one sink's `emit` is caught and logged at
`WARNING` with `exc_info=True` (logger name
`harness.telemetry.sinks`). The remaining sinks in the fan-out still
run. Telemetry is best-effort observation: a misbehaving sink must
never break the operation it's observing. Individual sinks
(`JSONLSink`, `MemorySink`, `OpenTelemetrySink`) do *not* wrap their
own `emit` — only `MultiSink` provides the isolation. If you wire a
single sink directly into `Telemetry`, its exceptions will surface.

### `harness.speculate.Speculator._dispatch_via_hooks` — converts

The speculator's internal dispatch path wraps the
`PreToolUse → dispatch → PostToolUse` cycle in `try/except`. A hook
handler that raises during a speculative dispatch becomes
`ToolResult(id=call.id, content=f"speculation error: {exc!r}", is_error=True)`.
On a `try_resolve` hit, that error result flows back to the model
exactly as if the tool itself had failed. The dispatcher layer below
this point already converts handler exceptions to `is_error=True`, so
the extra wrap exists specifically to contain *hook-handler*
exceptions in the speculative path — a buggy hook must not crash the
runner just because a prediction happened to fire it.

## Rationale for the asymmetry

- **Hook handlers are policy.** A policy that crashes means the
  operator's intent is undefined; safer to abort the turn than to
  silently swallow the failure and continue with ambiguous
  enforcement.
- **Tool handlers are observable behavior.** The model's job already
  includes recovering from tool failures (network errors, bad inputs,
  rate limits). An exception is just another observable outcome — the
  model sees `is_error=True` and adapts.
- **Telemetry is best-effort observation.** A sink that breaks the
  operation it's observing has inverted the relationship. Fan-out
  isolation is the only correct behavior.
- **Speculation is opportunistic optimization.** A failed speculation
  must not crash the real path. The wrong prediction is supposed to
  be cheap; a crash makes it expensive.

## What to do

- **You want a hook to fail soft:** catch inside your handler and
  return a sentinel `HookDecision` (e.g. `block=False`), or `None` to
  opt out. The runner won't see the exception.
- **You want a tool exception to propagate:** that's not the contract.
  Wrap the failure case in your own exception type, surface it via
  `ToolResult.content`, and inspect from your caller's side. The
  model will still see `is_error=True`.
- **You want sink failures to be visible:** wrap `MultiSink` with
  your own audit sink that watches the logger, or write a sink that
  raises into a queue your supervisor drains. The default behavior
  is a `WARNING` log line — silent only if your logging is silent.
- **You want a speculator exception to crash:** also not the
  contract. If you need that signal, instrument with telemetry on
  `SpeculationMiss` and inspect — speculation is allowed to fail
  invisibly to the runner by design.

## Future direction

The asymmetry is intentional today. A unified
`user_code_execution_policy` flag — letting an operator pick "abort,"
"convert," or "isolate" per surface — is a backlog item (see
`audit/RELEASE-TODO.md` M2.4 follow-up) but requires a deeper design
decision about how to express the policy without forcing every
caller-supplied callable through the same shape. Until then, the
table at the top of this page is the contract.
