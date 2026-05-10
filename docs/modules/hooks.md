# `harness.hooks`

Typed lifecycle events and an ordered `HookRunner` with `block`-aware
short-circuit semantics. The structural seam used by policy,
contracts, telemetry, the privacy boundary, and your own observers.

## When to reach for this

- You want to observe everything that happens during a session
  (start, every prompt, every tool call, every assistant message).
- You want to *gate* tool calls on a runtime predicate (block, or
  replace with a synthetic result).
- You want to fan out the same lifecycle to multiple consumers
  (policy + telemetry + contracts) without coupling them.

## Quick example

```python
from harness import HookRunner, PostToolUse, PreToolUse, SessionEnd
from harness.hooks import HookDecision

hooks = HookRunner()

# Observe — print every tool call.
hooks.register(PostToolUse, lambda e: print(f"{e.call.name} → {e.result.content!r}"))

# Block — refuse calls with empty arguments.
def gate(event: PreToolUse) -> HookDecision | None:
    if not event.call.arguments:
        return HookDecision(block=True, reason="empty arguments")
    return None
hooks.register(PreToolUse, gate)

# Cleanup — async handlers also work.
async def on_end(_event: SessionEnd) -> None:
    await flush_metrics()
hooks.register(SessionEnd, on_end)
```

## Gotchas

- **Hook handlers can be sync or async.** Detected via
  `inspect.isawaitable` on the return. Don't mix the two return
  styles in the same handler.
- **`HookDecision.block=True`** short-circuits the dispatcher with
  `ToolResult(is_error=True, content=reason)`. The model sees the
  block and can react.
- **`HookDecision.replacement=ToolResult(...)`** (Wave 10 #5)
  replaces the dispatched result without running the handler
  (PreToolUse) or rewrites it after dispatch (PostToolUse). First
  matching hook wins; later hooks see the replaced result.
- **`PostAssistantMessage` is observational** — by the time it
  fires, the model already produced the message. Use `PreToolUse`
  for blocking; `PostAssistantMessage` for inspection / contracts /
  telemetry.

## Related

- [`harness.policy`](policy.md) — pre-built `PreToolUse` policies (allow/deny/argument match).
- [`harness.contracts`](contracts.md) — declarative invariants that attach as hooks.
- [`harness.privacy`](privacy.md) — the boundary fires hooks for every detection.

## API reference

::: harness.hooks
