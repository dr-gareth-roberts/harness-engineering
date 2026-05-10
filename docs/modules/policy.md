# `harness.policy`

`AllowList`, `DenyList`, and `ArgumentMatcher` policies for tool
calls. Attach as `PreToolUse` hooks; a blocked policy short-circuits
the dispatcher with a `ToolResult(is_error=True)`.

## When to reach for this

- You want to constrain which tools an agent can call (allow/deny by
  name).
- You want to gate tools on argument shape (`delete` allowed only
  when `confirm=True`).
- You want a declarative, hook-based gate rather than an `if`
  ladder inside every handler.

## Quick example

```python
from harness import AllowList, DenyList, HookRunner
from harness.policy import ArgumentMatcher, attach_pre_tool_policies

hooks = HookRunner()

# Allow only specific tools.
attach_pre_tool_policies(hooks, AllowList.of({"search", "summarize"}))

# Deny destructive tools entirely.
attach_pre_tool_policies(hooks, DenyList.of({"delete_user", "drop_table"}))

# Allow `delete` only with confirm=True.
attach_pre_tool_policies(
    hooks,
    ArgumentMatcher(tool="delete", required={"confirm": True}),
)
```

## Gotchas

- **`SubAgent.allowed_tools` is the *runner-level* gate**: only those
  tools' schemas are sent to the model. Policies are *runtime*
  gates: even if the model could call a tool (because it's in
  `allowed_tools` and the schema went up), policies can still
  block.
- **Block reason surfaces to the model.** It's part of the error
  ToolResult. Don't put secrets in your reason string.
- **First matching block wins.** Multiple policies attached as hooks
  are consulted in registration order; the first that returns
  `HookDecision(block=True)` short-circuits.

## Related

- [`harness.hooks`](hooks.md) — the underlying hook protocol.
- [Cookbook: Redact PII](../cookbook/redact-pii.md) — privacy boundary uses similar gating.

## API reference

::: harness.policy
