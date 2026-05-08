# `harness.policy`

`AllowList`, `DenyList`, and `ArgumentMatcher` policies for tool
calls. Attach as `PreToolUse` hooks; a blocked policy short-circuits
the dispatcher with a `ToolResult(is_error=True)`.

::: harness.policy
