# `harness.sandbox`

`PathScope` for resolving caller-supplied paths against an allow /
deny set, `PathPolicy` for blocking `PreToolUse` events whose path
arguments fall outside scope, and `safe_subprocess_run` for shelling
out with a scrubbed env and a wall-clock timeout. Sandbox concerns
stay out of the tool handlers — they ask the scope to validate a
path and get either a real `Path` back or a typed error.

## When to reach for this

- You wrote a tool that reads/writes files and you want to constrain
  it to a directory.
- You wrote a tool that shells out and you want a clean environment
  + timeout without re-deriving them per handler.
- You want sandbox failures to surface as `ToolResult(is_error=True)`,
  not crash the handler.

## Quick example

```python
from harness import (
    HookRunner, PathPolicy, PathScope, PreToolUse,
    safe_subprocess_run, scrub_env,
)

# 1. Scope file access to a workspace dir.
scope = PathScope.of(allow=["./workspace"])

def read_file(args):
    # validate() resolves the path symlink-aware and raises
    # PathDenied if it falls outside the allow set.
    p = scope.validate(args.path)
    return p.read_text()

# 2. (Optional) wire the same scope into a PreToolUse hook handler
#    that blocks calls to listed tools when their `path` argument is
#    outside the scope. Catches escapes before the handler runs.
hooks = HookRunner()
hooks.register(
    PreToolUse,
    PathPolicy.of(scope, tool_names={"read_file", "write_file"}),
)

# 3. Shell out safely. `safe_subprocess_run` is async.
result = await safe_subprocess_run(
    ["uv", "run", "pytest", "-q"],
    cwd="./workspace",
    timeout=30,
    env=scrub_env(),  # only the variables you allowlist
)
print(result.returncode, result.duration_ms)
```

## Gotchas

- **`PathScope.validate` is the gate** — direct `Path(...)`
  construction inside a handler bypasses the scope. Always call
  `scope.validate(user_supplied)` first. `is_allowed(path)` is the
  predicate variant when you want to branch rather than raise.
- **`safe_subprocess_run` is async** — `await` it; don't expect a
  sync return. The child is spawned via `create_subprocess_exec`,
  so there's no shell layer and `cmd` quoting is not a risk.
- **`scrub_env` is allowlist-based.** The kwarg is `allow_keys=`
  (iterable of env var names). The default is `DEFAULT_ALLOWED_ENV_KEYS`
  — `PATH`, `HOME`, `TMPDIR`, `TMP`, `TEMP`, `LANG`, `LC_ALL`.
  Everything else (`ANTHROPIC_API_KEY`, `AWS_*`, `GITHUB_TOKEN`, …)
  is dropped before reaching the child.
- **`PathScope` is advisory.** Between `is_allowed()` returning
  True and the caller opening the path, a concurrent symlink swap
  (TOCTOU) can redirect. Use OS-level isolation if real safety
  matters.

## Related

- [`harness.tools`](tools.md) — handlers that use the sandbox primitives.
- `examples/` doesn't have a dedicated sandbox demo today; the
  primitives are simple enough that the docstring + this page
  cover them.

## API reference

::: harness.sandbox
