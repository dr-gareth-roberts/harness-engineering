# `harness.sandbox`

`PathScope` + `PathPolicy` for filesystem-scoped tool calls and
`safe_subprocess_run` with a scrubbed environment and timeout.
Keeps sandbox concerns out of the tool handlers themselves — your
handler asks the scope to resolve a path and gets a typed error if
it's outside the allow set.

## When to reach for this

- You wrote a tool that reads/writes files and you want to constrain
  it to a directory.
- You wrote a tool that shells out and you want a clean environment
  + timeout without re-deriving them per handler.
- You want sandbox failures to surface as `ToolResult(is_error=True)`,
  not crash the handler.

## Quick example

```python
from harness import PathPolicy, PathScope, safe_subprocess_run, scrub_env

# 1. Scope file access to a workspace dir.
scope = PathScope(roots=["./workspace"], policy=PathPolicy.READ_WRITE)

def read_file(args):
    p = scope.resolve(args.path)  # raises if outside roots
    return p.read_text()

# 2. Shell out safely.
result = safe_subprocess_run(
    ["uv", "run", "pytest", "-q"],
    cwd="./workspace",
    timeout=30,
    env=scrub_env(),  # only the variables you allowlist
)
```

## Gotchas

- **`PathScope.resolve` is the only safe path method** — direct
  `Path(...)` construction inside a handler bypasses the scope.
  Always call `scope.resolve(user_supplied)` first.
- **`safe_subprocess_run` is sync.** Wrap with
  `asyncio.to_thread(...)` if you're calling from an async context
  and the subprocess is long-running.
- **`scrub_env` is allowlist-based.** Pass `keep=` with the env
  vars your subprocess actually needs (e.g., `PATH`); everything
  else is dropped.
- **`PathPolicy.READ_ONLY` + `scope.resolve(write_path)`** raises;
  the scope decides whether the operation is allowed before the
  Path object is exposed.

## Related

- [`harness.tools`](tools.md) — handlers that use the sandbox primitives.
- `examples/` doesn't have a dedicated sandbox demo today; the
  primitives are simple enough that the docstring + this page
  cover them.

## API reference

::: harness.sandbox
