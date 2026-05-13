# `harness.tools`

The model-callable surface. `Tool` wraps a Pydantic input schema +
a handler; `Dispatcher` validates inputs, executes handlers, and
surfaces any exception as `ToolResult(is_error=True)` rather than
raising.

## When to reach for this

- You want the model to call your code and receive structured input.
- You need input validation between the model and your handler.
- You want a *uniform* failure shape (`ToolResult(is_error=True)`) so
  the model can react to errors without your handler having to
  hand-craft error JSON.
- You want telemetry per tool dispatch (duration, error rate, args).

## Quick example

```python
from pydantic import BaseModel
from harness import Dispatcher, Tool

class SearchIn(BaseModel):
    query: str
    limit: int = 10

async def search(args: SearchIn) -> list[str]:
    return [f"result-{i} for {args.query}" for i in range(args.limit)]

dispatcher = Dispatcher([
    Tool(name="search", description="Search records.",
         input_model=SearchIn, handler=search),
])
```

The `tools_schema()` method returns the JSON schemas runners send to
their model providers. You don't write per-vendor schemas.

## Gotchas

- **Handler exceptions become `ToolResult(is_error=True)`**, not
  raised. The dispatcher swallows them so the loop continues; the
  model sees the error and can recover. If you want raises to
  propagate, that's not the contract today.
- **`Tool.idempotent=True` is a promise.** It tells the speculator
  this tool can be safely re-executed. Mark it only when re-running
  with the same args is observably equivalent to running once.
- **The handler's input type must be a Pydantic `BaseModel`** —
  primitives or dataclasses don't validate via `model_validate`.
- **Async vs sync handlers both work**, detected via
  `inspect.isawaitable` on the return.

## Related

- [`examples/end_to_end.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/end_to_end.py) — runnable end-to-end demo.
- [Cookbook: Fuzz a tool](../cookbook/fuzz-a-tool.md) — Hypothesis-driven fuzz testing for tool handlers.
- [`harness.policy`](policy.md) — gating tool calls before dispatch.

## API reference

::: harness.tools
