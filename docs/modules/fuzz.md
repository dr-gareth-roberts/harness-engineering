# `harness.fuzz`

Hypothesis-driven tool and agent fuzzing (extra `[fuzz]`).
`fuzz_tool` drives Pydantic-typed inputs through `Dispatcher.dispatch`
and collects failures; `fuzz_agent` does the same through a full
`Orchestrator` turn. The `harness_property` pytest decorator wires
generated inputs into a property-based test.

## When to reach for this

- You wrote a tool with a Pydantic input model and want to find
  inputs that crash it before the model does.
- You want a property test ("the agent's reply must mention the
  asked city") evaluated across many generated inputs.
- You want deterministic fuzz runs in CI (same seed → same
  examples) without an external example database.

## Quick example

```bash
uv add 'harness-engineering[fuzz]'
```

```python
import asyncio
from harness.fuzz import fuzz_tool

report = asyncio.run(fuzz_tool(
    dispatcher=dispatcher,
    tool_name="parse",
    n=100,
    seed=0,
))

print(f"{report.passed}/{report.total} passed")
for failure in report.failures:
    print(f"  {failure.input!r} → {failure.exception or failure.result}")
```

pytest property test:

```python
from harness.fuzz import harness_property

@harness_property(input_model=ParseIn, n=100, seed=0)
async def test_parse_never_raises(args):
    result = await dispatcher.dispatch(...)
    assert result.is_error is False
```

## Gotchas

- **Hypothesis settings are pinned for reproducibility:**
  `derandomize=True`, `database=None`, no deadline. CI runs are
  stable across machines.
- **Strategy bridge handles primitives + `Optional` / `Literal` /
  `list[X]` / basic `dict`.** Exotic types (custom validators,
  `RootModel` wrappers) need an `overrides=` mapping with a
  hand-built strategy.
- **Failures are shrunk** by Hypothesis. `failure.input` is the
  minimum input that reproduces; the original generated input
  isn't preserved.
- **`fuzz_agent` runs real model calls** by default. Gate behind a
  CI flag and use `CannedRunner` / `EchoRunner` for routine fuzz.

## Related

- [Cookbook: Fuzz a tool](../cookbook/fuzz-a-tool.md) — extended walkthrough.
- [`examples/fuzz.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/fuzz.py)
- [`harness.tools`](tools.md) — the dispatcher under test.

## API reference

::: harness.fuzz
