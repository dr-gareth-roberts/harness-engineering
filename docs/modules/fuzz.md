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

<!-- reason: shell example, not executed in the codeblock gate -->
<!--pytest.mark.skip-->
```bash
uv add 'harness-engineering-toolkit[fuzz]'
```

<!-- reason: illustrative; references undefined dispatcher and needs the [fuzz] extra -->
<!--pytest.mark.skip-->
```python
import asyncio
from harness.fuzz import fuzz_tool

report = asyncio.run(fuzz_tool(
    dispatcher,
    "parse",
    n=100,
    seed=0,
))

passed = report.total - len(report.failures)
print(f"{passed}/{report.total} passed")
for failure in report.failures:
    print(f"  {failure.input!r} -> {failure.exception or failure.result}")
```

pytest property test:

<!-- reason: illustrative; decorates at module scope with an undefined dispatcher -->
<!--pytest.mark.skip-->
```python
from harness.fuzz import harness_property
from harness.tools.schema import ToolCall

@harness_property(dispatcher=dispatcher, tool="parse", n=100, seed=0)
async def parse_never_errors(payload):
    result = await dispatcher.dispatch(
        ToolCall(name="parse", arguments=payload.model_dump())
    )
    assert result.is_error is False
```

## Gotchas

- **Strategy bridge handles primitives + `Optional` only.** The
  supported types are `str`, `int`, `float`, `bool`, and
  `Optional[X]` / `X | None`, plus the `annotated_types`
  constraints attached by `Field` (`min_length`, `max_length`,
  `ge`, `le`). Anything else — `list[X]`, `dict`, nested models,
  `Literal`, `Decimal`, `datetime`, unions of two non-None types —
  raises `FuzzStrategyUnsupported`. Pass `overrides=` with a
  hand-built `hypothesis.strategies` strategy per such field.
- **Hypothesis is an optional dependency.** Importing
  `harness.fuzz` always works; the first call into `fuzz_tool` /
  `fuzz_agent` / `pydantic_strategy` raises a structured
  `ImportError` if `[fuzz]` isn't installed. Under pytest,
  `harness_property` calls `pytest.skip` so missing-extra runs
  surface cleanly.
- **`fuzz_agent` runs real model calls** if you give it a real
  runner. Gate behind a CI flag, or pass `CannedRunner` /
  `EchoRunner` to the orchestrator for routine fuzz.
- **`harness_property` consumes the dispatcher at decoration time.**
  Pass the dispatcher and tool *name*; the wrapped function
  receives a validated input-model instance per generated example
  and is expected to assert whatever matters.

## Related

- [Cookbook: Fuzz a tool](../cookbook/fuzz-a-tool.md) — extended walkthrough.
- [`examples/fuzz.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/fuzz.py)
- [`harness.tools`](tools.md) — the dispatcher under test.

## API reference

::: harness.fuzz
