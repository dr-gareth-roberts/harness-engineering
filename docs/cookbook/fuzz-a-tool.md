# Fuzz a tool with Hypothesis

## Problem

You wrote a tool that takes a Pydantic input model. You want to
catch the inputs your handler doesn't gracefully handle — the
`""` empty string, the `0` boundary, the negative number, the
unicode you didn't think about — *before* a model finds them at
runtime.

## Solution sketch

`harness.fuzz` bridges Pydantic to Hypothesis. `fuzz_tool` walks
your tool's `input_model.model_fields`, derives a Hypothesis strategy
per field, generates `n` examples deterministically (seedable), runs
each through `Dispatcher.dispatch`, and reports the inputs that
either raised or returned `is_error=True`.

`fuzz_agent` is the orchestrator-level version: it drives generated
inputs through a full agent turn and tests an invariant on the
resulting assistant message.

Install the extra:

```bash
uv add 'harness-engineering[fuzz]'
```

## Working code

```python
import asyncio

from pydantic import BaseModel, Field

from harness import Dispatcher, Tool
from harness.fuzz import fuzz_tool


class ParseIn(BaseModel):
    raw: str = Field(min_length=1)
    count: int = Field(ge=0, le=100)


def parse(args: ParseIn) -> str:
    # Assume this divides by count somewhere — we want to find the bug
    # where count=0 sneaks through despite ge=0.
    return args.raw * (10 // args.count)


dispatcher = Dispatcher(
    [Tool(name="parse", description="Parse stuff.", input_model=ParseIn, handler=parse)]
)

report = asyncio.run(
    fuzz_tool(
        dispatcher=dispatcher,
        tool_name="parse",
        n=50,         # 50 examples
        seed=0,       # deterministic; same seed → same examples
    )
)

print(f"{report.passed}/{report.total} passed")
for failure in report.failures:
    print(f"  input={failure.input!r}  result={failure.result!r}  exc={failure.exception!r}")
```

A typical run will surface `count=0` as a divide-by-zero (Hypothesis
hits the boundary deterministically). The report contains structured
`FuzzFailure`s — input dict, the tool result (if any), the exception
(if any).

## Agent-level invariant fuzzing

For checking properties of *the assistant's response* given fuzzed
inputs:

```python
from harness.fuzz import fuzz_agent
from harness.prompts import Message

async def invariant(reply: Message) -> bool:
    """The agent's reply must mention the city we asked about."""
    return any("Berlin" in (b.text or "") for b in reply.content)

await fuzz_agent(
    orchestrator=orchestrator,
    agent=agent,
    input_model=WeatherQuery,
    invariant=invariant,
    n=20,
)
```

## pytest integration

Use the `harness_property` decorator for property-based tests:

```python
from harness.fuzz import harness_property

@harness_property(input_model=ParseIn, n=100, seed=0)
async def test_parse_never_raises(args):
    result = await dispatcher.dispatch(ToolCall(name="parse", arguments=args.model_dump()))
    assert result.is_error is False, f"raised on {args}"
```

Hypothesis settings ship sensible defaults: `derandomize=True`,
`database=None` (so CI runs are reproducible without a shared
example DB), no deadline (long handlers don't trip the default 200ms
limit).

## Gotchas

- **Hypothesis strategies for complex types** — the bridge handles
  primitive types, `Optional`, `Literal`, `list[X]`, basic `dict`.
  For exotic types (custom validators, `RootModel` wrappers, etc.),
  pass `overrides={"field_name": custom_strategy}` to inject a
  hand-built Hypothesis strategy.
- **Determinism is per-seed**, not per-test-run. `seed=0` always
  generates the same sequence; if you want different examples,
  bump the seed.
- **Failures may be reduced** — Hypothesis automatically shrinks
  failing inputs to their minimum form. `failure.input` is the
  shrunk version, not the original generated one.
- **`fuzz_agent` runs real model calls** unless your `runner` is a
  fake. For CI, gate behind a `--fuzz-real-runner` flag and use
  `CannedRunner` / `EchoRunner` by default.

## Related

- [`harness.fuzz`](../modules/fuzz.md) — module reference.
- [`examples/fuzz.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/fuzz.py)
  — runnable end-to-end demo.
- [Cookbook: Replay evaluation](replay-evaluation.md) — once you
  find a failing input, replay it as a regression test.
