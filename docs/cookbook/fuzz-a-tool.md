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

<!-- reason: shell example, not executed in the codeblock gate -->
<!--pytest.mark.skip-->
```bash
uv add 'harness-engineering-toolkit[fuzz]'
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
        dispatcher,
        "parse",
        n=50,         # 50 examples
        seed=0,       # deterministic; same seed → same examples
    )
)

print(f"{report.passed}/{report.total} passed")
for failure in report.failures:
    print(
        f"  input={failure.input!r}  "
        f"result={failure.result!r}  "
        f"exc={failure.exception!r}"
    )
```

A typical run will surface `count=0` as a divide-by-zero (Hypothesis
hits the boundary deterministically). The report contains structured
`FuzzFailure`s — input dict, the tool result (if any), the exception
(if any).

## Agent-level invariant fuzzing

For checking properties of *the assistant's response* given fuzzed
inputs, give `fuzz_agent` the tool whose `input_model` to fuzz and
an invariant over the assistant `Message`:

<!-- reason: illustrative; references undefined orchestrator / agent and uses `await` at module scope -->
<!--pytest.mark.skip-->
```python
from harness.fuzz import fuzz_agent
from harness.prompts.messages import Message

def invariant(reply: Message) -> bool:
    """The agent's reply must mention the city we asked about."""
    return any(
        block.type == "text" and "Berlin" in (block.text or "")
        for block in reply.content
    )

report = await fuzz_agent(
    orchestrator,
    agent,
    "weather",           # tool_name registered on the dispatcher
    n=20,
    invariant=invariant,
    seed=0,
)
```

The user message embeds the generated `tool.input_model` payload by
default; pass `prompt_template=lambda example: ...` to customise.

## pytest integration

Use the `harness_property` decorator for property-based tests. It
takes the dispatcher and the tool *name* (not the input model):

<!-- reason: illustrative; references undefined dispatcher and decorates at module scope -->
<!--pytest.mark.skip-->
```python
from harness.fuzz import harness_property
from harness.tools.schema import ToolCall

@harness_property(dispatcher=dispatcher, tool="parse", n=100, seed=0)
async def parse_never_errors(payload):
    result = await dispatcher.dispatch(
        ToolCall(name="parse", arguments=payload.model_dump())
    )
    assert result.is_error is False, f"errored on {payload}"
```

Hypothesis settings ship sensible defaults: `derandomize=True`,
`database=None` (so CI runs are reproducible without a shared
example DB), no deadline (long handlers don't trip the default
200 ms limit).

## Gotchas

- **The strategy bridge is small.** It covers `str`, `int`, `float`,
  `bool`, and `Optional[X]` / `X | None`, plus the
  `annotated_types` constraints from `Field` (`min_length`,
  `max_length`, `ge`, `le`). Anything else — `list[X]`, `dict`,
  nested models, `Literal`, `Decimal`, `datetime` — raises
  `FuzzStrategyUnsupported`. Use `overrides={"field": strategy}`
  to plug a hand-built `hypothesis.strategies` strategy.
- **Determinism is per-seed**, not per-test-run. `seed=0` always
  generates the same sequence; if you want different examples,
  bump the seed.
- **Failures are not shrunk.** Hypothesis only runs the
  `generate` phase here, so `failure.input` is the exact input
  the generator produced — useful for surfacing edge cases, but
  not minimised.
- **`fuzz_agent` runs the orchestrator's runner per example.** Give
  the orchestrator a `CannedRunner` or `EchoRunner` for routine CI
  runs; gate real-model runs behind an explicit flag.

## Related

- [`harness.fuzz`](../modules/fuzz.md) — module reference.
- [`examples/fuzz.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/fuzz.py)
  — runnable end-to-end demo.
- [Cookbook: Replay evaluation](replay-evaluation.md) — once you
  find a failing input, replay it as a regression test.
