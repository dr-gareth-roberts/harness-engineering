# harness-engineering

Reusable building blocks for the layer around an LLM — the
"harness" that turns a model into an agent. Eighteen modules covering
tool dispatch, prompts, hooks, policies, runners, memory, replay,
behavioral contracts, privacy, plans, fuzzing, causal attribution,
prefix-cache drift, an interactive debugger, and speculative tool
execution.

## Install

```bash
uv add harness-engineering
# or
pip install harness-engineering
```

Optional extras pull in heavier dependencies on demand:

| Extra | Adds |
|---|---|
| `[anthropic]` | `AnthropicRunner` (Anthropic Messages API tool-use loop). |
| `[openai-compat]` | `OpenAICompatRunner` (OpenAI / vLLM / Ollama / llama.cpp / LM Studio). |
| `[otel]` | `OpenTelemetrySink` for emitting telemetry events to an OTel collector. |
| `[fuzz]` | Hypothesis-based tool/agent fuzzers. |
| `[attribute]` | `EmbeddingSimilarity` (sentence-transformers) for causal attribution. |
| `[docs]` | MkDocs + mkdocs-material + mkdocstrings to build this site locally. |

## Why "harness"?

A model alone doesn't do useful work — it produces tokens. The
*harness* is everything around the model that turns those tokens into
behavior: the tools the model can call, the prompts it sees, the
policies that constrain its tool use, the memory that persists between
turns, the replay layer that lets you re-run a trajectory
deterministically, and the observability stack that tells you what
happened.

This package gives you those primitives. They're decoupled — each
module satisfies a clear protocol, so you can plug in your own runner,
your own memory store, your own privacy detector, etc., without
touching the rest. The defaults are batteries-included; the seams are
small.

## A 30-second tour

```python
import asyncio
from harness import (
    Dispatcher, Tool, HookRunner, Orchestrator,
    SubAgent, EchoRunner,
)
from pydantic import BaseModel

class GreetIn(BaseModel):
    name: str

def greet(args: GreetIn) -> str:
    return f"Hello, {args.name}!"

dispatcher = Dispatcher([
    Tool(name="greet", description="Say hello.", input_model=GreetIn, handler=greet),
])
runner = EchoRunner()
orchestrator = Orchestrator(dispatcher, HookRunner(), runner)

agent = SubAgent(name="demo", system_prompt="", model="echo")
asyncio.run(orchestrator.run(agent, [...]))
```

For runnable end-to-end examples per module, see the
[`examples/`](https://github.com/dr-gareth-roberts/harness-engineering/tree/main/examples)
directory. Each example is also a smoke-tested entry in CI.

## Where to read next

- [**Architecture**](architecture.md) — how the modules fit together,
  and which seams are designed to swap out.
- [**CLI**](cli.md) — the `harness` subcommands (`debug`, `cache-audit`,
  `--dap`).
- [**Modules**](modules/tools.md) — per-module reference (auto-generated
  from docstrings).
- [**Roadmap**](roadmap.md) — what's been shipped and what's next.

## Building the docs locally

```bash
uv sync --extra docs
uv run mkdocs serve
```

Browse to `http://localhost:8000`. Hot-reload on file save.
