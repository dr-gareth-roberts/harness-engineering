# harness-engineering

Reusable building blocks for the layer around an LLM — the
"harness" that turns a model into an agent. Nineteen modules covering
tool dispatch, prompts, hooks, policies, runners, memory, replay,
behavioral contracts, privacy, plans, fuzzing, causal attribution,
prefix-cache drift, an interactive debugger, speculative tool
execution, and event-streaming output.

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
| `[privacy-ml]` | `PresidioDetector` (Microsoft Presidio NLP-backed PII detection). |
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

from pydantic import BaseModel

from harness import (
    CannedRunner, Dispatcher, HookRunner, Orchestrator,
    SubAgent, Tool, text,
)


class GreetIn(BaseModel):
    name: str


def greet(args: GreetIn) -> str:
    return f"Hello, {args.name}!"


dispatcher = Dispatcher(
    [Tool(name="greet", description="Say hello.", input_model=GreetIn, handler=greet)]
)
runner = CannedRunner(replies=["Hello back!"])  # no API key needed
orchestrator = Orchestrator(dispatcher, HookRunner(), runner)
agent = SubAgent(name="demo", system_prompt="", model="canned")

reply = asyncio.run(orchestrator.run(agent, [text("user", "hi")]))
print(reply.content[0].text)
```

That's a runnable program. The [Quickstart](quickstart.md) walks
through it line by line, then upgrades the runner to real Anthropic.

For runnable end-to-end examples per module, see the
[`examples/`](https://github.com/dr-gareth-roberts/harness-engineering/tree/main/examples)
directory. Each example is also a smoke-tested entry in CI.

## Where to read next

If you're evaluating the library:

- [**Quickstart**](quickstart.md) — 10 minutes to a working agent
  (`CannedRunner` first, real Anthropic second).
- [**Comparison**](comparison.md) — where harness sits relative to
  LangChain / DSPy / AutoGen / CrewAI. Honest placement, not a
  takedown.
- [**Cookbook**](cookbook/index.md) — concrete recipes: redact PII,
  replay a session, debug a bad trajectory, fuzz a tool, cache +
  speculate, observability with OpenTelemetry.

If you're building on it:

- [**Architecture**](architecture.md) — the protocol seams (`Runner`,
  `Sink`, `MemoryStore`) and the composition pattern.
- [**CLI**](cli.md) — `harness debug`, `harness debug --dap`,
  `harness cache-audit`.
- [**Modules**](modules/tools.md) — per-module reference, with
  use-cases / examples / gotchas / API ref for each.
- [**FAQ**](faq.md) — common pitfalls and "why does X behave that way?"
- [**Roadmap**](roadmap.md) — what's shipped, what's deferred.

## Building the docs locally

```bash
uv sync --extra docs
uv run mkdocs serve
```

Browse to `http://localhost:8000`. Hot-reload on file save.
