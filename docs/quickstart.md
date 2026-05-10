# Quickstart

Goal: a working agent in 10 minutes, with no API key required for the
first half. By the end you'll have:

1. A tool the model can call,
2. A hook that observes every call,
3. An orchestrator running both — first against a deterministic
   `EchoRunner`, then (optionally) against the real Anthropic API.

If you finish step 1, you've already built something runnable.

## Install

```bash
uv add harness-engineering
# or
pip install harness-engineering
```

That gives you the core (Pydantic-only). Vendor SDKs and heavy
dependencies are opt-in extras — you don't need any of them yet.

## Step 1 — A tool, an agent, a runner (no API key)

Save this as `tour.py`:

```python
import asyncio

from pydantic import BaseModel

from harness import (
    CannedRunner,
    Dispatcher,
    HookRunner,
    Message,
    Orchestrator,
    PostToolUse,
    SubAgent,
    Tool,
    text,
)


# 1. Declare a tool. Pydantic gives you input validation for free.
class WeatherIn(BaseModel):
    city: str


def weather(args: WeatherIn) -> str:
    """Pretend we hit a weather API. The real handler is your own code."""
    return f"It is 22°C and sunny in {args.city}."


# 2. Wire the tool into a Dispatcher.
dispatcher = Dispatcher(
    [
        Tool(
            name="weather",
            description="Get the current weather for a city.",
            input_model=WeatherIn,
            handler=weather,
        ),
    ]
)

# 3. Hook every tool call so we can observe what the agent did.
hooks = HookRunner()
hooks.register(PostToolUse, lambda e: print(f"  -> {e.call.name}({e.call.arguments}) = {e.result.content}"))

# 4. A SubAgent describes the role. The runner is what actually
#    talks to the model. CannedRunner returns a scripted reply,
#    so we can demo without an API key.
runner = CannedRunner(replies=["The weather in Berlin is 22°C and sunny."])
orchestrator = Orchestrator(dispatcher, hooks, runner)
agent = SubAgent(name="weather-bot", system_prompt="", model="canned", allowed_tools=["weather"])


async def main() -> None:
    reply = await orchestrator.run(
        agent,
        messages=[text("user", "What's the weather in Berlin?")],
    )
    print(reply.content[0].text)


asyncio.run(main())
```

Run it:

```bash
uv run python tour.py
```

You'll see the canned reply print. The CannedRunner is deterministic
— good for tests, demos, replay. There's nothing magic happening.

## Step 2 — Replace `CannedRunner` with the real model

Add the optional Anthropic extra:

```bash
uv add 'harness-engineering[anthropic]'
export ANTHROPIC_API_KEY=...
```

Swap the runner:

```python
from harness import AnthropicRunner

# replaces `runner = CannedRunner(...)`
runner = AnthropicRunner(dispatcher, hooks)
agent = SubAgent(
    name="weather-bot",
    system_prompt="You are a concise weather assistant.",
    model="claude-opus-4-7",
    allowed_tools=["weather"],
)
```

That's the only change. Run again, and the model now decides whether
to call `weather` itself. Your `PostToolUse` hook fires for every
real call. The orchestrator + dispatcher + hooks are model-agnostic
— `AnthropicRunner` translates your `Tool` definitions into Anthropic's
tool-use schema, runs the loop, and returns the final assistant
`Message`. You don't write any of that boilerplate.

OpenAI works the same way:

```bash
uv add 'harness-engineering[openai-compat]'
```

```python
from harness import OpenAICompatRunner
runner = OpenAICompatRunner(dispatcher, hooks)  # OpenAI default
# or point at a local server:
# runner = OpenAICompatRunner(dispatcher, hooks, base_url="http://localhost:11434/v1")
```

`OpenAICompatRunner` works against any OpenAI-compatible endpoint:
OpenAI itself, vLLM, Ollama, llama.cpp's server, LM Studio, Together,
Groq.

## Step 3 — What you got for free

The runner you just swapped is one seam. Here's what's already wired:

- **Replay** (`harness.replay`): record any session, deterministically
  replay it later, run differential evaluation across providers
  (Anthropic vs OpenAI vs cached) with HTML reports.
- **Debug** (`harness.debug`): wrap any runner in `DebugRunner` to
  pause mid-trajectory, inspect state, fire ad-hoc tool calls,
  mutate the next turn, resume. Same surface drives an interactive
  REPL or a [DAP server](cli.md#harness-debug) for VS Code /
  neovim-dap.
- **Privacy boundary** (`harness.privacy`): wrap a runner with a
  `PrivacyBoundary` to scrub PII / secrets in both directions.
  Regex + entropy detectors ship; Presidio adapter under
  `[privacy-ml]` adds NLP-backed recognizers.
- **Behavioral contracts** (`harness.contracts`): declarative
  invariants over agent trajectories (`Always`, `Eventually`,
  `Never`, etc.) that compile to one DFA used both at runtime and
  for offline auditing.
- **Speculative tool execution** (`harness.speculate`): pre-execute
  likely tool calls in parallel with model generation. Hits skip the
  dispatch cycle; misses are cancelled. Idempotency-gated.
- **Streaming output** (`harness.streaming`): `Orchestrator.run_stream`
  yields `TextDelta` / `ToolUseStart` / `ToolUseEnd` / `MessageEnd`
  events as the model generates.
- **Telemetry** (`harness.telemetry`): pluggable `Sink` protocol;
  JSONL / Memory / OpenTelemetry sinks ship. Trace_id / span_id
  correlation propagated automatically through the orchestrator and
  dispatcher.

Each of those is a wrapper around the same `Runner` you wrote in
Step 2. Composition is the model:

```python
DebugRunner(                             # adds breakpoints
    PlanGuardedRunner(                   # enforces a plan-as-contract
        PrivacyBoundary(...).wrap(       # scrubs PII outbound + inbound
            AnthropicRunner(             # the real model
                dispatcher, hooks,
                speculator=Speculator(...),       # parallel pre-execution
                prefix_watcher=PrefixWatcher(...) # cache-drift audit
            )
        )
    )
)
```

Each layer is independent; the model layer stays oblivious.

## Where to read next

- [**Cookbook**](cookbook/index.md) — concrete recipes for the
  features above.
- [**Architecture**](architecture.md) — how the seams fit together.
- [**Comparison**](comparison.md) — where harness sits relative to
  LangChain / DSPy / AutoGen / CrewAI.
- [**Module reference**](modules/tools.md) — the per-module API.
- [**Examples**](https://github.com/dr-gareth-roberts/harness-engineering/tree/main/examples)
  — 13 runnable end-to-end examples, one per module-or-feature.

If you got stuck, [FAQ](faq.md) covers the common pitfalls.
