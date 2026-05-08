# harness-engineering

Opensource toolbox for harness engineering — utilities, primitives, and patterns for building robust harnesses around LLM-powered agents and coding tools.

## Scope

The "harness" is everything around the model: prompt assembly, tool wiring, permission gating, hook execution, sub-agent dispatch, memory, retries, sandboxing, telemetry. This repo aims to collect reusable building blocks for that layer — independent of any one CLI or vendor — so harness authors can compose rather than rebuild.

The MVP ships seven small, composable modules:

| Module             | What it gives you                                                       |
| ------------------ | ----------------------------------------------------------------------- |
| `harness.tools`    | Pydantic-backed `Tool` + async `Dispatcher` with validation             |
| `harness.prompts`  | `Message` / `ContentBlock`, file attachment, last-N compaction, summarization-based compaction |
| `harness.hooks`    | Typed `Event`s, ordered `HookRunner` with `block`-aware short-circuit   |
| `harness.policy`   | `AllowList` / `DenyList` / `ArgumentMatcher` policies for tool calls    |
| `harness.agents`   | `SubAgent` + `Orchestrator` that emits lifecycle hooks (model-agnostic) |
| `harness.runner`   | Pluggable runners satisfying the `Runner` protocol: `EchoRunner` / `CannedRunner` (no deps), `AnthropicRunner` (extra `[anthropic]`), `OpenAICompatRunner` for OpenAI / vLLM / Ollama / llama.cpp / LM Studio (extra `[openai-compat]`) |
| `harness.telemetry`| Pluggable `Sink` protocol + `JSONLSink` / `MemorySink` / `MultiSink`; opt-in observability for dispatcher and orchestrator |
| `harness.memory`   | `SessionRecord`, `MemoryStore` protocol, `InMemoryStore` / `FileStore`, plus a `Session` helper that snapshots after every turn |
| `harness.sandbox`  | `PathScope` + `PathPolicy` for filesystem-scoped tool calls, `safe_subprocess_run` with scrubbed env and timeout |
| `harness.replay`   | `ReplayRunner` for deterministic playback, `run_eval` over a list of cases, `compare_sessions` that ignores tool-call IDs |

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Usage

```python
import asyncio
from pydantic import BaseModel
from harness.tools import Dispatcher, Tool, ToolCall

class EchoIn(BaseModel):
    text: str

dispatcher = Dispatcher([
    Tool(name="echo", description="Echo back.", input_model=EchoIn, handler=lambda a: a.text),
])

async def main() -> None:
    result = await dispatcher.dispatch(ToolCall(name="echo", arguments={"text": "hi"}))
    print(result.content)  # -> "hi"

asyncio.run(main())
```

A runnable script that wires all four base modules together lives at
[`examples/end_to_end.py`](examples/end_to_end.py):

```bash
uv run python examples/end_to_end.py
```

## Runners

The `Orchestrator` is model-agnostic — it takes any callable matching the
`Runner` protocol:

```python
Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]
```

The package ships several:

| Runner               | Install                                           | Use it for |
| -------------------- | ------------------------------------------------- | ----------- |
| `EchoRunner`         | base install                                      | smoke tests, examples — echoes the last user message back |
| `CannedRunner`       | base install                                      | unit tests — returns canned strings in order |
| `ReplayRunner`       | base install                                      | replaying captured `SessionRecord`s deterministically |
| `AnthropicRunner`    | `uv sync --extra anthropic`                       | Claude via the Anthropic SDK |
| `OpenAICompatRunner` | `uv sync --extra openai-compat`                   | OpenAI, plus any OpenAI-compatible server: vLLM, llama.cpp, Ollama, LM Studio, Together, Groq |

Adding another vendor is mechanical: create `harness/runner/<vendor>.py`
with a class satisfying the protocol, register it in `harness/runner/__init__.py`,
add an optional extra to `pyproject.toml`. Nothing in `harness.agents`,
`harness.tools`, etc. needs to change.

```bash
ANTHROPIC_API_KEY=sk-ant-... uv run python examples/anthropic_runner.py
```

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
uv run mypy          # type-check (strict)
```

## Roadmap

Out of scope for the MVP (PRs welcome):

- Persistent memory / session storage.
- Telemetry / OpenTelemetry export.
- Sandbox primitives (filesystem, network, subprocess) — `harness.policy` ships
  the tool-call layer; sandbox execution is later.
- Replay / eval harness.
- Additional model runners (OpenAI, etc.) — the `harness.runner` package
  leaves room.

## License

Apache-2.0. See [LICENSE](LICENSE).
