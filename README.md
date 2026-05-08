# harness-engineering

Opensource toolbox for harness engineering — utilities, primitives, and patterns for building robust harnesses around LLM-powered agents and coding tools.

## Scope

The "harness" is everything around the model: prompt assembly, tool wiring, permission gating, hook execution, sub-agent dispatch, memory, retries, sandboxing, telemetry. This repo aims to collect reusable building blocks for that layer — independent of any one CLI or vendor — so harness authors can compose rather than rebuild.

The MVP ships six small, composable modules:

| Module             | What it gives you                                                       |
| ------------------ | ----------------------------------------------------------------------- |
| `harness.tools`    | Pydantic-backed `Tool` + async `Dispatcher` with validation             |
| `harness.prompts`  | `Message` / `ContentBlock`, file attachment, last-N compaction, summarization-based compaction |
| `harness.hooks`    | Typed `Event`s, ordered `HookRunner` with `block`-aware short-circuit   |
| `harness.policy`   | `AllowList` / `DenyList` / `ArgumentMatcher` policies for tool calls    |
| `harness.agents`   | `SubAgent` + `Orchestrator` that emits lifecycle hooks (model-agnostic) |
| `harness.runner`   | `AnthropicRunner` — a real Anthropic SDK runner that closes the tool loop (optional extra: `[anthropic]`) |

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

For a real Anthropic-API loop with prompt caching and the tool dispatcher,
install the extra and run:

```bash
uv sync --extra anthropic
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
