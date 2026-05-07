# harness-engineering

Opensource toolbox for harness engineering — utilities, primitives, and patterns for building robust harnesses around LLM-powered agents and coding tools.

## Scope

The "harness" is everything around the model: prompt assembly, tool wiring, permission gating, hook execution, sub-agent dispatch, memory, retries, sandboxing, telemetry. This repo aims to collect reusable building blocks for that layer — independent of any one CLI or vendor — so harness authors can compose rather than rebuild.

The MVP ships four small, composable modules:

| Module             | What it gives you                                                       |
| ------------------ | ----------------------------------------------------------------------- |
| `harness.tools`    | Pydantic-backed `Tool` + async `Dispatcher` with validation             |
| `harness.prompts`  | `Message` / `ContentBlock`, file attachment, simple compaction          |
| `harness.hooks`    | Typed `Event`s, ordered `HookRunner` with `block`-aware short-circuit   |
| `harness.agents`   | `SubAgent` + `Orchestrator` that emits lifecycle hooks (model-agnostic) |

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

A runnable script that wires all four modules together lives at
[`examples/end_to_end.py`](examples/end_to_end.py):

```bash
uv run python examples/end_to_end.py
```

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
uv run mypy          # type-check (strict)
```

## Roadmap

Out of scope for the MVP (PRs welcome):

- Real model API calls (today the `Orchestrator` takes an injected `runner`).
- Persistent memory / session storage.
- Telemetry / OpenTelemetry export.
- Permission/sandbox primitives — `HookDecision.block` is the seed; full policy engine is later.
- Replay / eval harness.
- Summarization-based compaction.

## License

Apache-2.0. See [LICENSE](LICENSE).
