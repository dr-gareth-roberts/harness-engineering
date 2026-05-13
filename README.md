# harness-engineering

Opensource toolbox for harness engineering — utilities, primitives, and patterns for building robust harnesses around LLM-powered agents and coding tools.

## Scope

The "harness" is everything around the model: prompt assembly, tool wiring, permission gating, hook execution, sub-agent dispatch, memory, retries, sandboxing, telemetry. This repo aims to collect reusable building blocks for that layer — independent of any one CLI or vendor — so harness authors can compose rather than rebuild.

### Core primitives

| Module             | What it gives you                                                       |
| ------------------ | ----------------------------------------------------------------------- |
| `harness.tools`    | Pydantic-backed `Tool` + async `Dispatcher` with validation             |
| `harness.prompts`  | `Message` / `ContentBlock`, file attachment, last-N compaction, summarization-based compaction |
| `harness.hooks`    | Typed `Event`s, ordered `HookRunner` with `block`-aware short-circuit; events: `SessionStart` / `SessionEnd` (emitted by `Orchestrator.run`), `PromptSubmit` (emitted by `Session.send`), `PreToolUse` / `PostToolUse` / `PostAssistantMessage` (emitted by tool-loop runners), `Stop` |
| `harness.policy`   | `AllowList` / `DenyList` / `ArgumentMatcher` policies for tool calls    |
| `harness.agents`   | `SubAgent` + `Orchestrator` that emits lifecycle hooks (model-agnostic) |
| `harness.runner`   | Pluggable runners satisfying the `Runner` protocol: `EchoRunner` / `CannedRunner` (no deps), `AnthropicRunner` (extra `[anthropic]`), `OpenAICompatRunner` for OpenAI / vLLM / Ollama / llama.cpp / LM Studio (extra `[openai-compat]`); structural `prefix_watcher` / `speculator` extension kwargs |
| `harness.telemetry`| Pluggable `Sink` protocol + `JSONLSink` / `MemorySink` / `MultiSink`; opt-in observability for dispatcher and orchestrator. `OpenTelemetrySink` available under `[otel]` extra — synthesizes one OTel span per `TelemetryEvent` (span name = `event.kind`, attributes lifted from the event payload), with trace continuity via the recorder's `trace_id` / `span_id` correlation IDs |
| `harness.memory`   | `SessionRecord`, `MemoryStore` protocol, `InMemoryStore` / `FileStore`, plus a `Session` helper that snapshots after every turn |
| `harness.sandbox`  | `PathScope` + `PathPolicy` for filesystem-scoped tool calls, `safe_subprocess_run` with scrubbed env and timeout |
| `harness.replay`   | `ReplayRunner` for deterministic playback, `run_eval`, `compare_sessions` (ignores tool-call IDs), `counterfactual` mutation + continuation, `diff_eval` cross-provider matrix |

### Behavior & enforcement

| Module             | What it gives you                                                       |
| ------------------ | ----------------------------------------------------------------------- |
| `harness.contracts`| Declarative invariants over agent trajectories; predicates (`HasToolUse` / `TextMatches` / `RoleIs` / `ArgMatches`) compose with `&` / `\|`, patterns (`Always` / `Eventually` / `Earlier(...).when(...)` / `Never`); shared DFA backs both `attach_contracts(hooks, ...)` runtime and offline `check(record, ...)`; three actions (`forbid` / `warn` / `require`) |
| `harness.privacy`  | `PrivacyBoundary(detectors).wrap(real_runner)` returns a runner that scans every text fragment, tool_use argument, and tool_result content (recursively, depth-capped) for secrets / PII; `RegexDetector` + `EntropyDetector` with pre-built `SECRET_PACK` / `PII_PACK` / `HIPAA_PACK`; per-detector `direction` (`outbound` / `inbound` / `both`) and `action` (`redact` / `block` / `audit`); audit events never carry the matched value |
| `harness.plan`     | `Plan` (Pydantic, JSON-serializable) of expected `PlannedToolCall`s; `PlanGuardedRunner(real_runner, plan, mode=...)` enforces it via the contracts DFA — deviation raises `PlanViolation`; `derive_plan()` asks a live planner agent to emit one; `infer_plan_from_records(records)` mines a plan from successful past trajectories (modal sequence, default heuristic for "successful", `mode="superset"` so deviations don't fail) |

### Quality & exploration

| Module             | What it gives you                                                       |
| ------------------ | ----------------------------------------------------------------------- |
| `harness.fuzz`     | Hypothesis-driven fuzzing (extra `[fuzz]`); `fuzz_tool` (drives Pydantic-typed inputs through `Dispatcher.dispatch`), `fuzz_agent` (drives them through a full `Orchestrator` turn), `harness_property` pytest decorator; lazy imports — module loads without the extra |
| `harness.attribute`| Causal provenance via leave-one-out ablation; `attribute(session, target, runner, agent, granularity, similarity)` ranks input chunks by influence on a target output. `JaccardSimilarity` / `LengthRatio` zero-dep, `EmbeddingSimilarity` opt-in (extra `[attribute]`) |
| `harness.cache`    | Prompt-prefix-drift watcher; `PrefixWatcher` satisfies the runner's structural `prefix_watcher=` protocol, fingerprints each cache breakpoint per request, `audit(store, window_hours)` surfaces silent invalidators with `unified_diff`; ships `harness cache-audit` CLI subcommand |
| `harness.debug`    | `pdb`-flavored debugger for orchestrator runs; `DebugRunner(real_runner, ...)` wraps any runner, pauses on a configurable predicate, exposes a `DebugContext` for inspect / mutate / fire / resume / abort. Three modes: programmatic (callback), interactive REPL (`harness debug`), and **DAP server over stdio** (`harness debug --dap`) so VS Code / neovim-dap / Emacs dap-mode drive the same replay-driven session |
| `harness.speculate`| Pre-execute likely tool calls in `asyncio` tasks while the model generates; `Speculator` satisfies the runner's structural `speculator=` protocol, hits skip the runner's `PreToolUse` / dispatch / `PostToolUse` cycle. Per-event lifecycle (`begin` / `observe` / `cancel_unobserved` / `try_resolve` / `end`): event-aware runners (`AnthropicRunner`) surface each `tool_use` block as it arrives in the stream so unmatched speculations get cancelled at stream-end, before dispatch begins. Ships `LastCallPredictor`, `SequencePredictor` (bigram), and `CrossSessionPredictor` (loads the K most-recent SessionRecords from a `MemoryStore`, runs bigram logic across the union with sentinel boundaries between sessions); plug a custom `Predictor` as needed. Wired into both `AnthropicRunner` and `OpenAICompatRunner`. Idempotency is a tool-author promise: speculator only fires for `Tool.idempotent=True` |

### CLI

`harness --help` lists the subcommands; new features register their own subparser via a `register(subparsers)` callable lazily discovered via `importlib.util.find_spec`. Currently shipped: `cache-audit`, `debug`.

## Install

Requires Python 3.11+.

```bash
pip install harness-engineering-toolkit
# or
uv add harness-engineering-toolkit
```

The distribution name on PyPI is `harness-engineering-toolkit`
(the shorter `harness-engineering` is held by an unrelated package).
The importable module is `harness`, so user code is unaffected:
`from harness import Orchestrator, Tool, …`.

For local development, clone the repo and:

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

The full per-wave decision log lives in
[`progress.md`](progress.md); the forward plan from where the
codebase is today is in [`docs/plan.md`](docs/plan.md). All ten
standout features from the original design plus the four post-1.0
follow-up waves (CI/CD, observability, modality, streaming) are
shipped; see the [docs site](docs/) for the per-module API reference.

Currently deferred (PRs welcome):

- **Vendor cassette tests** — recorded Anthropic / OpenAI sessions
  replayed in CI to catch SDK shape drift. Needs a one-time recording
  step against the real APIs, gated on credentials.
- **Anthropic Files API upload helper** — `harness.prompts.attach_file(file_id=...)`
  works today; the convenience `upload_file(client, path) -> file_id`
  helper is deferred for the same credential reason.
- **`OpenAICompatRunner.run_stream()`** — `AnthropicRunner.run_stream()`
  ships; OpenAI's chat-completions streaming has a different
  delta-by-delta shape and is queued for a follow-up.

## License

Apache-2.0. See [LICENSE](LICENSE).
