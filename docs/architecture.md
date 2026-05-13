# Architecture

The package is a collection of small modules wired through structural
protocols. The model itself is opaque; everything else is replaceable.

## The three core seams

**`Runner`** — the model interface.

```python
Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]
```

Anything that takes a subagent + history and returns the next
assistant message satisfies this protocol. Ships:
`EchoRunner`/`CannedRunner` (no deps), `AnthropicRunner` (`[anthropic]`),
`OpenAICompatRunner` (`[openai-compat]`), `ReplayRunner` (deterministic
playback). Wrappers like `DebugRunner`, `PlanGuardedRunner`, and
`PrivacyBoundary.wrap(...)` decorate other runners without taking a
runtime dep on any vendor SDK.

**`Sink`** — the telemetry surface.

```python
class Sink(Protocol):
    async def emit(self, event: TelemetryEvent) -> None: ...
```

Where every observable event lands. `JSONLSink`, `MemorySink`,
`MultiSink` ship in-package; `OpenTelemetrySink` lives behind `[otel]`.
The `Sink` protocol is structural so new sinks (e.g., a Slack notifier)
need only one method.

**`MemoryStore`** — persistence for `SessionRecord`.

```python
class MemoryStore(Protocol):
    async def save(self, record: SessionRecord) -> None: ...
    async def load(self, session_id: str) -> SessionRecord: ...
    ...
```

`InMemoryStore` for tests, `FileStore` for real use. Anything else
(SQLite, S3, Postgres) is a drop-in.

## Composition pattern

Wrapping runners is the standard composition idiom. Each layer adds
one cross-cutting concern:

```
DebugRunner(                       # adds breakpoints (#10)
  PlanGuardedRunner(                # adds plan-as-contract (#9)
    PrivacyBoundary(...).wrap(      # adds outbound/inbound redaction (#6)
      AnthropicRunner(               # the model
        dispatcher, hooks,
        speculator=Speculator(...),  # adds parallel speculation (#5)
        prefix_watcher=PrefixWatcher(...),  # adds drift-audit (#3)
      )
    )
  )
)
```

All five are independent. The model layer stays oblivious to whether
you're recording, debugging, redacting, planning, or speculating —
it just sees a `Runner` interface above and below it.

## Hook taxonomy

Hooks are the structural seam for observability and policy
enforcement. Each event is a Pydantic model dispatched through
`HookRunner.emit(...)`:

| Event | Fired when |
|---|---|
| `SessionStart` | `Orchestrator.run` enters |
| `SessionEnd` | `Orchestrator.run` exits (success or error) |
| `PromptSubmit` | `Session.send` accepts a user message, before the orchestrator runs |
| `PreToolUse` | Before the dispatcher invokes a tool. Returning a `block` decision short-circuits dispatch. |
| `PostToolUse` | After dispatch — every tool call (success or error) sees this |
| `PostAssistantMessage` | After the runner produces an assistant message — observable both in DebugRunner mutations and live model output |
| `Stop` | Convenient terminal-step hook for cleanup |

`harness.policy` (AllowList/DenyList/ArgumentMatcher) attaches to
`PreToolUse` to enforce tool-call invariants without modifying the
runner.

## Idempotency contract

Several features depend on tools that can be safely re-run with the
same arguments. The package treats `Tool.idempotent=True` as a *promise*
by the tool author:

- `Speculator` only fires for `idempotent=True` tools by default.
- `ReplayRunner` re-runs idempotent tools in counterfactual analysis.

Marking a tool idempotent when it actually has side effects produces
silent duplicate side effects on a speculator miss. This is documented
in both module docstrings and [Speculator's class doc](modules/speculate.md).

## Where things live

```
src/harness/
├── tools/      Tool + Dispatcher (the model-callable surface)
├── prompts/    Message, ContentBlock, file attachments, compaction
├── hooks/      HookRunner + typed events
├── policy/     AllowList / DenyList / ArgumentMatcher
├── agents/     SubAgent + Orchestrator
├── runner/     EchoRunner, CannedRunner, AnthropicRunner, OpenAICompatRunner
├── telemetry/  Sink protocol + JSONLSink/MemorySink/MultiSink
├── memory/     SessionRecord + MemoryStore + Session helper
├── sandbox/    PathScope/PathPolicy + safe_subprocess_run
├── replay/     ReplayRunner + run_eval + counterfactual + diff_eval
├── contracts/  Predicates + patterns + DFA + attach_contracts
├── privacy/    PrivacyBoundary + Detectors + packs
├── plan/       Plan + PlanGuardedRunner + plan inference
├── fuzz/       Pydantic→Hypothesis bridge + tool/agent fuzzers
├── attribute/  Leave-one-out causal attribution
├── cache/      Prompt-prefix-drift watcher
├── debug/      DebugRunner + REPL + DAP server
└── speculate/  Predictor + Speculator + cross-session predictor
```
