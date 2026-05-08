# Roadmap progress log

> Living document for the post-MVP roadmap work on `harness-engineering`.
> Each item gets its own section with plan, decisions, and a per-step log.
> Append-only — older entries stay; status is updated in place.

## Status snapshot

| # | Item                                   | Status      | Branch / PR                                    |
| - | -------------------------------------- | ----------- | ---------------------------------------------- |
| 0 | MVP scaffold (tools/prompts/hooks/agents/policy) | shipped | PR #1 (`chore/initial-scaffold` → `main`)      |
| 1 | Real model runner + summarization-compaction | shipped | PR #1                                          |
| 2 | Telemetry / structured event stream    | shipped     | PR #1                                          |
| 3 | Persistent memory / session storage    | shipped     | PR #1                                          |
| 4 | Sandbox execution primitives           | pending     | TBD                                            |
| 5 | Replay / eval harness                  | pending     | TBD                                            |

## Order rationale

The dependency graph determined the order. Telemetry could have gone first
(foundation for replay/eval) but the real model runner is the highest-visibility
gap — without it, the library is glue with no model. Telemetry comes next so
the runner, memory, sandbox, and replay all emit through the same stream.

```
[1] real-model-runner ──┐
                         ├──► [2] telemetry ──► [5] replay/eval
                         │           ▲
                         ▼           │
   summarization-compaction          │
                                     │
[3] persistent-memory  ──────────────┤
[4] sandbox-execution  ──────────────┘
```

## Cross-cutting decisions

- **Optional extras over runtime deps.** Each item that pulls in a heavy
  dependency (Anthropic SDK, OpenTelemetry, …) lands as `[extras]` so the
  base install stays at `pydantic` only. Imports at the top of submodules use
  guarded `try/except ImportError` with a clear error pointing at the extra.
- **Vendor-neutral primitives, vendor-specific glue.** Core types live in
  the base package; concrete integrations live in `harness.<module>.<vendor>`
  submodules (e.g. `harness.runner.anthropic`).
- **Append to PR #1, not a stack of separate PRs.** PR #1 is still pending
  review and the items are conceptually one delivery — "the post-MVP layer".
  Each item is one focused commit on `chore/initial-scaffold`.

---

## Item 1 — Real model runner + summarization-compaction

### Goal
Provide a real `Orchestrator` runner that talks to a Claude model via the
Anthropic SDK, handles a complete tool-use loop using the existing
`harness.tools.Dispatcher`, supports prompt caching markers, and ships a
summarization-based compaction strategy that uses the runner for its summary call.

### Status
- Shipped. PR #1 commits `87…` (TBD on push) — see Implementation log.

### Decisions
- **Vendor namespace.** Anthropic-specific code lives in `harness.runner.anthropic`. The base package keeps zero non-Pydantic deps; `anthropic` is an optional extra (`pip install harness-engineering[anthropic]`). Other vendors can land alongside (`harness.runner.openai`, etc.) without churn.
- **Manual tool loop, not the SDK tool runner.** Our `Tool` already carries an explicit Pydantic input model and `json_schema()` returns Anthropic-shaped tool definitions. A manual loop lets the runner reuse the existing `Dispatcher` (validation, error wrapping) and fire `PreToolUse`/`PostToolUse` hooks around each call — both lost if we delegate to `client.beta.messages.tool_runner()`.
- **Streaming by default.** Per the `claude-api` skill, "default to streaming for any request that may involve long input, long output, or high `max_tokens`." Use `client.messages.stream()` + `get_final_message()` so we never have to hand-handle SSE events; the SDK accumulates state for us. `max_tokens` defaults to `16_000` (under the SDK's no-stream guard) and can go higher when streaming.
- **Adaptive thinking on by default for Opus 4.7 / 4.6 / Sonnet 4.6.** The skill is explicit: `thinking: {type: "adaptive"}` for "anything remotely complicated", with `effort` controlling depth. Older models would need `thinking: {type: "enabled", budget_tokens: N}` — out of scope for MVP. Default model becomes `claude-opus-4-7` (the skill's mandated default).
- **System messages map to the API's `system` field, not into `messages[]`.** Anthropic's Messages API only accepts `user`/`assistant` roles in `messages`; `system` is a separate top-level parameter. The translator pulls all `role="system"` messages out of the harness `Message` list, joins their text, and sends them as `system`.
- **Cache markers propagate via `cache_control: {"type": "ephemeral"}`.** Any harness `ContentBlock` with `cache=True` becomes the cacheable boundary on the rendered Anthropic block. We honour the prefix-match invariant from `shared/prompt-caching.md` — cache flags must sit at stable prefix boundaries; users misuse them at their own risk, but we don't try to be clever about it.
- **`SubAgent.allowed_tools` is an explicit allowlist.** Empty list → no tools sent to the model. The dispatcher remains the source of truth for *what* tools exist; the agent decides *which* to expose. This is how `harness.policy` plugs in: the same `HookRunner` policy stack runs around dispatch regardless of who initiated the call.
- **`SubAgent` stays vendor-neutral.** Knobs that are vendor-specific (`max_tokens`, `effort`, `thinking_mode`) live on the runner constructor, not on `SubAgent`. If/when we need per-agent overrides, we add an `AnthropicRunner.config_for(agent)` hook — out of scope for MVP.
- **`summarize_compact()` lives in `harness.prompts.compaction`** next to the existing `compact()`. It takes a `Runner`-shaped callable so it stays vendor-neutral; in practice callers pass an `AnthropicRunner`. Bundled with item 1 because it needs a model to do its work.
- **No real API hits in CI.** Unit tests inject a `FakeAsyncAnthropic` (a small protocol-shaped fake; the SDK's `AsyncAnthropic` is too heavy and changes shape across versions). A real-API smoke test lives at `examples/anthropic_runner.py`, gated on `ANTHROPIC_API_KEY` being set.

### Plan

#### Architecture

```mermaid
graph LR
    subgraph harness
        agents["Orchestrator.run()"] -->|injects| runner
        runner["AnthropicRunner.__call__(agent, messages)"]
        runner -->|translate| ap["anthropic-shaped<br/>system + messages + tools"]
        ap --> sdk["AsyncAnthropic.messages.stream"]
        sdk -->|response| runner
        runner -->|tool_use| disp["Dispatcher.dispatch"]
        runner -->|emit Pre/PostToolUse| hooks["HookRunner"]
        policy["AllowList / etc"] -.-> hooks
        disp -->|tool_result| runner
        runner -->|final Message| agents
    end
    style runner fill:#fef3c7
```

The runner is the only module that imports `anthropic`. Everything else stays vendor-neutral.

#### Files

**New:**
- `src/harness/runner/__init__.py` — re-export `AnthropicRunner` (guarded import).
- `src/harness/runner/anthropic.py` — `AnthropicRunner` class + message/tool translators. Top-of-module `try: import anthropic except ImportError: raise ImportError("install harness-engineering[anthropic]") from None`.
- `tests/runner/__init__.py`
- `tests/runner/test_anthropic.py` — unit tests with a fake client.
- `tests/runner/fakes.py` — `FakeAsyncAnthropic` and helpers to script tool-use loops.
- `examples/anthropic_runner.py` — real API smoke test, gated on env var, demonstrates a tool loop end-to-end.

**Modified:**
- `pyproject.toml` — add `[project.optional-dependencies] anthropic = ["anthropic>=0.60"]`. Floor verified: `output_config` and `thinking` are present in the SDK type system at this version; `claude-opus-4-7` is a valid model string. The actual install pins via `uv lock`.
- `src/harness/__init__.py` — re-export `AnthropicRunner` from the top level.
- `src/harness/agents/definition.py` — change default `model` from `"claude-sonnet-4-6"` to `"claude-opus-4-7"` per the skill's mandated default.
- `src/harness/prompts/compaction.py` — add `summarize_compact()` and a `_DEFAULT_SUMMARY_PROMPT` constant. Keeps the existing `compact()` untouched.
- `src/harness/prompts/__init__.py` — re-export `summarize_compact`.
- `tests/prompts/test_compaction.py` — add tests for `summarize_compact` using a fake `Runner` callable.
- `examples/end_to_end.py` — leave untouched (it's the no-API smoke test).
- `README.md` — small Usage section addition showing the runner; move "Real model API calls" from the Roadmap to the module table.
- `progress.md` (this file) — keep updating the per-item status + log.

#### `AnthropicRunner` shape

```python
class AnthropicRunner:
    """Implements the Runner protocol for Anthropic-hosted Claude models.

    Drives a manual tool-use loop using harness.tools.Dispatcher, fires
    Pre/PostToolUse hooks around each dispatch, and respects HookDecision.block
    by returning an error tool_result to the model instead of dispatching.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        hooks: HookRunner,
        *,
        client: AsyncAnthropic | None = None,    # injectable for tests
        max_tokens: int = 16_000,
        thinking_mode: Literal["adaptive", "disabled"] = "adaptive",
        effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
        max_iterations: int = 10,                # cap on tool-use loop turns
    ) -> None: ...

    async def __call__(
        self,
        agent: SubAgent,
        messages: list[Message],
    ) -> Message: ...
```

It satisfies `Runner = Callable[[SubAgent, list[Message]], Awaitable[Message]]` so it slots straight into `Orchestrator(dispatcher, hooks, runner=AnthropicRunner(...))`.

#### Translation rules

| Harness | Anthropic API |
| --- | --- |
| `Message(role="system", ...)` | extracted, joined into the top-level `system` parameter |
| `Message(role="user"/"assistant", content=[...])` | one `messages[]` entry |
| `ContentBlock(type="text", text=t, cache=True)` | `{"type":"text","text":t,"cache_control":{"type":"ephemeral"}}` |
| `ContentBlock(type="tool_use", tool_use=tc)` | `{"type":"tool_use","id":tc.id,"name":tc.name,"input":tc.arguments}` |
| `ContentBlock(type="tool_result", tool_result=tr)` | `{"type":"tool_result","tool_use_id":tr.id,"content":..., "is_error":tr.is_error}` |
| `ContentBlock(type="file", path=p, text=body)` | `{"type":"text","text":f"<file path={p}>\n{body}\n</file>"}` (Files API integration is out of scope for MVP) |

`tool_result.content` is rendered as `str(tr.content)` if it's already a string or scalar; `dict` / `list` get `json.dumps(..., default=str)` so the model sees a clean JSON value rather than `"{'a': 1}"`. Anthropic accepts strings or content-block lists, so this stays simple.

`Tool.json_schema()` already returns `{"name", "description", "input_schema"}` — feed the list straight to `tools=...` after filtering by `agent.allowed_tools`.

#### Loop body (sketch)

```
def __call__(agent, messages):
    api_messages, system = translate_in(messages)
    tools = [s for s in dispatcher.tools_schema() if s["name"] in agent.allowed_tools]
    request_kwargs = build_kwargs(agent, system, api_messages, tools)
    for _ in range(max_iterations):
        async with client.messages.stream(**request_kwargs) as s:
            response = await s.get_final_message()
        if response.stop_reason in ("end_turn", "stop_sequence"):
            return translate_out_assistant(response)        # final assistant Message
        if response.stop_reason == "tool_use":
            api_messages.append({"role":"assistant","content":response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use": continue
                call = ToolCall(name=block.name, arguments=block.input, id=block.id)
                decisions = await hooks.emit(PreToolUse(call=call))
                blocked = next((d for d in decisions if d.block), None)
                if blocked:
                    result = ToolResult(id=block.id, content=blocked.reason or "blocked", is_error=True)
                else:
                    result = await dispatcher.dispatch(call)
                await hooks.emit(PostToolUse(call=call, result=result))
                tool_results.append(translate_tool_result(result))
            api_messages.append({"role":"user","content":tool_results})
            continue
        raise RuntimeError(f"unexpected stop_reason: {response.stop_reason}")
    raise RuntimeError(f"tool-use loop exceeded {max_iterations} iterations")
```

Error stop reasons (`refusal`, `pause_turn`) are out of scope for MVP — surface as `RuntimeError` so callers see them clearly. We can grow these out later.

#### `summarize_compact()` shape

```python
async def summarize_compact(
    messages: list[Message],
    runner: Runner,          # vendor-neutral — Callable[[SubAgent, list[Message]], Awaitable[Message]]
    *,
    keep_last: int = 8,
    keep_system: bool = True,
    summary_agent: SubAgent | None = None,    # defaults to a small "summarizer" SubAgent
) -> list[Message]: ...
```

Returns: kept system messages + a synthesised `system`-role summary message + last N non-system messages. The runner is called once with the messages we're about to drop, prompted to produce a tight summary. Pure async function; no I/O beyond the runner call.

#### Tests

`tests/runner/test_anthropic.py` (with `FakeAsyncAnthropic`):
1. **Translation round-trip.** Feed a mixed conversation in, assert the synthesised API request shape (system extracted, cache markers placed, tool_use/tool_result blocks well-formed).
2. **No-tool happy path.** Fake returns `stop_reason="end_turn"` with one text block → runner returns assistant `Message` with one text block.
3. **One-iteration tool loop.** Fake returns `stop_reason="tool_use"` with one tool_use → runner dispatches via `Dispatcher` → second fake call returns `stop_reason="end_turn"` → final assistant message returned.
4. **Hook block short-circuits dispatch.** Register an `AllowList` policy that rejects the tool the fake "model" wants → runner sends a `tool_result` with `is_error=True` and the rejection reason → second call still happens.
5. **`max_iterations` cap.** Fake keeps returning `tool_use` → runner raises `RuntimeError` after the configured cap.
6. **`allowed_tools` filter.** Tools not in `agent.allowed_tools` are not sent to the API, even if they're registered in the dispatcher.
7. **Cache marker propagation.** A harness `ContentBlock(cache=True)` becomes `cache_control={"type":"ephemeral"}` on the rendered API block.
8. **Missing dep error.** Use `monkeypatch.setitem(sys.modules, "anthropic", None)` + `importlib.reload(harness.runner.anthropic)` so the test runs deterministically whether or not the extra is installed in CI. Asserts the raised `ImportError` mentions `harness-engineering[anthropic]`.

`tests/prompts/test_compaction.py`:
- Add 3 tests for `summarize_compact`: keeps system + last N + injects summary; honours `keep_system=False`; calls the runner exactly once.

#### Verification

Same gates as the MVP, plus the example:
- `uv sync --extra dev --extra anthropic` — installs cleanly.
- `uv run pytest` — all tests green.
- `uv run ruff check .` — clean.
- `uv run mypy` — clean (strict).
- `uv run python examples/end_to_end.py` — still passes (sanity check we didn't regress the no-API path).
- `ANTHROPIC_API_KEY=… uv run python examples/anthropic_runner.py` — exits 0, transcript shows a real tool loop.

#### Caveats / explicit non-handling

- **`HookDecision.replacement` is ignored.** The runner only acts on `block`. Replacement-based steering (rewriting tool args, splicing in synthetic results) lands later if we need it; for MVP it's a typed escape hatch we don't honour.
- **Cache-marker cap.** Anthropic caps `cache_control` at 4 breakpoints per request; we render markers 1:1 from `cache=True` flags and don't enforce the cap. If the user marks 5+ blocks, the API will 400. Documented in the runner docstring; users can use `compact()` or trim before calling.
- **Default model spillover.** Changing `SubAgent.model` default from `claude-sonnet-4-6` to `claude-opus-4-7` ripples through `tests/agents/test_orchestrator.py` and `examples/end_to_end.py`, both of which construct `SubAgent` without specifying `model`. The fake runner ignores it so tests stay green; flag in the impl log so reviewers see the change.

#### Out of scope (deferred)

- Files API integration (`file` blocks become text-wrapped instead).
- `pause_turn` / `refusal` stop-reason handling.
- Streaming events to the caller (we accumulate the full message via `get_final_message`).
- Per-agent runner config overrides on `SubAgent`.
- A non-Anthropic runner. Module structure leaves room.

### Implementation log

- **Plan reviewed by advisor.** Three blocking items addressed before code:
  - Fixed `dispatcher.tools_schema()` usage in the loop sketch (avoiding `_tools` private access).
  - Bumped declared anthropic floor from `>=0.39` to `>=0.60`. Verified empirically by installing into a scratch dir: `0.100.0` resolves with `output_config` and `thinking` present in the type system.
  - Switched the missing-dep test to `monkeypatch.setitem(sys.modules, "anthropic", None)` + `importlib.reload`, so it runs deterministically in CI regardless of whether the extra is installed.
- **Default model spillover.** `SubAgent.model` default changed from `claude-sonnet-4-6` to `claude-opus-4-7`. Rippled through `tests/agents/test_orchestrator.py` and `examples/end_to_end.py` (both construct `SubAgent` without `model`) — no behavioural change because the fake runners don't read it.
- **Lazy import on the package root.** `from harness import AnthropicRunner` works only when `[anthropic]` is installed; `import harness` always works. Implemented via module `__getattr__` on both `harness` and `harness.runner`.
- **Translation rules implemented as documented.** System messages flatten to the top-level `system` parameter; `cache=True` propagates as `cache_control: {"type": "ephemeral"}`; tool result content is `json.dumps`-serialized for dicts/lists, `str()` otherwise; file blocks render as `<file path=...>\n...\n</file>` text.
- **Hook block path.** When a `PreToolUse` hook returns `block=True`, the dispatcher is skipped entirely and the API gets a `tool_result` with `is_error=True` and the block reason. `PostToolUse` still fires (with the synthesized error result) so audit hooks see every attempted call.
- **Verification (final gates).**
  - `uv sync --extra dev --extra anthropic` — clean.
  - `uv run pytest` — 54 passed (was 38; +13 runner, +3 summarize_compact).
  - `uv run ruff check .` — clean.
  - `uv run mypy` — clean (strict, 18 source files).
  - `uv run python examples/end_to_end.py` — exits 0; the no-API smoke path still works after the default-model change.
  - `examples/anthropic_runner.py` — wired up; gated on `ANTHROPIC_API_KEY`. Not run in CI; would need a real key to smoke-test.
- **Commit:** `feat(runner): add AnthropicRunner + summarization-based compaction` (TBD on push).

---

## Item 2 — Telemetry / structured event stream

### Goal
Emit a typed event stream covering every dispatcher call and orchestrator turn,
with timestamps and durations. Provide a pluggable `Sink` protocol with a few
concrete implementations (Null / Memory / JSONL / Multi). Keep the base install
zero-dependency; OTel integration is deferred (the structure leaves room).

### Status
- Shipped.

### Decisions
- **Separate from hooks.** `harness.hooks` is about *control* (`HookDecision.block`); telemetry is about *observation* — sinks never block the run, never delay it materially, and never crash it. Sink errors are swallowed at the `Telemetry` boundary. Different audience, different semantics, different module.
- **Pydantic event types.** `TelemetryEvent` base + concrete subclasses (`ToolDispatched`, `OrchestratorTurn`). Carries `event_id: UUID`, `timestamp: datetime`, plus payload-specific fields (durations in ms, agent names, tool names, error strings). Schema evolves freely without affecting `harness.hooks.events`.
- **`Sink` is a Protocol, not an ABC.** Anyone with `async emit(event)` qualifies. We ship `NullSink` (default), `MemorySink` (testing), `JSONLSink` (file or stream), `MultiSink` (fan-out). OTel sink lands later under `[otel]`.
- **Wire-in is opt-in via constructor injection.** `Dispatcher(tools, *, telemetry=None)` and `Orchestrator(dispatcher, hooks, runner, *, telemetry=None)` both accept an optional `Telemetry` instance. Default `None` → no events emitted, no overhead, no behaviour change for existing callers. This is backward-compatible because `Dispatcher`'s positional contract (the iterable of tools) is unchanged.
- **MVP scope is dispatcher + orchestrator only.** `AnthropicRunner` is not instrumented in this round (its tool calls already flow through `Dispatcher`, so `ToolDispatched` events still fire). `HookRunner` is not instrumented either — adding a `HookFired` event would also be useful but each module touched expands scope; defer to a follow-up if user demand surfaces.
- **Failure isolation at the recorder, not the sink.** `Telemetry.emit()` wraps each `await sink.emit(event)` in `try/except Exception` and logs at WARNING via the stdlib `logging` module. The base library never silently swallows errors except at this one boundary. `MultiSink` does the same per-sink so one failing sink doesn't poison the others.
- **No background task or async queue.** `await telemetry.emit(...)` is awaited inline. A background-queued sink can wrap `JSONLSink` later if needed. Keeps the failure model simple — back-pressure shows up as awaitable latency at the call site.

### Plan

#### Architecture

```mermaid
graph LR
    disp["Dispatcher.dispatch()"] -->|emit| tel["Telemetry.emit()"]
    orch["Orchestrator.run()"] -->|emit| tel
    tel -->|fan-out| ms["MultiSink"]
    ms --> n["NullSink"]
    ms --> mem["MemorySink"]
    ms --> jl["JSONLSink<br/>(file / stream)"]
    ms -.future.-> otel["OpenTelemetrySink<br/>([otel] extra)"]
    style tel fill:#fef3c7
```

#### Files

**New:**
- `src/harness/telemetry/__init__.py` — re-exports
- `src/harness/telemetry/events.py` — `TelemetryEvent`, `ToolDispatched`, `OrchestratorTurn`
- `src/harness/telemetry/sinks.py` — `Sink` protocol, `NullSink`, `MemorySink`, `JSONLSink`, `MultiSink`
- `src/harness/telemetry/recorder.py` — `Telemetry` (the central emit hub)
- `tests/telemetry/__init__.py`
- `tests/telemetry/test_sinks.py`
- `tests/telemetry/test_integration.py` — exercises Dispatcher + Orchestrator wired to a `MemorySink`

**Modified:**
- `src/harness/tools/dispatcher.py` — accept `telemetry: Telemetry | None = None` kwarg; emit `ToolDispatched` at the end of each `dispatch()`.
- `src/harness/agents/orchestrator.py` — same kwarg; emit `OrchestratorTurn` after `run()` completes (success or failure).
- `src/harness/__init__.py` — add `Telemetry`, `MemorySink`, `JSONLSink` to top-level exports.
- `README.md` — add a Telemetry row to the module table.
- `progress.md` — status + impl log.

#### Event types

```python
class TelemetryEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    kind: str

class ToolDispatched(TelemetryEvent):
    kind: Literal["tool.dispatched"] = "tool.dispatched"
    tool_name: str
    call_id: str | None
    arguments: dict[str, Any]
    is_error: bool
    duration_ms: float

class OrchestratorTurn(TelemetryEvent):
    kind: Literal["orchestrator.turn"] = "orchestrator.turn"
    agent_name: str
    duration_ms: float
    error: str | None = None        # exception class + message if the runner raised
```

#### `Sink` protocol and concretions

```python
class Sink(Protocol):
    async def emit(self, event: TelemetryEvent) -> None: ...

class NullSink: ...        # no-op
class MemorySink:
    events: list[TelemetryEvent]
    async def emit(self, event): self.events.append(event)

class JSONLSink:
    """Writes one JSON line per event to a file path or open text stream.

    When backed by a path: opens in append mode per emit; `O_APPEND` makes
    single writes atomic for typical event sizes, but a per-instance
    `asyncio.Lock` around writes guards against torn lines under
    `Orchestrator.run_parallel`. Sufficient for in-process concurrency;
    cross-process locking is out of scope.
    """
    def __init__(self, target: TextIO | Path | str): ...
    async def emit(self, event):
        line = event.model_dump_json()
        # if path: lock + open(append) + write(line+'\n') + flush + close.
        # if stream: lock + write(line+'\n') + flush.

class MultiSink:
    def __init__(self, *sinks: Sink): ...
    async def emit(self, event):
        for s in self._sinks:
            try: await s.emit(event)
            except Exception: logger.warning(...)
```

#### `Telemetry` recorder

```python
class Telemetry:
    def __init__(self, sink: Sink | None = None) -> None:
        self._sink: Sink = sink if sink is not None else NullSink()

    async def emit(self, event: TelemetryEvent) -> None:
        try:
            await self._sink.emit(event)
        except Exception:
            logger.warning("telemetry sink %r failed", self._sink, exc_info=True)
```

#### Wire-in

The existing `Dispatcher.dispatch()` body is factored into a private `_dispatch_inner(call)` coroutine; the public `dispatch()` becomes a thin timing-and-emit wrapper. Argument dicts are passed through `json.loads(json.dumps(..., default=str))` at event-construction time so a `Path` or other non-JSON-native value never crashes a `JSONLSink`. Documented as a field invariant on `ToolDispatched.arguments`.

```python
# Dispatcher.dispatch()
start = time.perf_counter()
result = await self._dispatch_inner(call)
duration_ms = (time.perf_counter() - start) * 1000
if self._telemetry is not None:
    await self._telemetry.emit(ToolDispatched(
        tool_name=call.name,
        call_id=call.id,
        arguments=_jsonify(call.arguments),       # coerce to JSON-safe dict
        is_error=result.is_error,
        duration_ms=duration_ms,
    ))
return result

# Orchestrator.run()
start = time.perf_counter()
err: str | None = None
try:
    return await self._runner(agent, messages)
except Exception as exc:
    err = f"{type(exc).__name__}: {exc}"
    raise
finally:
    duration_ms = (time.perf_counter() - start) * 1000
    if self._telemetry is not None:
        await self._telemetry.emit(OrchestratorTurn(
            agent_name=agent.name, duration_ms=duration_ms, error=err,
        ))
```

#### Tests

`tests/telemetry/test_sinks.py`:
1. `MemorySink` collects events in emit order.
2. `JSONLSink` to a `StringIO` writes one valid JSON line per event; trailing newline; flushed.
3. `JSONLSink` to a `Path` opens in append mode (so a second `emit` doesn't truncate).
4. `MultiSink` fans out to every sink; one failing sink does not stop the others.
5. `Telemetry` swallows sink exceptions.
6. `NullSink` returns `None`.

`tests/telemetry/test_integration.py`:
1. `Dispatcher(..., telemetry=t)` emits `ToolDispatched` per `dispatch()` with correct `tool_name`, `is_error`, and a positive `duration_ms`.
2. `Dispatcher(..., telemetry=None)` emits nothing (sanity check that the default truly is silent).
3. `Orchestrator(..., telemetry=t)` emits `OrchestratorTurn` after a successful `run()`.
4. `Orchestrator(..., telemetry=t)` emits `OrchestratorTurn` with `error` populated when the runner raises (and re-raises the exception).
5. `Orchestrator.run_parallel(...)` with a shared `MemorySink` emits N `OrchestratorTurn` events; with a shared `JSONLSink` writes N well-formed JSON lines (no torn lines).
6. Non-JSON-native arguments (e.g. a `Path`) round-trip through `JSONLSink` without raising.

#### Verification

- `uv sync --extra dev --extra anthropic` — clean.
- `uv run pytest` — green.
- `uv run ruff check .` — clean.
- `uv run mypy` — clean (strict).

#### Out of scope (deferred)

- OpenTelemetry sink (under a future `[otel]` extra).
- `HookFired` events / instrumenting `HookRunner`.
- `AnthropicRunner` directly emitting `api.request.*` events; the dispatcher path covers tool dispatch already.
- Background-queue sinks; users can wrap `JSONLSink` if they need it.

### Implementation log

- **Plan reviewed by advisor.** Three blocking items addressed before code:
  - Factored existing `Dispatcher.dispatch()` body into a private `_dispatch_inner()` so the public method is a clean timing-and-emit wrapper.
  - Added `jsonify(value)` helper at `harness.telemetry.events` and applied it at `ToolDispatched` construction time so `Path` / dataclass / etc. arguments survive a JSONL sink (mirrors `_serialize_tool_content` instinct from item 1).
  - Added a per-instance `asyncio.Lock` to `JSONLSink`, plus `MemorySink`, so concurrent writes from `run_parallel` cannot tear lines.
- **Wire-in is opt-in.** Both `Dispatcher` and `Orchestrator` accept `telemetry: Telemetry | None = None` (kwarg-only). Default `None` → no events emitted, no overhead, fully backward-compatible. All existing tests still pass without modification.
- **Failure isolation at two layers.** `MultiSink` catches per-sink exceptions and logs at WARNING; `Telemetry.emit` catches a final outer exception and logs the same way. A misbehaving sink can never crash a dispatch or orchestrator turn.
- **Top-level re-exports.** `Telemetry`, `MemorySink`, `JSONLSink` available from `harness` directly. `Sink` / `NullSink` / `MultiSink` / `TelemetryEvent` / `OrchestratorTurn` / `ToolDispatched` accessible via `harness.telemetry`.
- **Verification (final gates).**
  - `uv sync --extra dev --extra anthropic` — clean.
  - `uv run pytest` — 69 passed (was 54; +15 telemetry).
  - `uv run ruff check .` — clean.
  - `uv run mypy` — clean (strict, 22 source files).
  - `uv run python examples/end_to_end.py` — still passes; the no-API smoke path is unaffected by the wire-in.
- **Commit:** `feat(telemetry): pluggable sink protocol + dispatcher/orchestrator wire-in` (TBD on push).

---

## Item 3 — Persistent memory / session storage

### Goal
Capture a `SessionRecord` (id, agent, full message history, metadata, timestamps)
and persist it via a `MemoryStore` protocol. Ship `InMemoryStore` and `FileStore`
implementations. Provide a small `Session` helper that wraps an `Orchestrator` +
`MemoryStore` to give callers a "send a message, get a reply, snapshot
automatically" surface — the higher-level convenience that turns the per-turn
runner into a multi-turn conversation.

### Status
- Shipped.

### Decisions
- **Vendor-neutral.** The store and record types depend only on `harness.prompts`/`harness.agents`; nothing here imports the Anthropic SDK. Persistence is a pure data layer.
- **`SessionRecord` is the source of truth.** Pydantic model containing `session_id`, `agent` (full `SubAgent`), `messages` (full conversation history), `created_at`, `updated_at`, and a free-form `metadata: dict[str, Any]`. Serializes to JSON via `model_dump_json()`. Tool calls and decisions are *already in the messages* (as `ContentBlock(type="tool_use"/"tool_result")`); we don't duplicate them at the record level.
- **`MemoryStore` is a `Protocol`, not an ABC.** Methods: `save(record)`, `load(session_id) -> SessionRecord | None`, `list(*, limit) -> list[SessionRecord]`, `delete(session_id) -> bool`. Mirrors the `Sink` shape from telemetry.
- **Two implementations.** `InMemoryStore` (dict-backed, async-locked, deep-copy on save/load to prevent caller mutation), `FileStore` (one JSON file per session in a directory, atomic writes via tmp-file + rename). Both lock per-instance with `asyncio.Lock`.
- **`Session` helper, not Orchestrator wire-in.** Persistence is a higher-level concern than per-turn execution. `Session(orchestrator, agent, store, session_id=…)` holds the message list, exposes `await session.send(text_or_message)` and a `Session.restore(...)` classmethod. The `Orchestrator` itself stays unchanged — keeps the lower layer free of session-state assumptions and aligns with the existing pattern where messages are caller-owned.
- **`session.send()` accepts `str | Message`.** A bare string becomes `text("user", s)`. Convenience over ceremony.
- **No torn writes (not full crash safety).** `FileStore.save` writes to `{name}.tmp` and `os.replace()` to the final path. `os.replace` is atomic on POSIX and on Windows (same-volume, no open handles). This guarantees no reader sees a partially-written file — but without `fsync` it does NOT guarantee the rename survives sudden power loss; on a crash you'll see either the previous good copy or the new one, not garbage. That's the right MVP trade-off; we document this contract precisely. `InMemoryStore.save` deep-copies the record (`model_copy(deep=True)`) so subsequent caller mutations don't bleed into stored state.
- **Tmp filename is unambiguous.** Use `path.parent / (path.name + ".tmp")`, not `path.with_suffix(".tmp")` — the latter has interpretation edge cases when the session_id contains dots.
- **`list()` returns most-recently-updated first.** Both stores sort by `updated_at` descending so the contract is consistent regardless of backend (the directory walk for `FileStore` would otherwise have filesystem-defined order).
- **`Session` is single-writer per session_id.** Two concurrent `Session.restore(same_id)` instances racing `send()` is a last-writer-wins race; the second save silently overwrites the first. Documented in the `Session` docstring as the caller's responsibility — optimistic concurrency / etag preconditions are deferred to a follow-up.
- **No automatic load on Session construction.** `Session(...)` always starts fresh; `Session.restore(...)` is the explicit path. This avoids a surprising "did I get an empty session or a hydrated one?" question at construction time.

### Plan

#### Architecture

```mermaid
graph LR
    user["caller"] -->|send| sess["Session"]
    sess -->|run| orch["Orchestrator"]
    sess -->|save| store["MemoryStore"]
    store --> mem["InMemoryStore"]
    store --> fs["FileStore"]
    sess -.restore.-> store
```

#### Files

**New:**
- `src/harness/memory/__init__.py`
- `src/harness/memory/record.py` — `SessionRecord`, `SessionNotFound` exception
- `src/harness/memory/store.py` — `MemoryStore` protocol, `InMemoryStore`, `FileStore`
- `src/harness/memory/session.py` — `Session` helper
- `tests/memory/__init__.py`
- `tests/memory/test_record.py` — round-trips, timestamps, custom metadata
- `tests/memory/test_store.py` — parametrized across both stores: save/load/list/delete, missing-ID returns None, deep-copy isolation, FileStore atomicity (write + crash sim by leaving a `.tmp` file)
- `tests/memory/test_session.py` — send accumulates messages, save fires after each turn, restore round-trips a stored session

**Modified:**
- `src/harness/__init__.py` — add `Session`, `SessionRecord`, `InMemoryStore`, `FileStore`
- `README.md` — module table row
- `progress.md` — status + log

#### `SessionRecord`

```python
class SessionRecord(BaseModel):
    session_id: str
    agent: SubAgent
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def touched(self) -> "SessionRecord":
        """Return a copy with `updated_at` set to now."""
        return self.model_copy(update={"updated_at": datetime.now(UTC)})
```

#### `MemoryStore` protocol

```python
class MemoryStore(Protocol):
    async def save(self, record: SessionRecord) -> None: ...
    async def load(self, session_id: str) -> SessionRecord | None: ...
    async def list(self, *, limit: int = 100) -> list[SessionRecord]: ...
    async def delete(self, session_id: str) -> bool: ...   # True if a record was removed
```

#### `InMemoryStore`

```python
class InMemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()

    async def save(self, record):
        async with self._lock:
            self._records[record.session_id] = record.model_copy(deep=True)

    async def load(self, session_id):
        async with self._lock:
            r = self._records.get(session_id)
            return r.model_copy(deep=True) if r is not None else None
    # list / delete similarly
```

#### `FileStore`

```python
class FileStore:
    """One JSON file per session in `root`. Atomic writes via tmp + rename."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path_for(self, session_id: str) -> Path:
        # session_id is caller-provided; sanitize against path traversal
        if "/" in session_id or "\\" in session_id or session_id.startswith("."):
            raise ValueError(f"unsafe session_id: {session_id!r}")
        return self._root / f"{session_id}.json"

    async def save(self, record):
        path = self._path_for(record.session_id)
        tmp = path.parent / (path.name + ".tmp")    # avoids with_suffix dot edge cases
        async with self._lock:
            tmp.write_text(record.model_dump_json(), encoding="utf-8")
            os.replace(tmp, path)

    async def load(self, session_id):
        path = self._path_for(session_id)
        async with self._lock:
            if not path.exists():
                return None
            return SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    # list iterates root, delete unlinks
```

#### `Session` helper

```python
class Session:
    def __init__(
        self,
        orchestrator: Orchestrator,
        agent: SubAgent,
        store: MemoryStore,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    @property
    def messages(self) -> list[Message]: ...    # read-only view (returns a copy)

    async def send(self, message: str | Message) -> Message:
        msg = text("user", message) if isinstance(message, str) else message
        self._messages.append(msg)
        reply = await self._orch.run(self._agent, self._messages)
        self._messages.append(reply)
        await self._store.save(self._to_record())
        return reply

    @classmethod
    async def restore(
        cls,
        session_id: str,
        store: MemoryStore,
        orchestrator: Orchestrator,
    ) -> "Session":
        record = await store.load(session_id)
        if record is None:
            raise SessionNotFound(session_id)
        s = cls(orchestrator, record.agent, store, session_id=session_id, metadata=record.metadata)
        s._messages = list(record.messages)
        return s

    def _to_record(self) -> SessionRecord:
        return SessionRecord(
            session_id=self._session_id,
            agent=self._agent,
            messages=list(self._messages),
            metadata=self._metadata,
            created_at=self._created_at,        # preserved across saves
            updated_at=datetime.now(UTC),
        )
```

#### Tests

`tests/memory/test_record.py`:
1. JSON round-trip preserves all fields including agent.
2. `touched()` advances `updated_at` without mutating `created_at`.
3. Records with tool_use / tool_result content blocks survive a round-trip.

`tests/memory/test_store.py` (parametrized over `InMemoryStore`, `FileStore`):
1. `save` then `load` returns an equal record.
2. `load` of a missing ID returns `None`.
3. `list` returns all records up to `limit`.
4. `delete` removes the record and returns `True`; deleting a missing ID returns `False`.
5. Deep-copy isolation: mutating the loaded record does not affect the stored copy.
6. Concurrency, distinct IDs: `asyncio.gather` of 8 concurrent `save(record_i)` with 8 distinct session_ids — final state contains all 8 records.
7. Concurrency, same ID: `asyncio.gather` of 8 concurrent `save(...)` with the *same* session_id but different `metadata` — final state equals exactly one of the 8 inputs (no torn / merged record). This is the test that distinguishes "lock works" from "we got lucky".
8. `list` returns records sorted by `updated_at` descending across both backends.
9. (FileStore-only) Path-traversal session IDs (`../etc`) raise `ValueError`.
10. (FileStore-only) Stray `.tmp` files in the root are ignored by `list`.

`tests/memory/test_session.py`:
1. `send("hi")` appends a user text message, calls runner, appends assistant reply, saves a record with both messages.
2. Multiple `send()` calls accumulate; saved record has the full history.
3. `restore(session_id, store, orch)` returns a `Session` with the prior messages; subsequent `send` continues from there.
4. `restore` of a missing ID raises `SessionNotFound`.
5. Custom `metadata` round-trips through save/restore.

#### Verification

- `uv sync --extra dev --extra anthropic` — clean.
- `uv run pytest` — green.
- `uv run ruff check .` — clean.
- `uv run mypy` — clean (strict).

#### Out of scope (deferred)

- SQL-backed `MemoryStore`. Easy follow-up; the protocol is exactly the right shape.
- Streaming/incremental save (only full snapshots in MVP).
- Multi-process locking for `FileStore` (in-process locking only).
- Encryption-at-rest.

### Implementation log

- **Plan reviewed by advisor.** Four blocking/concrete items addressed before code:
  - Tmp filename uses `path.parent / (path.name + ".tmp")` instead of `path.with_suffix(".tmp")` to dodge dot-suffix interpretation edge cases (e.g. `session_id="v1.2"`).
  - Tightened the durability claim in the docstring: `os.replace()` gives "no torn writes," not full crash safety. Without `fsync()`, sudden power loss can leave either the old or new copy as the survivor; documented precisely.
  - Added a same-session-id concurrency test that gathers 8 concurrent `save()` calls with distinct metadata and asserts the final record matches exactly one of the inputs verbatim — the test that distinguishes "lock works" from "lock unnecessary."
  - `list()` sorts by `updated_at` descending in both `InMemoryStore` and `FileStore` so the contract is consistent regardless of backend.
- **`Session` is single-writer per session_id.** Two concurrent `restore(same_id)` instances racing `send()` is last-writer-wins; documented in the class docstring as a caller responsibility. Optimistic concurrency via etag/precondition deferred.
- **Path-traversal guard.** `FileStore._path_for()` rejects `session_id` values with `/`, `\`, leading `.`, or empty string. Test exercises all four.
- **Top-level re-exports.** `Session`, `SessionRecord`, `InMemoryStore`, `FileStore` available from `harness` directly.
- **Verification (final gates).**
  - `uv sync --extra dev --extra anthropic` — clean.
  - `uv run pytest` — 96 passed (was 69; +27 memory).
  - `uv run ruff check .` — clean.
  - `uv run mypy` — clean (strict, 26 source files).
- **Commit:** `feat(memory): SessionRecord + MemoryStore (InMemory/File) + Session helper` (TBD on push).

---

## Item 4 — Sandbox execution primitives

### Goal
Extend `harness.policy` with execution-side guards: filesystem path scoping
(allow/deny prefixes), a subprocess wrapper that respects policies and scrubs
the environment, and a network-deny-by-default helper. The aim is composable
primitives, not a full sandbox engine.

### Status
- Pending.

### Decisions
_(deferred)_

### Plan
_(deferred)_

### Implementation log
_(deferred)_

---

## Item 5 — Replay / eval harness

### Goal
Record a session's telemetry stream to JSONL, replay it against a deterministic
runner, and provide a small eval harness that runs the same prompt against
multiple runners and diffs the outputs.

### Status
- Pending.

### Decisions
_(deferred)_

### Plan
_(deferred)_

### Implementation log
_(deferred)_
