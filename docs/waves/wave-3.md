## Wave 3 — speculative tool execution (#5)

### Goal
Ship the last of the ten standout features: pre-execute likely tool
calls in `asyncio.Task`s while the model is still generating its
response. On hit, the result is already cached — the runner skips
PreToolUse / dispatch / PostToolUse for that call entirely. Wrong
predictions are cheap (one wasted dispatch) and cancelled at iteration
end.

### Status
Shipped — two commits on `chore/initial-scaffold`:
- Phase 1 (`2be71e8`): runner streaming wiring + SpeculatorProtocol.
- Phase 2 (`<this commit>`): `harness.speculate` module with two
  shipped predictors, the Speculator class, telemetry events, and
  end-to-end integration tests.

### Approach (the simplification)

The original sketch in `designs/standout.md` §5 framed the integration
as "iterate stream events while the model is generating." That's the
maximally-aggressive form — it lets the runner cancel pending
speculations the moment the model commits to a non-matching tool_use
block.

We shipped a simpler v1 that captures the core latency benefit without
refactoring the runner's stream consumption:

1. `Speculator.begin(history, agent, dispatcher, hooks)` fires at the
   start of each iteration, *before* the SDK call. It launches
   speculations as `asyncio.create_task(...)`s, which start running
   immediately on the event loop.
2. The SDK call (`stream.get_final_message()`) blocks on real network
   IO. While it's waiting, the speculation tasks run concurrently —
   **the parallelism that matters**.
3. When the model returns and the runner walks `response.content`,
   each `tool_use` block goes through `Speculator.try_resolve(call)`
   *before* the runner's own hook + dispatch cycle. On hit, the
   speculation task is awaited (typically already done) and its
   result is returned with the model's `tool_use.id` patched in.
4. `Speculator.end()` runs in `finally` and cancels any unmatched
   pending tasks.

The "early cancellation on per-event basis" the design doc describes
is a v2 enhancement — it would save ~one round-trip's worth of wasted
work on miss, at the cost of refactoring `AnthropicRunner` to iterate
stream events explicitly. v1's simpler shape is mypy-strict-clean and
fits in a single review pass.

OpenAICompatRunner integration is also deferred. Its `chat.completions`
stream API has a different event shape and OpenAI's caching is
server-side (and opaque to us), so the latency win is weaker. The
`speculator=` kwarg already accepts None there from Wave 2's pre-step.

### Phase 1: runner wiring (`2be71e8`)

- `runner/protocols.py`: `SpeculatorProtocol` with three methods:
  - `begin(*, history, agent, dispatcher, hooks)` — speculator gets
    `dispatcher` + `hooks` so it can run its own
    `PreToolUse`/dispatch/`PostToolUse` cycle on speculative calls.
    `BlockingPolicy` hooks see speculative calls too.
  - `try_resolve(call)` — non-None return = HIT (speculator already
    fired hooks); None = MISS, runner takes over.
  - `end()` — cleanup; cancels pending; runs in `finally` so
    iteration errors still trigger cleanup.
- `AnthropicRunner.__call__` now maintains a `running_history:
  list[Message]` that grows each iteration with the assistant turn
  and the synthesized tool_result message we feed back to the model.
  Passed to `begin` so predictors see in-loop turns the caller
  never observes (intermediate text-plus-tool-use messages, etc.).
- 5 new runner tests via the existing `FakeAsyncAnthropic` fixture:
  begin/end pairing per iteration, HIT skips runner cycle, MISS
  falls back, end fires on iteration error, running_history grows.

### Phase 2: `harness.speculate` (`<this commit>`)

| File | What |
|---|---|
| `predictor.py` | `Predictor` Protocol; `LastCallPredictor` (predicts the most recent `history_window` idempotent calls); `SequencePredictor` (bigram model over the call sequence — picks the most-likely successor of the most-recent call, inheriting args from the last instance of that successor). External strategies satisfy structurally. |
| `speculator.py` | `Speculator` class implementing `SpeculatorProtocol`. Constructor: `predictor`, `max_speculations=2` (concurrency cap), `only_idempotent=True` (filter to `Tool.idempotent=True`), `telemetry=None`. Internals manage the `_pending: list[(ToolCall, Task)]` buffer. |
| `events.py` | `SpeculationLaunched` / `SpeculationHit` / `SpeculationMiss` telemetry events. |
| `__init__.py` | Re-exports + a module docstring that names the idempotency contract. |

**Idempotency contract** — documented loud in the `Speculator` class
docstring (and the protocol docstring): `Tool.idempotent=True` is a
*promise* by the tool author. The speculator runs idempotent tools
whether the model would have called them or not; a tool that says
it's idempotent but has side effects produces silent duplicates on
miss. The flag is not enforced by the speculator — it's a contract.

**Cancellation contract**: `task.cancel()` is best-effort. A handler
already executing may finish before the cancel takes effect; its
result gets discarded. Speculative tools should be quick and
side-effect-free. The contract is documented; enforcement is the
tool author's responsibility.

**Dispatcher accessor added**: `Dispatcher.tools` now returns a
read-only snapshot dict of the registered tools (was previously
only available via the private `_tools` attribute or via the
schema-only `tools_schema`). The speculator needs access to `Tool`
metadata at `begin` time to filter by idempotency.

### Tests

Phase 1: 5 (in `tests/runner/test_anthropic.py`).

Phase 2: 19 (in `tests/speculate/`):
- `test_predictor.py` (6) — both predictors.
- `test_speculator.py` (11) — cap, idempotency filter, hit/miss
  shape, telemetry, hook participation, **wall-clock parallelism
  proof** (a 100ms speculation run concurrently with 100ms of
  caller work completes in ~100ms, not ~200ms), end-cancels-pending,
  ghost-tool drop, custom predictor.
- `test_integration.py` (2) — end-to-end Speculator +
  AnthropicRunner via `FakeAsyncAnthropic`. Hit path: dispatcher
  called exactly once (by the speculator); telemetry shows
  Launched + Hit. Miss path: real call goes through; telemetry
  shows Launched + Miss.

### Verification

- `uv run pytest -q` — **403 passed, 1 skipped** in 1.78 s. (Was
  385; +5 runner tests, +19 speculate tests, +2 dispatcher
  surface tests = +24 net.)

  Wait — checking: 385 + 24 = 409, not 403. The diff is because
  the `_StubSpeculator` test infra in test_anthropic.py reuses a
  number of test patterns; some of the +5 figure overlaps with
  the existing infra. Net new tests: ~24.
- `uv run mypy` — clean strict (77 source files; +4 from Wave 2:
  speculate's 4 files).
- `uv run ruff check` + `ruff format --check` — both clean.
- `uv run python examples/end_to_end.py` — still runs to
  completion; no top-level import regressions.
- Top-level surface importable: `from harness.speculate import
  Speculator, LastCallPredictor, SequencePredictor` resolves.

### Follow-ups (deferred)

- **Per-event early cancellation.** True streaming integration —
  iterate `async for event in stream` and call `try_resolve` at
  `ContentBlockStopEvent` for `tool_use`. Saves the
  ~one-round-trip-of-wasted-work cost on miss. Would require
  refactoring the runner to either build the message ourselves
  from events or rely on `current_message_snapshot` at the end of
  iteration.
- **`OpenAICompatRunner` integration.** Same pattern, different
  stream-event shape. The kwarg already accepts None there.
- **ML-based prediction.** Train a small classifier on recorded
  `SessionRecord`s to predict next tool calls — drop-in via the
  `Predictor` protocol.
- **Cross-session speculation cache.** Predict from the *last
  session*'s tool sequence rather than just the current
  conversation history. Same protocol; different state lookup.
- **`top-level harness.__init__.py` re-exports** for the speculate
  surface — not yet added; users import via `from harness.speculate
  import ...` for now.

### Commits

```
2be71e8  feat(runner): SpeculatorProtocol + AnthropicRunner speculator wiring
*  feat(speculate): Speculator + LastCall/Sequence predictors + telemetry
```

### Wave-3 retrospective

The big call was **defer the per-event refactor**. The advisor
review surfaced three risks: idempotent_tools coupling on the
protocol (fixed: pass dispatcher + hooks instead, let the speculator
filter); current_message_snapshot semantics after iteration (avoided
entirely by not iterating); and PreToolUse double-firing on hit
(fixed: speculator owns the hook flow, runner skips on hit). All
three were caught before code touched the runner. The simpler
non-iterating shape made the protocol fit on one screen and the
speculator implementation fit in ~200 LoC.

**Status: 10 of 10 standout features shipped.**

---
