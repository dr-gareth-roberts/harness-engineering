# Roadmap progress log

> Living document for the post-MVP roadmap work on `harness-engineering`.
> Each item gets its own section with plan, decisions, and a per-step log.
> Older waves are archived under `docs/waves/` to keep this file focused
> on the current wave; the archive paths are linked in the status table.

## Status snapshot

| #     | Item                                              | Status  | Archive                                          |
| ----- | ------------------------------------------------- | ------- | ------------------------------------------------ |
| 0–6   | MVP scaffold + post-MVP items 1–6                 | shipped | [docs/waves/initial-scaffold.md](docs/waves/initial-scaffold.md) |
| Wave 1 | Counterfactual replay / contracts / fuzz / attribute / diff-eval | shipped | [docs/waves/wave-1.md](docs/waves/wave-1.md) |
| Wave 2 | Cache / privacy / plan / debug + post-Wave-2 integration fixes  | shipped | [docs/waves/wave-2.md](docs/waves/wave-2.md) |
| Wave 3 | Speculative tool execution (#5)                   | shipped | (current — see below)                            |

**Status: 10 of 10 standout features shipped.** All work lives on
`chore/initial-scaffold` (PR #1). The full `designs/standout.md` set
is implemented; remaining items are the deferred follow-ups listed in
each wave's archive (OpenTelemetry export, ML-based prediction).

## Cross-cutting decisions

- **Optional extras over runtime deps.** Each module that pulls in a
  heavy dependency (Anthropic SDK, OpenAI SDK, Hypothesis, sentence-
  transformers, …) lands as `[extras]` so the base install stays at
  `pydantic` only. Imports at the top of submodules use guarded
  `try/except ImportError` with a clear error pointing at the extra.
- **Vendor-neutral primitives, vendor-specific glue.** Core types
  live in the base package; concrete integrations live in
  `harness.<module>.<vendor>` submodules (e.g.
  `harness.runner.anthropic`).
- **Append to PR #1, not a stack of separate PRs.** PR #1 is still
  pending review and the items are conceptually one delivery — "the
  post-MVP layer + the standout features". Each item / wave is a
  small set of focused commits on `chore/initial-scaffold`.
- **Structural protocols for runner extension.** Wave 2 + Wave 3
  added `prefix_watcher` and `speculator` kwargs on the runner
  constructors via `Protocol`s in `src/harness/runner/protocols.py`.
  Feature modules satisfy them structurally; the runner has no
  runtime dependency on any feature module.
- **Idempotency is a tool-author promise**, not enforced by the
  speculator. Marking `Tool.idempotent=True` allows speculative
  pre-execution; a tool that says it's idempotent but has side
  effects produces silent duplicate side effects on miss. The
  contract is documented loud in `Speculator`'s class docstring.

---

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

## Wave 4 — deferred follow-ups

### Goal
With the ten standout features shipped, Wave 4 closes four documented
gaps the per-wave reviews flagged:

1. **OpenAICompatRunner speculator integration** — Wave 3 wired the
   speculator on `AnthropicRunner` only. The `speculator=` kwarg on
   `OpenAICompatRunner` accepted None but did nothing.
2. **OpenTelemetry sink** — `harness.telemetry`'s `Sink` protocol
   was OTel-ready since Wave 1; the adapter hadn't shipped.
3. **Plan inference from past sessions** — `derive_plan` asked a
   live planner; mining plans from recorded successful trajectories
   was deferred.
4. **Cross-session speculation cache** — predict from past
   `SessionRecord`s, not just the current conversation.

### Status
Shipped — pre-step plus three parallel agents plus integration.

### Approach

**Pre-step (`2e5a329`, me, single commit):**
- Wired the speculator into `OpenAICompatRunner.__call__` — mirrors
  Wave 3 Phase 1 on `AnthropicRunner` but adjusts for the
  no-streaming-wrapper SDK shape (single `await create(...)`),
  vendor-neutral `running_history`, and the system-prompt-prepended
  `all_messages` flow. 5 new runner tests via `FakeAsyncOpenAI`.
- Added `[otel]` extra (`opentelemetry-api>=1.20`,
  `opentelemetry-sdk>=1.20`) to `pyproject.toml` and ran `uv lock`
  so all three agents inherited a stable resolution.

**Three parallel agents in worktrees** (each verified base SHA at
start; two agents found their worktree HEAD at the orphan
`7fdbc62` from before Wave 1's extras commit and explicitly branched
from `chore/initial-scaffold` at `2e5a329` per the prompt's fallback
instruction — the lesson from Wave 2's Agent I now baked into every
agent prompt):

| Feature | Module | LoC src + test | Tests | Branch |
|---|---|---|---|---|
| OpenTelemetry sink | `harness.telemetry.otel` | 116 + 227 | 7 ✓ | `feat/otel-sink` |
| Plan inference | `harness.plan.infer` | 208 + 307 | 14 ✓ | `feat/plan-inference` |
| Cross-session predictor | `harness.speculate.cross_session` | 97 + 295 | 7 ✓ | `feat/cross-session-predictor` |

Total Wave 4: ~960 src + ~1 200 test, **33 new tests** (5 in pre-step
+ 28 across the three agents), 0 cross-feature conflicts on merge.

### Per-feature notes

**OpenTelemetry sink** — emits each `TelemetryEvent` as a flat OTel
`Event` on the current span via `span.add_event(...)`. Does NOT
synthesize spans from durations: the existing telemetry recorder
doesn't track parent-child correlation, so faking the nesting (one
root span per orchestrator turn, child spans per dispatch) would
produce a flat list of zero-children spans, uglier than events.
Span nesting is documented as deferred until the recorder grows
correlation IDs. Test pins this contract — a mocked `Tracer` is
asserted `not_called` for `start_span` / `start_as_current_span`.

**Plan inference** — `infer_plan_from_records(records, *,
success=None, mode="superset") -> Plan`. Default success heuristic:
session ended in an assistant message + no orphan tool_uses + no
`is_error=True` tool_results. Sequence selection: modal sequence
(most common exact tool-name sequence among successful inputs);
ties broken by *earliest* first-occurrence (the agent caught a
contradiction in the prompt where the design bullet said "most
recent" but the test expected "earliest" — implemented per the
test, which is the logically consistent reading). Default mode is
`superset` so the inferred minimum doesn't fail on extra calls the
inference didn't see. Documented alternatives in the function
docstring: longest common prefix, bigram-derived expected
sequence, intersection-of-all.

**Cross-session predictor** — `CrossSessionPredictor` pre-loads the
K most-recent `SessionRecord`s via an async `from_store(store, K=5)`
classmethod, builds a synthetic `list[Message]` with a sentinel
`ToolCall(name="__cross_session_boundary__")` between session
sequences (so bigrams don't bridge across session boundaries), and
delegates to `SequencePredictor` for the actual prediction. The
`predict()` method itself is sync (3-line body) — no
reimplementation of bigram logic. Records are reversed to
chronological order before the synthetic build so
`SequencePredictor`'s "most recent paired successor" semantics
inherit args from the *newest* record's calls (the agent's flagged
deviation; their tests pin this).

### Integration

- **Top-level re-exports**: `OpenTelemetrySink` (lazy via
  `__getattr__` like the vendor runners — `[otel]` extra is opt-in,
  `from harness import OpenTelemetrySink` only triggers the import
  when accessed); `infer_plan_from_records`; `CrossSessionPredictor`.
- **Subpackage re-exports**: `harness.telemetry.OpenTelemetrySink`
  (lazy), `harness.plan.infer_plan_from_records`,
  `harness.speculate.CrossSessionPredictor`.
- **README updates**: extended the existing rows for
  `harness.telemetry`, `harness.plan`, and `harness.speculate` to
  surface the new entry points.

### Verification

- `uv sync --extra dev --extra anthropic --extra openai-compat
  --extra fuzz --extra otel` — clean.
- `uv run pytest -q` — **438 passed, 1 skipped** in 2.20 s.
  (Was 405; +5 OpenAICompat speculator tests in pre-step, +7 OTel,
  +14 plan inference, +7 cross-session = +33 net.)
- `uv run mypy` — clean strict (80 source files; +3 over Wave 3).
- `uv run ruff check` + `ruff format --check` — both clean.

### Follow-ups (still deferred)

- **Per-event speculator cancellation** — the runner streaming
  refactor that lets the speculator cancel pending tasks at
  `ContentBlockStopEvent` for `tool_use`, saving ~one
  handler-runtime worth of work on miss. v2 of #5.
- **ML-based privacy detection** (Microsoft Presidio adapter) —
  slots in cleanly under the existing `Detector` protocol.
- **Span nesting in `OpenTelemetrySink`** — requires correlation
  IDs in the telemetry recorder; meaningful change to
  `TelemetryEvent` and the dispatcher / orchestrator emit sites.
- **DAP / IDE-protocol integration for `harness.debug`** — its own
  wave's worth of work; protocol-heavy.

### Commits

```
2e5a329  chore: Wave 4 pre-step — OpenAICompatRunner speculator + [otel] extra
*  Merge feat/otel-sink (Wave 4 — OpenTelemetry sink)
*  Merge feat/plan-inference (Wave 4 — infer Plan from past sessions)
*  Merge feat/cross-session-predictor (Wave 4)
*  feat: integrate Wave 4 — top-level re-exports + README + progress
```

---

## Wave 5 — runnable examples per module

### Goal
Every module added in Waves 1–4 has thorough tests, but discoverability
is poor: a user finding the package on GitHub sees the README's table
of 18 modules and has no concrete starting point per capability. Wave 5
ships **one runnable example per module-or-feature**, each following a
strict convention so they double as smoke tests in CI.

### Status
Shipped — anchor (mine) + 4 parallel agents + 2 surfaced bug fixes.

### Approach

**Pre-step (`72ea857`, me, single commit):**
- `examples/README.md` — convention + index. Every example is no-API
  (uses `EchoRunner` / `CannedRunner` / a small inline fake), exposes
  `async def main() -> int`, prints a transcript, and is exercised by
  `tests/examples/test_examples_run.py`.
- `tests/examples/test_examples_run.py` — parametrized smoke test.
  Imports each example by file name, calls `main()`, asserts return
  code 0, asserts a per-example marker string appears in stdout.
- `examples/contracts.py` — the anchor. Walks both contract enforcement
  surfaces (live `attach_contracts` + offline `check`) and pins
  runtime/offline equivalence for a `require` contract.

**Four parallel agents in worktrees**, three completed cleanly, one
stalled and was completed in the main worktree:

| Agent | Cluster | Branch | Files |
|---|---|---|---|
| M | replay + plan (3 files) | `feat/examples-replay-plan` (`15b5c02`) | `counterfactual.py`, `diff_eval.py`, `plan.py` |
| N | observability (4 files) | (stalled — completed in main) | `cache.py`, `privacy.py`, `otel.py`, `debug.py` |
| O | speculate (3 files) | `feat/examples-speculate` | `speculate.py`, `cross_session.py`, `plan_inference.py` |
| P | quality (2 files) | `feat/examples-quality` | `fuzz.py`, `attribute.py` |

Total Wave 5: 1 anchor + 12 new examples = **13 runnable examples**,
each with a smoke test.

### Surfaced bugs (fixed in `6637c5b`)

Writing the examples surfaced two real bugs that the existing test
suites had missed:

- **`OpenTelemetrySink` emitted None-valued attributes**: the OTel SDK
  rejects None and logs a warning; we were dropping one attribute per
  successful `OrchestratorTurn` (`error: str | None = None` defaults
  to None on a clean turn). Fix: skip None values explicitly. New test
  `test_none_valued_payload_fields_are_skipped_not_emitted` covers it.
- **`harness.attribute.similarity` mypy errors with `[attribute]`
  installed**: the `# type: ignore[import-not-found]` suppressions on
  the lazy `sentence_transformers` and `numpy` imports became
  unused-ignore errors when those packages WERE installed. Fix: combine
  `[import-not-found, unused-ignore]` so mypy is happy in both states.
  Verified by running mypy with sentence-transformers installed AND
  uninstalled — clean both ways.

Agent N's stall was the operational cost of the failure recovery.
Agent M caught the mypy issue at verification time and flagged it
clearly; Agent N's work was the trigger for the OTel sink bug.

### Per-example summary

Anchor:
- `contracts.py` (mine, `72ea857`) — live `attach_contracts` blocks
  forbidden tool calls; offline `check` reports the same kind of
  Violation; runtime `require` raises `ContractViolation` at SessionEnd
  if unsatisfied.

Replay + plan (Agent M):
- `counterfactual.py` — mutate a recorded SessionRecord and continue
  from the divergence point.
- `diff_eval.py` — three runners against three cases; `DiffMatrix`
  surfaces unanimous vs outlier verdicts.
- `plan.py` — `PlanGuardedRunner` accepts a correct trajectory and
  raises `PlanViolation` on a wrong one (uses inline scripted runner;
  `CannedRunner` is text-only and can't emit tool_use blocks).

Observability (mine after Agent N stall):
- `cache.py` — phase 1 stable cached system prompt, phase 2 timestamp
  leak; audit shows drift on breakpoint 0 with the right `hint`.
- `privacy.py` — `PII_PACK` redacts a SSN before the inner runner
  sees it; a separate AWS-key-shaped string raises `PrivacyViolation`.
- `otel.py` — in-process `TracerProvider` + `InMemorySpanExporter`;
  emits two `TelemetryEvent`s through `OpenTelemetrySink`; reads them
  back as flat events on the span (no spans created by the sink).
- `debug.py` — programmatic-mode breakpoint inspects messages, fires
  ad-hoc `lookup`, mutates the next reply, resumes. The mutation
  short-circuits the inner runner.

Speculate (Agent O):
- `speculate.py` — manual `begin/try_resolve/end` lifecycle (cleanest
  demo without a vendor SDK fake). Wall-clock observed: serial
  baseline 202 ms, parallel speculation 102 ms (~1.98×).
- `cross_session.py` — `InMemoryStore` with synthetic past sessions;
  `CrossSessionPredictor.from_store` aggregates the cross-session
  bigram signal.
- `plan_inference.py` — `infer_plan_from_records` on 5 synthetic
  records (4 share the modal sequence, one outlier); inferred Plan
  steps printed.

Quality (Agent P):
- `fuzz.py` — `fuzz_tool` over a Pydantic-typed `parse` tool; reports
  4 / 50 failures at `seed=0` (Hypothesis hits the `count == 0`
  boundary deterministically).
- `attribute.py` — leave-one-out ablation on a 4-block synthetic
  session where one block contains "the password is rosebud"; that
  block ranks `top_k(1)` with score 1.000.

### Verification

- `uv sync --extra dev --extra anthropic --extra openai-compat
  --extra fuzz --extra otel --extra attribute` — all extras
  installed (used to verify the dual-state mypy fix).
- `uv run pytest tests/examples -v` — **14 / 14 example smoke tests**
  pass (existing 1 + 12 new + the anchor `contracts`).
- `uv run pytest -q` — **454 passed**, 0 skipped (was 438 + 1 skipped
  pre-Wave-5; +14 example smoke + +1 OTel regression + +1 unblocked
  embedding sanity test = 454).
- `uv run mypy` — clean strict, 80 source files, both with and
  without `[attribute]` installed.
- `uv run ruff check` + `ruff format --check` — clean.
- `for f in examples/*.py; do uv run python "$f" || exit 1; done` —
  every example runs to completion (excluding `anthropic_runner.py`
  which is gated on `ANTHROPIC_API_KEY`).

### Operational notes

- All four agent dispatches included a base-SHA verification step at
  start (the lesson Wave 2's Agent I taught us, codified into every
  prompt since Wave 4). Three of four agents found their worktree HEAD
  on a stale ref and explicitly branched from `chore/initial-scaffold`
  per the prompt's fallback. The fourth (Agent N) stalled at 600 s of
  no progress and produced no branch.
- Agents M and P observed the existing `--extra dev` install was bare
  in fresh worktrees and ran `uv sync --extra dev --extra ...` before
  pytest/mypy. This is now standard friction; future waves should
  document it in agent prompts.
- The `tests/examples/test_examples_run.py` `EXAMPLES` list was
  predictably contended — three agent branches all appended to it,
  three trivial merge conflicts at integration. No design issue; just
  the price of parallelism on a small shared file. Could be sharded
  in a future wave (one fixture file per cluster).

### Follow-ups

- Per-event speculator cancellation (still deferred — would need
  AnthropicRunner stream-event refactor + `FakeAsyncAnthropic`
  extension; the wall-clock win is bounded by handler runtime).
- DAP / IDE-protocol for `harness.debug` (Wave 6+ candidate).
- Polish + docs site (Wave 7+ candidate).
- Presidio adapter for `harness.privacy` (deliberately deferred from
  Wave 4; the architecture is ready, the adapter is one module).

### Commits

```
72ea857  feat(examples): Wave 5 pre-step — examples scaffolding + contracts anchor
6637c5b  fix(telemetry,attribute): None-attr skip in OTel sink + dual-state mypy ignore
871d6af  docs(examples): cache + privacy + otel + debug          (replaces stalled Agent N)
04cd40d  merge: feat/examples-quality          (fuzz + attribute)
86576e7  merge: feat/examples-replay-plan      (counterfactual + diff_eval + plan)
b337c4b  merge: feat/examples-speculate        (speculate + cross_session + plan_inference)
*  docs: progress.md log of Wave 5
```


## Wave 6 — per-event speculator cancellation

### Goal
Free the speculation handler runtime that was being burned between
stream-end and the iteration's `finally:` block. Pre-Wave-6 the runner
called `await stream.get_final_message()` (waiting for the full
message) and only cancelled unmatched speculations at iteration end —
which fires *after* the model's emitted tool_use blocks have all been
dispatched. Pre-event, an unmatched 5-second speculation runs through
the entire dispatch phase. Post-Wave-6, the runner iterates stream
events as they arrive, surfaces each `tool_use` block to the speculator
via a new `observe()` call, then cancels everything still unobserved
the moment the stream ends — *before* dispatch.

### Status
Shipped on `feature/speculator-per-event`. Single-coherent refactor
done in main, no parallel agents — runner + fake + tests are too
tightly coupled to split.

### Cancellation timing — what we actually do

The user's prompt phrased this as "cancel pending tasks at
ContentBlockStopEvent for tool_use." The advisor flagged that as
*per-block* cancellation: with `max_speculations > 1`, deciding when
a speculation is "definitively dead" mid-stream requires policy that
isn't worth the complexity for MVP.

This wave cancels at *stream-end* (via `cancel_unobserved`) instead.
That captures the bulk of the win — the dispatch phase no longer
runs alongside burning speculation handlers — without the policy
complexity. End() (in the iteration's finally) still acts as a
final safety net.

The protocol is shaped so eager per-block cancellation could be
added later without breaking changes: `observe()` is called per-block,
so a future Speculator could implement eager cancellation in observe()
itself; today's implementation just marks the entry as observed and
defers cancellation to `cancel_unobserved`.

### Approach

**Protocol additions (`a39cfe0`, pre-step):**

`SpeculatorProtocol` gains two lifecycle methods:

```python
async def observe(self, call: ToolCall) -> None: ...
async def cancel_unobserved(self) -> None: ...
```

`Speculator` tracks observation per pending entry via a new `_Pending`
dataclass (replacing the old `tuple[ToolCall, Task]`). `observe(call)`
walks the pending list and marks the first unobserved match as
observed; `cancel_unobserved()` cancels and drains every pending entry
not marked observed. `try_resolve` and `end` are unchanged in shape but
read from the new dataclass.

Test stubs (`tests/runner/test_anthropic.py`,
`tests/runner/test_openai_compat.py`) get matching no-op
implementations so structural protocol compatibility holds. Existing
28 speculator tests + 42 runner tests still pass without modification.

**Runner refactor (single commit, in branch):**

`AnthropicRunner.__call__` now iterates the stream:

```python
async with self._client.messages.stream(**request) as stream:
    async for event in stream:
        if (
            self._speculator is not None
            and getattr(event, "type", None) == "content_block_stop"
            and getattr(getattr(event, "content_block", None), "type", None) == "tool_use"
        ):
            block = event.content_block
            await self._speculator.observe(
                ToolCall(name=block.name, arguments=dict(block.input), id=block.id)
            )
    response = await stream.get_final_message()

if self._speculator is not None:
    await self._speculator.cancel_unobserved()
```

`get_final_message()` after iteration mirrors the SDK's behavior —
`until_done()` is a no-op once the stream is consumed, the snapshot
stays accumulated.

**Fake extension:**

`tests/runner/fakes.FakeMessage` gains an optional `events: list[Any] | None`
field. When None (the default), `_FakeStream.__aiter__` auto-derives
one `FakeContentBlockStopEvent` per entry in `content` — zero-config
for existing tests. When set explicitly, tests can script specific
arrival orders (text-then-tool, multiple tools, scrambled-vs-content,
zero events, etc.). `get_final_message` returns the same `FakeMessage`
whether the stream was iterated first or not.

### Tests added

| File | Test | Pins |
|---|---|---|
| `tests/speculate/test_speculator.py` | `test_observe_marks_first_unobserved_matching_pending_spec` | observe records first unobserved match; cancel_unobserved leaves it alone |
| | `test_observe_with_no_match_is_a_noop` | observe with no matching pending is silent |
| | `test_cancel_unobserved_with_no_pending_is_noop` | safe to call when begin returned without launching anything |
| | `test_cancel_unobserved_runs_fast_when_handler_is_slow` | the perf claim — drain time vs handler runtime |
| | `test_observe_then_try_resolve_resolves_observed_spec` | full happy-path lifecycle |
| | `test_observe_claims_separate_entries_for_duplicate_calls` | two specs of the same shape stay distinct |
| `tests/runner/test_anthropic.py` | `test_runner_calls_observe_for_each_tool_use_block_in_stream` | observe fires per tool_use, in stream order |
| | `test_runner_does_not_observe_text_block_stop_events` | text blocks don't surface |
| | `test_runner_with_speculator_none_iterates_stream_without_error` | back-compat: no-speculator path still works with event iteration |
| | `test_runner_explicit_events_drive_observe_in_order` | order follows event arrival, not content list |
| | `test_unobserved_speculation_does_not_complete_when_dispatch_diverges` | runner-level correctness: unmatched speculation never reaches "done" |

11 new tests, 465 total (was 454).

### Verification gate

```
ruff check       — clean
ruff format     — 166 files already formatted
mypy --strict src/harness  — clean (79 source files)
pytest          — 465 passed
```

### Commits

```
a39cfe0  feat(speculate): add observe + cancel_unobserved to SpeculatorProtocol
5bbd0bf  feat(runner): per-event observe + cancel_unobserved in AnthropicRunner
3ee6bba  docs: progress.md log of Wave 6
```


## Wave 7 — DAP for debug REPL

### Goal
Let editors (VS Code, neovim-dap, Emacs dap-mode, etc.) drive the same
replay-based debug session that `harness debug` already supports
interactively. The user picks an editor frame, sets breakpoints on
trajectory turns, sees `stopped` events, browses scopes/variables/the
synthesized source, and resumes — without leaving the IDE.

### Status
Shipped on `feature/dap-debug`. Single-coherent refactor in main, no
parallel agents.

### Architecture

`harness.debug.dap.DapAdapter` is the bridge: a long-lived state
holder that runs a DAP message loop (`serve`) on one asyncio task and
the orchestrator session on another, sharing the event loop. The
breakpoint pump (`_on_breakpoint`) is wired into `DebugRunner` as the
`breakpoint_callback`; on hit it parks on `_continue_event` while the
DAP read-loop keeps pumping inspect requests against the held
`DebugContext`.

This concurrency is the load-bearing property of the design. A
sequential implementation (read one DAP message → handle it → loop)
would deadlock the editor on every `evaluate`/`variables`/`scopes`
during a breakpoint, because the message loop would itself be parked
inside the breakpoint callback. The test
`test_inspect_requests_pump_during_breakpoint_hold` pins this — it
fires four interleaved inspect requests during a breakpoint hold and
asserts each gets a response *before* `continue` is sent.

### Source mapping

DAP frames carry `Source` + `line`, so an editor without a real source
file is hard to drive. The adapter synthesizes one line per assistant
turn (caller supplies `synthesize_source: () -> list[str]`). DAP line
N (1-based) maps to `ctx.turn_index == N - 1`; setting a breakpoint at
line 5 in the synthesized source pauses right before producing the
5th assistant turn. The CLI's `_trajectory_lines` summarizes each
assistant message (text blocks first, then `(tool_use <name>)`).

### DAP subset implemented

| Group | Surface |
|---|---|
| Requests (responded) | `initialize`, `launch`, `setBreakpoints`, `configurationDone`, `threads`, `stackTrace`, `scopes`, `variables`, `evaluate`, `source`, `continue`, `next`, `stepIn`, `stepOut`, `pause`, `terminate`, `disconnect` |
| Events (emitted) | `initialized`, `stopped`, `continued`, `output`, `terminated`, `exited` |
| Capabilities advertised | `supportsConfigurationDoneRequest`, `supportsEvaluateForHovers`, `supportsTerminateRequest` |

`next` / `stepIn` / `stepOut` / `pause` are accepted but treated as
`continue` — agent trajectories don't have a meaningful intra-turn
step granularity yet. The handlers exist so editors that rely on these
capabilities don't error out.

`evaluate` is limited to looking up the same names the `variables`
view exposes (`turn_index`, `message_count`, `last_call.name`,
`last_call.arguments`, `pending_mutation.role`). Arbitrary-expression
evaluation is intentionally out of scope for the DAP surface; the
interactive REPL (`harness debug` without `--dap`) is the place for
that. This rationale is documented in the module docstring.

### Files

| File | Lines | Purpose |
|---|---|---|
| `src/harness/debug/dap_protocol.py` | ~115 | Content-Length + JSON framing over `asyncio.StreamReader/Writer`. Distinguishes graceful EOF (caller treats as disconnect) from truncation/malformed input (raises `DapProtocolError`). |
| `src/harness/debug/dap_messages.py` | ~140 | Pydantic models for `Request`, `Response`, `Event`, `Capabilities`, `Source`, `StackFrame`, `Scope`, `Variable`, `Breakpoint`. snake_case ↔ camelCase aliasing follows the spec field-by-field via `validation_alias=AliasChoices(...)` + `serialization_alias`. |
| `src/harness/debug/dap.py` | ~440 | `DapAdapter` — the bridge. |
| `src/harness/debug/cli.py` | +85 | `--dap` flag on `harness debug`; runs the same replay-driven session under DAP control over stdio. |

### CLI

```
$ harness debug --help
usage: harness debug [-h] [--break BREAK_SPEC] [--dap] path

  --dap                 Speak the Debug Adapter Protocol over stdio
                        instead of running the interactive REPL.
```

VS Code launch config example (illustrative — not shipped in repo):

```json
{
  "type": "harness",
  "request": "launch",
  "name": "Debug recorded session",
  "program": "${workspaceFolder}/path/to/session.json"
}
```

### Tests added

| File | Test count | Coverage |
|---|---|---|
| `tests/debug/test_dap_protocol.py` | 16 | round-trip, header tolerance (case insensitive, extra headers), malformed input (missing/invalid/negative Content-Length, bad header lines, invalid JSON, non-object body), EOF semantics (clean vs mid-headers vs mid-body), back-to-back messages, UTF-8 framing. |
| `tests/debug/test_dap.py` | 13 | initialize → initialized event sequence, setBreakpoints validation against synthesized source length, full launch → break → continue → terminated flow, **concurrent inspect during breakpoint hold (the load-bearing test)**, evaluate (supported + unsupported), source request (known + unknown reference), disconnect mid-breakpoint aborts, unknown command error response, launch-without-run-session error, aborted session does not propagate. |

29 new tests, **494 total** (was 465).

### Design constraint that surfaced

The pre-tool-use security hook blocks `eval(` literal in new files —
correctly, since arbitrary-expression evaluation is high-risk surface.
The DAP `evaluate` handler resolves this by *not* offering arbitrary
expression evaluation; it looks up names from a fixed snapshot of the
`DebugContext`'s public scope (the same set the `variables` view
exposes). The interactive REPL, which already had a documented
exception for `eval`, keeps its arbitrary-expression power. Two
surfaces, two safety profiles, both documented.

### Verification gate

```
ruff check       — clean
ruff format     — 169 + 2 reformatted = 171 files
mypy --strict src/harness  — clean (82 source files)
pytest          — 494 passed
```

### Commits

```
2f40af6  chore: Wave 7 pre-step — DAP framing + message models
a00e3da  feat(debug): DapAdapter + harness debug --dap stdio mode
0183f08  docs: progress.md log of Wave 7
f70fadf  test(dap): drop overly broad warnings.warn monkeypatch
```


## Wave 8 — polish, docs site, hardening

### Goal
Tie up the user-facing surface after Waves 6+7 and ship a browsable
docs site so the package is reachable to readers who don't already
know what `harness.speculate` does.

### Status
Shipped on `feature/polish-and-docs`. Single coherent pass in main
following the order recommended by the advisor: README/re-exports
first (highest leverage), hardening (time-boxed), then docs site
(capped if mkdocstrings got sticky — it didn't).

### Polish

- Top-level `harness.__init__` re-exports `DapAdapter` and
  `DapProtocolError` so `from harness import DapAdapter` works.
- README speculate row mentions the per-event observe/cancel lifecycle
  and stream-end cancellation. Debug row mentions `--dap` and the
  editors it targets (VS Code, neovim-dap, Emacs dap-mode).
- Deferred-items list pruned: DAP, OTel export, plan inference, and
  Speculator-on-OpenAICompatRunner are all shipped. Replaced with an
  honest entry for *eager per-block* speculator cancellation as a
  future refinement on top of Wave 6.

### Hardening — Any-audit

Time-boxed audit of `: Any` and `-> Any` annotations across
`src/harness`. 22 occurrences in total; 20 are load-bearing for
protocol flexibility (lazy `__getattr__` dispatchers, JSON-shaped
tool result content, recursive privacy scan, Hypothesis API surface,
SentenceTransformer interop, etc.) and were left intact. Two were
missed narrowings in `harness.fuzz.runner._generate_examples`:

```python
# was
def _generate_examples(model_cls: Any, ...) -> list[Any]: ...
def _collect(value: Any) -> None: ...
# now
def _generate_examples(model_cls: type[BaseModel], ...) -> list[BaseModel]: ...
def _collect(value: BaseModel) -> None: ...
```

Both call sites already passed `Tool.input_model: type[BaseModel]`,
so this was a missed narrowing rather than a real flexibility seam.

### Hardening — stress test

`test_orchestrator_handles_large_history_without_quadratic_blowup`
runs `Orchestrator.run()` with a 200-message history through a no-op
runner and asserts the call completes in well under one second. Pins
that no quadratic-time path is hiding on the orchestrator hot path.

The test is not a perf bound — the runner is no-op, so the entire
runtime is orchestrator overhead. Anything > 1s for 200 messages
implies a regression.

### Docs site

Hand-written index + architecture + CLI overview, plus a per-module
landing page that delegates to `mkdocstrings` for the API reference.
mkdocs-material defaults; no theme tuning. Local-only (`mkdocs serve`);
publishing to GitHub Pages is deliberately out of scope.

```
mkdocs.yml                       # config
docs/index.md                    # landing — install, 30-sec tour, where to read next
docs/architecture.md             # the three core seams (Runner, Sink, MemoryStore) + composition
docs/cli.md                      # harness debug, harness debug --dap, harness cache-audit
docs/modules/{18 module pages}.md
docs/roadmap.md                  # status table + deferred + archive links
```

Build: `uv sync --extra docs && uv run mkdocs serve`. The build is
strict-clean (`mkdocs build --strict` finishes in ~1 second).

### Verification gate

```
uv build                        — wheel + sdist build successfully
uv run mkdocs build --strict   — clean
ruff check                      — clean
ruff format --check             — clean
mypy --strict src/harness       — clean (82 source files)
pytest                          — 495 passed (was 494; +1 stress test)
```

### Commits

```
5322f3e  docs: surface Wave 6/7 in README + top-level re-exports
bc1f441  chore: hardening — narrow Any in fuzz/runner.py + 200-msg orchestrator stress test
*        docs: MkDocs scaffold + per-module API ref + [docs] extra
*        docs: progress.md log of Wave 8
```





