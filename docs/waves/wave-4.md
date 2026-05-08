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

