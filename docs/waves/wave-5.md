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
