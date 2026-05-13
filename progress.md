# Roadmap progress log

> Living document for the post-MVP roadmap work on `harness-engineering`.
> Each wave gets its own section with plan, decisions, and a per-step log.
> Older waves are archived under `docs/waves/` to keep this file focused
> on the current wave; the archive paths are linked in the status table.

## Status snapshot

| #     | Item                                              | Status  | Archive                                          |
| ----- | ------------------------------------------------- | ------- | ------------------------------------------------ |
| 0–6   | MVP scaffold + post-MVP items 1–6                 | shipped | [docs/waves/initial-scaffold.md](docs/waves/initial-scaffold.md) |
| Wave 1 | Counterfactual replay / contracts / fuzz / attribute / diff-eval | shipped | [docs/waves/wave-1.md](docs/waves/wave-1.md) |
| Wave 2 | Cache / privacy / plan / debug + post-Wave-2 integration fixes  | shipped | [docs/waves/wave-2.md](docs/waves/wave-2.md) |
| Wave 3 | Speculative tool execution (#5)                   | shipped | [docs/waves/wave-3.md](docs/waves/wave-3.md) |
| Wave 4 | OTel sink / plan inference / cross-session predictor / OpenAICompat speculator | shipped | [docs/waves/wave-4.md](docs/waves/wave-4.md) |
| Wave 5 | Runnable example per module                       | shipped | [docs/waves/wave-5.md](docs/waves/wave-5.md) |
| Wave 6 | Per-event speculator cancellation (`observe` / `cancel_unobserved`) | shipped | [docs/waves/wave-6.md](docs/waves/wave-6.md) |
| Wave 7 | DAP for debug REPL (`harness debug --dap`)        | shipped | [docs/waves/wave-7.md](docs/waves/wave-7.md) |
| Wave 8 | Polish + docs site + hardening                    | shipped | [docs/waves/wave-8.md](docs/waves/wave-8.md) |
| Wave 9 | CI/CD + governance + housekeeping                  | shipped | [docs/waves/wave-9.md](docs/waves/wave-9.md) |
| Wave 10 | Vendor runner parity + robustness                 | shipped | [docs/waves/wave-10.md](docs/waves/wave-10.md) |
| Wave 11 | Deeper observability + verification               | shipped | [docs/waves/wave-11.md](docs/waves/wave-11.md) |
| Wave 12 | Modality + Files API                              | shipped | [docs/waves/wave-12.md](docs/waves/wave-12.md) |
| Wave 13a | Streaming output (`Orchestrator.run_stream`)     | shipped | [docs/waves/wave-13a.md](docs/waves/wave-13a.md) |
| Wave 13b | Speculator + privacy + DAP polish               | shipped | (current — see below)                            |

**Status: 10 of 10 standout features shipped, all Wave 8 audit gaps
addressed.** The forward plan from `0.2.0` to `1.0` lives in
[`docs/plan.md`](docs/plan.md). Waves 9–13b shipped (26 of 28 gaps
cleared); only #19 (cassette pattern, gated on real-API keys) and
the Wave-12 Files-API upload helper (also gated on keys) remain —
both flagged in their wave entries as deferred until credentials
are available. Ready for `1.0`.

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
- **One PR, multiple waves**. Waves 1–8 all landed on
  `chore/initial-scaffold` (PR #1) as conceptually one delivery —
  "the post-MVP layer + standout features." Waves 9+ branch
  individually off `main` and land separately.

---


## Wave 13b — Speculator + privacy + DAP polish

### Goal
The final wave to `1.0`. Five user-visible items, each independent:
DAP `pause` actually pauses; `next`/`stepIn`/`stepOut` drive distinct
semantics; `evaluate` gains opt-in arbitrary-expression mode;
speculator cancels eagerly in the simple case; Presidio joins the
privacy detector roster.

### Status
Shipped on `feature/wave-13b-final`. Five gaps cleared (#1, #2, #15,
#16, #17). Two items still deferred to a future wave (and documented
honestly): #19 (cassette pattern) and the Files-API `upload_file`
helper — both gated on real-API keys we don't have here.

### What landed

| # | Item | Implementation |
| --- | --- | --- |
| 16 | DAP `pause` | `DapAdapter._on_pause` sets `_pause_requested = True`. The `break_on_predicate` checks the flag first; if set, fires (and clears the flag) so the next runner invocation pauses unconditionally. Editor's pause button now works for the first time. |
| 15 | DAP step semantics | `DapAdapter` gains a `_step_mode` field set by `_on_next` / `_on_stepIn` / `_on_stepOut`. `break_on_predicate` reads it: `step_over` and `step_out` both pause before the next runner invocation (per-turn granularity). `step_in` is treated as `step_over` until the runner exposes finer "before-next-tool-handler" granularity — documented as a follow-up to enrich. The flag is consumed (cleared) on first read so subsequent runner invocations honor the per-turn breakpoints normally. |
| 17 | DAP `evaluate` parity | `DapAdapter.__init__` gains `allow_evaluate: bool = False` (constructor opt-in) and `_on_launch` reads `args["allowEvaluate"]` (per-launch opt-in). When on, `_on_evaluate` routes through the new `harness.debug.repl.evaluate_in_context` helper — the same code path the REPL's `inspect` command uses. Default behavior (restricted to `variables`-view names) stays for editors that didn't opt in. The opt-in is documented as carrying the same security trade-off as the REPL's `inspect`: arbitrary Python evaluation against `ctx`, only reachable in an opt-in debug session. |
| 2  | Eager per-block speculator cancellation | `Speculator.observe` gains an end-of-method check: when `max_speculations == 1` and the observed call didn't match, the lone pending speculation is definitively a miss; cancel it now instead of waiting for `cancel_unobserved` at stream-end. For `max_speculations > 1`, keep the existing stream-end policy — correctness with multiple pending entries requires policy that's not worth the complexity. New test pins the timing: a 10s slow handler is cancelled within ms of the first non-matching `observe`. |
| 1  | Presidio adapter | New `harness.privacy.presidio` module with `PresidioDetector` (wraps `presidio_analyzer.AnalyzerEngine` behind the existing `Detector` Protocol) and `build_pii_pack()` (preconfigured outbound-only pack covering common PII entities — PERSON, EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, etc.). `[privacy-ml]` extra in `pyproject.toml` (`presidio-analyzer>=2.2`). Lazy-import in `__init__` raises `ImportError` with a clear `[privacy-ml]` install hint when the extra isn't present. `harness.privacy.repl.evaluate_in_context` is the new module-level helper that both REPL and DAP route through (Wave 13b #17). |

### Tests added

| File | Count | Coverage |
| --- | --- | --- |
| `tests/debug/test_dap.py` | +6 | `pause` sets the flag and `break_on_predicate` consumes it; `next` sets `step_over` then resumes; `step_over` predicate fires once and clears; default `evaluate` rejects arbitrary expressions; opt-in `evaluate` returns arbitrary Python results; opt-in `evaluate` surfaces SyntaxError as a failed response. |
| `tests/speculate/test_speculator.py` | +2 | eager cancellation under `max_speculations=1` (10s handler cancelled in ms); no eager cancellation under `max_speculations > 1`. |
| `tests/privacy/test_presidio.py` | 9 | structural Detector-Protocol conformance; `RecognizerResult` → `Detection` conversion; `score_threshold` / `entities` / `language` / `direction` / `action` propagation; `build_pii_pack` shape; lazy-import error when `[privacy-ml]` is missing. |

17 new tests, **565 total** (was 548). Coverage stays at **89%**
(gate 85%).

### Verification gate

```
ruff check                       — clean
ruff format --check             — 178 files clean
mypy --strict src tests         — clean (163 source files)
pytest --cov=harness            — 565 passed, 1 skipped, 89% coverage
mkdocs build --strict           — clean
uv build                         — wheel + sdist build cleanly
```

### Deferred (still)

- **#19 cassette pattern for vendor SDK shape drift** — needs real-API
  keys to record. The `FakeAsync*` infrastructure already covers
  scripted-response replay; the recording step is the missing piece.
- **Files-API `upload_file` helper** — same constraint: needs API
  keys for an end-to-end smoke. Users can call
  `client.beta.files.upload(...)` directly via the SDK today; the
  harness side handles the resulting `file_id` correctly (Wave 12).

These two items from the original 28-gap audit didn't land; both are
honestly gated on credentials this environment doesn't have. The
"1.0 ready" line is met.

> Wave 13b initially shipped `step_in` aliased to `step_over` (see
> row #15 above) and listed the finer "step into the tool handler"
> granularity as a follow-up. The post-Wave-13b audit batch (1.3.0,
> M3.6) wired the frame-aware semantics via
> `DapAdapter.attach_hooks(hooks)`: `step_in` now pauses at the next
> `PreToolUse`, `step_out` from a tool frame pauses after the
> current `PostToolUse`. See CHANGELOG `[1.3.0]` and
> `src/harness/debug/dap.py:33-69`.

### Commits

```
*  chore(progress): rotate Wave 13a to docs/waves/
*  feat(debug): DAP pause + step semantics + evaluate opt-in
*  feat(debug,repl): evaluate_in_context helper shared by REPL + DAP
*  feat(speculate): eager per-block cancellation when max_speculations=1
*  feat(privacy): Presidio detector + build_pii_pack under [privacy-ml]
*  docs: CHANGELOG + progress.md log of Wave 13b
```
