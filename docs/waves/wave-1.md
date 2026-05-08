# Wave 1 — five standout features in parallel (archived)

> Archived from `progress.md` after Wave 3 shipped. Wave 1 implemented
> features #1, #2, #4, #7, #8 from `designs/standout.md` concurrently
> via five dispatched agents in isolated git worktrees. The
> implementation log below preserves the approach, per-feature
> summary, integration notes, follow-ups (most of which were closed
> by the post-Wave-2 fixes archived in `wave-2.md`), and the commit
> ledger. The features themselves live in `src/harness/{contracts,
> fuzz, attribute, replay/counterfactual, replay/diff_eval}` and are
> covered by their tests under `tests/`.

## Wave 1 — five standout features in parallel

### Goal
Implement five of the ten features from `designs/standout.md` (#1 Counterfactual replay, #2 Behavioral contracts, #4 Tool surface fuzzing, #7 Causal provenance via ablation, #8 Differential cross-provider runs) concurrently. Each feature lives in its own subpackage; none touch the runner-streaming internals or the (still-unbuilt) CLI, so they can ship independently of the runner-invasive features (#3, #5, #6, #10) and the contract-derived #9.

### Status
Shipped — five `feat/<name>` branches merged into `chore/initial-scaffold` with `--no-ff` merge commits, plus one integration commit on top.

### Approach
Five parallel `general-purpose` agents, one per feature, each in an isolated git worktree (auto-created via the agent runner's `isolation: "worktree"` mode). Each agent received:

- The exact spec section copied from `designs/standout.md` as their binding source-of-truth.
- An explicit "do not modify" list covering shared files: top-level `src/harness/__init__.py`, `pyproject.toml`, `uv.lock`, `README.md`, `progress.md`, `designs/standout.md`, and (for #1 / #8 which both touch `replay/`) `src/harness/replay/__init__.py`.
- Verification gates (pytest for their tests, full-suite pytest, mypy, ruff check, ruff format check) before the agent could declare done.
- Instruction to commit on a feature branch, never the base branch.

Pre-step before dispatch (single commit on `chore/initial-scaffold`): added `[fuzz]` and `[attribute]` to `pyproject.toml`, ran `uv lock` once. This eliminated lockfile conflicts that would otherwise arise from agents #4 and #7 each adding their own extra. The advisor flagged this as the load-bearing pre-step; without it, two parallel uv.lock diffs would have conflicted on cherry-pick.

### Per-feature summary

| # | Module | LoC src + test | Tests | Branch | Notes |
|---|--------|----------------|-------|--------|-------|
| 1 | `harness.replay.counterfactual` | 200 + 364 | 10 ✓ | `feat/counterfactual-replay` | Mutations as frozen dataclasses (`RewriteTurn` / `InsertTurn` / `DeleteTurn` / `ReplaceToolResult`), input deep-copied, `session_id` + `created_at` preserved as a sibling timeline. |
| 2 | `harness.contracts` | ~600 + ~700 | 20 ✓ | `feat/behavioral-contracts` | Predicates (`HasToolUse`, `TextMatches`, `RoleIs`, `ArgMatches`) compose with `&` / `\|`; patterns (`Always`, `Eventually`, `Earlier(...).when(...)`, `Never`); shared DFA powers both `attach_contracts(hooks, ...)` and offline `check(record, ...)`. Three actions: `forbid` / `warn` / `require`. |
| 4 | `harness.fuzz` | 509 + 540 | 18 ✓ | `feat/tool-fuzzing` | Pydantic-to-Hypothesis bridge (str / int / float / bool / `Optional` / PEP 604 `\| None`, `Field` constraints honoured). `fuzz_tool` and `fuzz_agent` modes; `harness_property` pytest decorator. Lazy imports — `import harness.fuzz` works without the extra, first call into a Hypothesis path raises a structured `ImportError`. |
| 7 | `harness.attribute` | 619 + 507 | 21 ✓, 1 skip | `feat/causal-provenance` | Leave-one-out ablation. `JaccardSimilarity` and `LengthRatio` are zero-dep; `EmbeddingSimilarity` is opt-in under `[attribute]` (lazy `sentence_transformers` import; clear `ImportError` when missing). `granularity ∈ {"message", "block", "sentence"}`; `estimate_only=True` reports cost without invoking. SHA-256 input cache. |
| 8 | `harness.replay.diff_eval` | 498 + 304 | 9 ✓ | `feat/diff-eval` | Per-runner sessions execute through `asyncio.gather` with `return_exceptions=True` so one runner failing doesn't kill the wave. `DiffMatrix.unanimous` / `.outliers` / `.report_html`. HTML template uses `string.Template` (no Jinja) and is packaged via `[tool.hatch.build.targets.wheel]`. |

Total: ~2 426 src + ~2 415 test, **78 new tests**, 0 cross-feature conflicts on merge.

### Integration

- **`Orchestrator.telemetry` is now a public read-only property.** Agent #1 needed to forward telemetry into a freshly-built inner orchestrator and reached for `_telemetry`; rather than ship a private-attribute access, added a public `@property` and updated the call site in `harness.replay.counterfactual`.
- **Subpackage re-exports.** `harness.replay` now exports the counterfactual mutation types and the differential-matrix family alongside the existing replay surface. `harness.contracts`, `harness.fuzz`, `harness.attribute` ship their own `__init__.py` re-export blocks (built by the dispatched agents).
- **Top-level re-exports** (`harness/__init__.py`): the headline entry points only — `counterfactual` + the four mutation types + `Mutation`; `Contract` / `Violation` / `ContractViolation` / `attach_contracts` / `check`; `fuzz_tool` / `fuzz_agent` / `FuzzReport` / `harness_property`; `attribute` / `AttributionResult` / `AttributionChunk` / `JaccardSimilarity` / `LengthRatio`; the three differential-matrix names. Subpackage-level imports remain the canonical place for the inner predicates / patterns / similarity protocol etc.

### Verification

- `uv sync --extra dev --extra anthropic --extra openai-compat --extra fuzz` — clean. (`[attribute]` deliberately not installed: pulls ~2 GB of torch + transformers; the embedding code is opt-in and tested via `pytest.importorskip` plus a `monkeypatch.setitem(sys.modules, "sentence_transformers", None)` import-error test.)
- `uv run pytest -q` — **229 passed, 1 skipped** in 1.46 s. (Was 151; +10 counterfactual, +20 contracts, +18 fuzz, +21 attribute, +9 diff_eval, +1 skipped attribute embedding sanity check.)
- `uv run mypy` — clean (strict, 52 source files).
- `uv run ruff check .` — clean across `src/`, `tests/`, `examples/`.
- Top-level surface importable from `harness` (counterfactual, the differential-matrix entry, `Contract`, `fuzz_tool`, `attribute` all resolve).

### Follow-ups (explicitly deferred)

- **Assistant-text hook event for contract runtime parity.** `attach_contracts` synthesizes `Message` instances from `PromptSubmit` / `PreToolUse` / `PostToolUse` because there's no event in `harness.hooks` for the assistant's own text. A contract like `Never(RoleIs("assistant") & TextMatches(...))` will fire correctly offline (via `check`) but won't fire live until a `PostAssistantMessage` event lands. Worth a small `harness.hooks` follow-up.
- **`require` mid-stream fail-fast.** Agent #2 extended the spec slightly: if a `require` contract's inner pattern hard-fails before session-end, the runtime raises `ContractViolation` immediately with `kind="forbid_match"`. The spec only pinned end-of-session behaviour. Reasonable; could grow its own kind label.
- **Pydantic synthesis gaps.** `harness.fuzz.pydantic_strategy` raises `FuzzStrategyUnsupported` for lists, nested `BaseModel`, dicts, two-non-None unions, `Decimal`, `datetime`. Callers pass an `overrides=` dict to fill the gap. Expanding the supported set is a follow-up.
- **Pre-existing formatting drift.** `uv run ruff format --check` flags ~10 pre-existing files that pre-date this wave. Out of scope here — clean-up commit can run `uv run ruff format` separately.
- **Wave 2 candidates** (touch the runner internals or build the CLI; deliberately deferred): #3 Prefix-drift watcher, #5 Speculative tool execution, #6 Privacy-boundary runner, #9 Plan-as-contract (build on #2's DFA), #10 Live agent REPL debugger + the shared `harness` CLI.

### Commits
```
cdc16e4  chore: declare [fuzz] and [attribute] extras                    (pre-step)
*  Merge feat/counterfactual-replay (#1)
*  Merge feat/behavioral-contracts (#2)
*  Merge feat/tool-fuzzing (#4)
*  Merge feat/causal-provenance (#7)
*  Merge feat/diff-eval (#8)
*  feat: integrate Wave 1 — Orchestrator.telemetry + re-exports + progress
```

---

