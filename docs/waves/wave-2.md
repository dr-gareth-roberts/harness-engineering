# Wave 2 — four runner-adjacent features + post-Wave-2 integration (archived)

> Archived from `progress.md` after Wave 3 shipped. Wave 2 implemented
> features #3, #6, #9, #10 from `designs/standout.md` (#5 was
> deferred to Wave 3) plus the cross-cutting CLI scaffold + runner
> protocol extension points that let three of the four agents work
> in pure new-module mode.
>
> The "Post-Wave-2 integration fixes" section captures the round of
> targeted fixes applied between Wave 2 and Wave 3: the
> `PostAssistantMessage` hook event closing the contracts
> runtime/offline asymmetry, the privacy-boundary scope extension
> to tool_use args + tool_result content, the README rewrite for the
> 18-module surface, and the format sweep on pre-existing files.
>
> Features live in `src/harness/{cache, privacy, plan, debug}` and
> the related extensions on `src/harness/{cli, hooks/events,
> runner/protocols, runner/anthropic, runner/openai_compat,
> contracts}`. Tests under `tests/`.

## Wave 2 — four runner-adjacent features in parallel

### Goal
Implement four of the remaining five standout features in parallel: #3 Prefix-drift watcher, #6 Privacy-boundary runner, #9 Plan-as-contract, #10 Live agent REPL debugger. **#5 Speculative tool execution is deferred to a later wave** — its implementation needs a streaming refactor of the runners (the SDK call path uses `get_final_message()` from the caller's perspective today), which is a research-grade unknown that doesn't belong inside a parallel-feature wave.

These features share a different parallelism profile than Wave 1: three of the four wrap an existing `Runner` rather than create a new one (#6 privacy, #9 plan-as-contract, #10 debug), and the fourth (#3 cache) plugs into the runner via a single kwarg. None modify the runner's tool-use loop.

### Status
Shipped — four `feat/<name>` branches merged into `chore/initial-scaffold` with `--no-ff` merge commits; one commit pre-staged the cross-cutting infrastructure; one final commit integrates re-exports.

### Approach

**Pre-step on `chore/initial-scaffold` (single commit, `322e057`)** — lands the cross-cutting infrastructure so each agent works in pure new-module mode:

- `src/harness/cli.py` — argparse subparser dispatcher. Each feature module that wants a CLI surface ships a `register(subparsers)` function in its own `cli.py`; the dispatcher discovers them lazily via `importlib.util.find_spec` (catching `ModuleNotFoundError` raised when a parent package is missing entirely, while letting real import-time failures surface from `import_module`). New features add their subcommand by adding their module path to `_SUBCOMMAND_MODULES` and shipping a `register`. Today: `harness.cache.cli` (cache-audit) + `harness.debug.cli` (debug).
- `[project.scripts] harness = "harness.cli:main"` in `pyproject.toml`.
- `src/harness/runner/protocols.py` — `PrefixWatcherProtocol` (structural). `harness.cache.PrefixWatcher` satisfies it; the runner stays vendor-SDK-only and doesn't import the cache module.
- `AnthropicRunner.__init__` and `OpenAICompatRunner.__init__` accept `prefix_watcher: PrefixWatcherProtocol | None = None` and `speculator: object | None = None`. Each iteration of the tool-use loop calls `await self._prefix_watcher.fingerprint(request)` immediately before the SDK call when the watcher is present. The `speculator` slot is reserved for #5 so that feature can land later without a constructor signature change.
- `harness.contracts` re-exports `DFA` + `compile_contract` publicly so #9 plan-as-contract composes them rather than reinventing the state machine.
- `Tool.idempotent: bool = False` on the schema. Reserved for #5; ignored elsewhere.

**Then dispatch four `general-purpose` agents in parallel via the agent runner's worktree-isolation mode**, with prompts following the same shape as Wave 1: spec section as binding source-of-truth, explicit "do not modify" list (`src/harness/__init__.py`, `pyproject.toml`, `uv.lock`, `README.md`, `progress.md`, `designs/standout.md`, the runner files, `src/harness/cli.py`), verification gates before declaring done, commit on a feature branch.

### Per-feature summary

| # | Module | LoC src + test | Tests | Branch | Notes |
|---|--------|----------------|-------|--------|-------|
| 3  | `harness.cache` | ~700 + ~660 | 28 ✓ | `feat/prefix-drift-watcher` | `PrefixWatcher` satisfies the runner's structural protocol; per-cache-breakpoint SHA-256 fingerprints on the rendered `messages`/`system`/`tools` segments; `FileFingerprintStore` writes JSONL; `audit(store, window_hours)` walks per-breakpoint hash sequences and emits `DriftEvent`s with `difflib.unified_diff` of full prompts when `full_capture="on_drift"`. `harness cache-audit --store <path> --since <duration>` ships as a CLI subcommand. OpenAI-compatible runner: single-segment fingerprint (no breakpoint markers in the SDK protocol). 18 supplementary tests beyond the spec's 10. |
| 6  | `harness.privacy` | ~470 + ~660 | 22 ✓ | `feat/privacy-boundary` | `PrivacyBoundary(detectors, on_detect=..., audit_sink=...).wrap(real_runner)` returns a runner satisfying the same protocol. `RegexDetector` and `EntropyDetector` (Shannon entropy on `[A-Za-z0-9_+/=-]+` tokens). Pre-built `SECRET_PACK` (AWS / Anthropic / GitHub / Stripe keys), `PII_PACK` (US SSN / phone / email), `HIPAA_PACK`. Per-detector `direction` (outbound / inbound / both) and `action` (redact / block / audit). `DetectionEvent` carries `name`, `direction`, `action`, `location`, `match_length`, `timestamp_utc` — never the detected value (verified by walking every field of `event.model_dump()` in test 8). |
| 9  | `harness.plan` | ~430 + ~770 | 26 ✓ | `feat/plan-as-contract` | `Plan` is a Pydantic model serializable to JSON. `Plan.to_contracts()` returns `[Contract(pattern=Always(HasToolUse(...) & ArgMatches(...)))]` per `PlannedToolCall`. `PlanGuardedRunner._compile_step_dfa` calls `compile_contract(self._contracts[step_index])` — the contracts DFA is the substrate, not reimplemented. Three modes: `strict` / `superset` / `subset`. `derive_plan(planner_agent, planner_runner, messages)` asks a planner to emit JSON; `Plan.model_validate_json` parses. Callable matchers stay out of the Pydantic schema (would not serialize) — the predicate-subclass path is documented and tested. |
| 10 | `harness.debug` | ~620 + ~1100 | 64 ✓ | `feat/debug-repl` | `DebugRunner(real_runner, *, break_on, breakpoint_callback OR interactive)` wraps any runner; satisfies the runner protocol. `DebugContext` exposes `.messages` / `.last_call` / `.turn_index` and queues `mutate(...)` / `fire(tool, args)` / `inspect(...)` / `resume()` / `abort()`. Interactive REPL is a small `cmd`-flavored loop driven by configurable streams — tests inject `io.StringIO` and `monkeypatch` instead of a `pexpect` dependency (which the spec mentioned but we elected not to take). `harness debug <session> --break <spec>` loads a recorded `SessionRecord` (`.json` or `.jsonl`), replays through `ReplayRunner`, breaks at `turn=N` or `tool=NAME`, drops to the REPL. |

Total Wave 2: ~2 220 src + ~3 190 test, **140 new tests**. Plus the pre-step (Wave 2 cli.py + protocols.py + 32 lines of edits across 5 files); 0 logic conflicts at merge (one expected-and-handled `cli.py` + `pyproject.toml` conflict for #10, where the agent recreated the pre-step files because their worktree initialization captured an earlier base; resolved at integration by keeping the pre-step versions, which already register #10's subcommand).

### Integration

- **Top-level re-exports** added for the headline entry points across all four features: `PrefixWatcher` / `FileFingerprintStore` / `DriftEvent` / `DriftReport`; `PrivacyBoundary` / `PrivacyViolation` / `RegexDetector` / `EntropyDetector` / `SECRET_PACK` / `PII_PACK`; `Plan` / `PlannedToolCall` / `PlanGuardedRunner` / `PlanViolation`; `DebugRunner` / `DebugContext`. Inner types (predicates, similarity protocols, individual detector packs, REPL helpers, etc.) remain at subpackage level.
- **Conflict resolution on `feat/debug-repl`**: the agent's worktree captured a base older than the Wave 2 pre-step (their `pyproject.toml` lacked the `[fuzz]` and `[attribute]` extras Wave 1 added, and they recreated `cli.py` from scratch). At merge, kept the pre-step's `cli.py` (which registers BOTH `harness.cache.cli` and `harness.debug.cli` in `_SUBCOMMAND_MODULES`) and the pre-step's `[project.scripts]` block; deduplicated the auto-merged `[project.scripts]` table. Their new files (`src/harness/debug/*` and `tests/debug/*`) merged cleanly.
- **CLI surface verified**: `uv run harness --help` lists `cache-audit` and `debug`; `uv run harness cache-audit --help` and `uv run harness debug --help` both render their subparsers.

### Verification

- `uv sync --extra dev --extra anthropic --extra openai-compat --extra fuzz` — clean. (`[attribute]` still deliberately not installed.)
- `uv run pytest -q` — **369 passed, 1 skipped** in 1.48 s. (Was 229; +28 cache, +22 privacy, +26 plan, +64 debug.)
- `uv run mypy` — clean (strict, 73 source files).
- `uv run ruff check .` — clean across `src/`, `tests/`, `examples/`.
- `uv run python examples/end_to_end.py` — runs to completion; no top-level import regressions.
- Top-level surface importable: `from harness import PrefixWatcher, PrivacyBoundary, Plan, PlanGuardedRunner, DebugRunner, ...` resolves.

### Follow-ups (explicitly deferred)

- **#5 Speculative tool execution.** Deliberately deferred until the runner streaming path is refactored. Wave 2's pre-step already accepts a `speculator` kwarg (typed `object | None`) on both runners, so #5 lands without re-touching their constructor signatures. `Tool.idempotent: bool = False` is similarly pre-staged.
- **Privacy boundary scope.** v1 scans `type == "text"` blocks only; `tool_result.content` and `tool_use.arguments` are not scanned in the inbound path even though the spec body mentions both. None of the 12 numbered tests cover those, but it's an acknowledged limitation in the boundary module docstring.
- **Privacy `on_detect` boundary default.** Every shipped detector specifies its own `action`; the boundary-level `on_detect` default is currently latent. Extension hook for future per-detector-default fallback semantics.
- **Wave 3 work** beyond #5: README updates for the new modules, push to remote, an example or two under `examples/` exercising privacy + plan + debug flows, formatting sweep on the ~10 pre-existing files that pre-date Wave 1.

### Commits

```
322e057  chore: Wave 2 pre-step — CLI scaffold + runner extension points  (pre-step)
*  Merge feat/prefix-drift-watcher (#3)
*  Merge feat/privacy-boundary (#6)
*  Merge feat/plan-as-contract (#9)
*  Merge feat/debug-repl (#10)            (cli.py + pyproject.toml conflicts; took pre-step versions)
*  feat: integrate Wave 2 — top-level re-exports + progress
```

---

## Post-Wave-2 integration fixes

Three follow-ups flagged in the Wave 1 + Wave 2 reviews, applied as a single
integration pass before Wave 3 begins. Each is a small, scoped fix on
`chore/initial-scaffold`; total +~600 LoC across src + tests + docs.

### Fix 1 (`dd3ed88`): `PostAssistantMessage` event + contracts runtime parity

Closes the asymmetry that Wave 1 left documented as a follow-up: a contract
like `Never(RoleIs("assistant") & TextMatches(r"i'?m sorry"))` only fired
offline via `check(...)` because no hook event carried the assistant
message. Now:

- `harness.hooks.events.PostAssistantMessage(Event)` carries the assistant
  `Message`. Docstring is explicit that it fires once per assistant message
  the model produces — including intermediate text-plus-tool-use messages,
  not only the terminal one.
- `AnthropicRunner` and `OpenAICompatRunner` both emit it immediately after
  `_translate_out(...)` builds the assistant `Message`, before the
  stop-reason check, so terminal AND continuing iterations surface.
- `attach_contracts` registers an observational handler. `forbid` and
  `warn` matches surface as `ContractWarning` telemetry (the message has
  already been produced — block is meaningless); `require` mid-stream still
  raises (existing fail-fast extension).

**Side effect worth noting for Wave 3:** the same observational helper now
backs the existing `PostToolUse` handler. Before this commit, a `forbid`
contract matching on a `PostToolUse` event was silently dropped — the
handler called `_react_to_violation`, which returned a
`HookDecision(block=True)`, but the handler's return type was `None` and
the decision was discarded. After this commit, the same match emits a
`ContractWarning` to the configured telemetry sink. Strictly an
improvement; no behavior depends on the silent drop. Flagged here so a
downstream user who built workflows around the silent-drop behavior knows
where to look.

Tests: 2 new contracts tests pin live assistant-text enforcement; 2 new
runner tests pin per-iteration `PostAssistantMessage` emission via the
existing `FakeAsyncAnthropic` fixture (terminal-only and full-loop cases).

### Fix 2 (`888819e`): privacy boundary scope extension

`PrivacyBoundary` v1 scanned `text` content blocks only. Wave 2's review
flagged `tool_use.arguments` and `tool_result.content` as a documented
limitation. Now:

- `tool_use.arguments` walked recursively to find string leaves. Each
  string value is scanned via the same `_scan_text` path that handles top-
  level text — so `block` actions raise `PrivacyViolation` before the inner
  runner is called, and `redact` actions replace the matched value in
  place. Audit events surface paths like
  `messages[i].content[j].tool_use.arguments.<key>`.
- `tool_result.content` (`Any`-typed): when the value is a string, scanned
  directly. When it's a dict / list, recursed.
- Recursion capped at `_MAX_RECURSION_DEPTH = 4`. Beyond the cap, the
  subtree is `json.dumps(default=str, sort_keys=True)`-ified and scanned
  flat — detection still works, audit location carries the `[depth-cap]`
  suffix so callers know recursion was truncated.
- Location-path grammar documented in the boundary module docstring:
  `<base>.<key>` for nested keys, `<base>[n]` for list indices,
  `<base>[depth-cap]` for the flat-scan suffix.

Tests: 6 new (string redaction in tool_use args, block before inner call,
string redaction in tool_result content, nested dict redaction with dotted
path, list-element redaction with `[n]` grammar, depth-cap fall-back).

### Fix 3 (`6a32fe4`): README rewrite + format sweep

The original README listed seven MVP modules and an "out-of-scope" Roadmap
that was now entirely shipped. Rewrote the module table for the current
18-module surface, grouping into:

- **Core primitives** — the original tools / prompts / hooks / policy /
  agents / runner / telemetry / memory / sandbox / replay set, with the
  new `PostAssistantMessage` event listed and the runner row updated to
  mention the `prefix_watcher` / `speculator` extension kwargs.
- **Behavior & enforcement** — `harness.contracts`, `harness.privacy`,
  `harness.plan`.
- **Quality & exploration** — `harness.fuzz`, `harness.attribute`,
  `harness.cache`, `harness.debug`.

Plus a CLI section documenting the `register(subparsers)` discovery
contract, and a Roadmap that reflects what's actually deferred today
(speculative tool execution, OpenTelemetry export, plan inference from
past sessions, ML-based privacy detection, DAP integration for the debug
REPL).

Format sweep: `uv run ruff format .` applied to ~13 pre-existing files
that pre-dated Wave 1's clean-format agents. Pure whitespace / line-break
changes; tests unchanged.

### Additional runner emission test (`<this commit>`)

Added in response to a post-fix review: Fix 1's runner emission of
`PostAssistantMessage` was previously only verified via mypy structural
typing — every test of the new event drove `attach_contracts` directly,
none through a runner instance. Two new tests in
`tests/runner/test_anthropic.py` close that gap using the existing
`FakeAsyncAnthropic` infrastructure. The OpenAI-compat runner's emission
mirrors the Anthropic runner's loop position; documented as a Wave 3
TODO if a regression-defending mirror is wanted.

### Verification (post-fixes)

- `uv run pytest -q` — **379 passed, 1 skipped** in 1.5 s (was 377; +2
  runner emission tests).
- `uv run mypy` — clean strict (73 source files).
- `uv run ruff check` + `ruff format --check` — both clean.
- `uv run python examples/end_to_end.py` — runs to completion.
- `uv run harness --help` — `cache-audit` + `debug` subcommands wired.

### Wave 3 prologue

Wave 3 is gated on a runner streaming refactor for `harness.speculate`
(#5). When that lands:

- Add `SpeculatorProtocol` to `src/harness/runner/protocols.py` (mirror of
  `PrefixWatcherProtocol`).
- Widen the `speculator` kwarg on both runners from `object | None` to
  `SpeculatorProtocol | None` — typing-only widening, doesn't break
  callers.
- Refactor the runner's tool-use loop to pass through stream events
  rather than `get_final_message()`-only.

`Tool.idempotent: bool = False` is already in the schema (added in the
Wave 2 pre-step); the speculator only fires on idempotent tools.

---

