# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.2] — 2026-05-13

**First public PyPI release.** Renames the distribution from
`harness-engineering` to `harness-engineering-toolkit` because the
former is already owned by an unrelated package on PyPI (uploaded
2026-04-28 by a different maintainer). The 1.0.0 and 1.0.1 tags
exist on GitHub but never published to PyPI — `1.0.2` is therefore
the first version anyone can `pip install`.

The importable module name remains `harness`. User code does not
change; only the install command does.

### Changed

- Distribution name: `harness-engineering` → `harness-engineering-toolkit`.
- Install commands across the docs / cookbook / README / source-side
  error messages (`AnthropicRunner`, `OpenAICompatRunner`, fuzz
  imports) updated to reference the new name. Tests that assert on
  the error string text updated to match.
- `pyproject.toml` `name` field updated; everything else
  (dependencies, extras, console-script entry point) unchanged.

### Migration

There is nothing to migrate. The 1.0.x tags before this release
were never on PyPI, so any prior `pip install harness-engineering`
would have hit a different package altogether (the same one that
holds the name today). To install this library, use:

```bash
pip install harness-engineering-toolkit
uv add harness-engineering-toolkit
```

## [1.0.1] — 2026-05-12

Trust-erosion fix release. A post-1.0.0 audit (run by Codex against
the rendered docs site) caught extensive doc/source API drift across
quickstart, the module pages, and the cookbook recipes — examples
invoked constructors with kwargs the source didn't accept, called
methods that didn't exist, and described semantics that no longer
matched the runtime. Each fix in this release was written after
reading the canonical source and running the rewritten snippet
end-to-end; the showcase pages are now CI-gated against future
regressions of the same kind.

No runtime API changes. The library's public surface is unchanged
between 1.0.0 and 1.0.1.

### Added

- Top-level re-exports of names the docs already documented as
  importable from `harness` but which were only reachable via the
  submodule paths: `text`, `attach_file`, `attach_image`, `compact`,
  `assistant_tool_use`, `user_tool_result` (from `harness.prompts`);
  `Event`, `HookDecision`, `PreToolUse`, `PostToolUse`,
  `PostAssistantMessage`, `SessionStart`, `SessionEnd`,
  `PromptSubmit`, `Stop`, `PauseTurn`, `Refusal` (from
  `harness.hooks`); `EvalCase` (from `harness.replay`); `MultiSink`
  (from `harness.telemetry`).
- `pytest-codeblocks` CI step (`pytest --codeblocks
  docs/quickstart.md docs/index.md`) — every fenced Python block on
  the showcase pages is now executed in CI. Catches doc/source
  drift before readers do.
- `pymdownx.snippets` mkdocs extension enabled, so cookbook recipes
  can embed canonical code from `examples/*.py` via `--8<--` syntax;
  the example files are smoke-tested by
  `tests/examples/test_examples_run.py`.

### Fixed

- **Quickstart and home page** — all imports now resolve against the
  real top-level surface.
- **`docs/modules/memory.md`** — `Session(orchestrator, agent, store)`
  is the real ctor; `session.send(...)` and `session.session_id` are
  the real methods. The page previously described an invented
  `Session(store=, agent=)` shape and a non-existent `session.run`.
- **`docs/modules/plan.md`** — `PlannedToolCall` takes `tool_name`,
  `arguments_match`, `arguments_regex`; `mode` lives on `Plan`, not
  on `PlanGuardedRunner`. Documents all three modes (`strict` /
  `superset` / `subset`).
- **`docs/modules/sandbox.md`** — `PathScope.of(allow=[...])` builds
  a scope; `scope.validate(path)` is the gate. `PathPolicy` is a
  `PreToolUse` hook policy, not an enum. `safe_subprocess_run` is
  async. `scrub_env` uses `allow_keys=` (not `keep=`).
- **`docs/modules/replay.md`** — `ReplayRunner` is input-blind: it
  returns canned assistant messages and does not re-run tool
  handlers itself. `RewriteTurn(index=, new_message=)` is the real
  shape. `counterfactual(session, mutation, runner, orchestrator)`
  requires an orchestrator.
- **`docs/modules/fuzz.md`** — strategy bridge covers primitives +
  `Optional` only; `list`, `dict`, `Literal`, nested models all
  raise `FuzzStrategyUnsupported`. `harness_property` is keyed by
  `(dispatcher, tool)`, not `input_model`. Failures are not
  shrunk.
- **`docs/modules/attribute.md`** — `target_message_index` (int) is
  the anchor; granularity is `"message" | "block" | "sentence"`.
  Drops the invented `target=` / `target_match=` API.
- **`docs/cookbook/replay-evaluation.md`** — the batch eval helper
  takes `(cases, orchestrator=, agent=)` and returns
  `list[EvalResult]`. `DiffOutlier` exposes `.case`,
  `.dissenting_runner`, `.consensus_runners`. `matrix.report_html(path)`
  (not `write_html`).
- **`docs/cookbook/fuzz-a-tool.md`** — `fuzz_agent` takes the
  `tool_name`, not `input_model`. The bridge support claim is
  honest about what it covers.
- **`docs/cookbook/cache-and-speculate.md`** — `harness cache-audit
  --store PATH --since 24h` is the real CLI; the previous
  `--window-hours` flag did not exist.
- **`docs/cli.md`** — cache-audit flags table corrected; the
  `harness debug` page now describes the real Wave 13b stepping /
  pause semantics.
- **`docs/faq.md`** — `SECRET_PACK`'s posture is
  `direction="both"`, `action="block"`; the prior outbound-only
  claim was wrong. Replay-handler answer aligns with `ReplayRunner`'s
  actual input-blind behaviour. Step / pause / step-out answers
  match the DAP source.
- **`docs/comparison.md`** — softens the LangChain framing to
  acknowledge LangGraph + LangSmith; drops outdated AutoGen class
  names; removes vendor-cassette replay from the strengths list
  (it is explicitly deferred on the roadmap).
- **`SECURITY.md`** — supported-versions table now reflects the
  post-1.0 reality (1.0.x current).
- **`src/harness/debug/dap.py`** — module docstring no longer
  claims `next` / `stepIn` / `stepOut` / `pause` are "treated as
  continue"; Wave 13b wired the real step semantics months ago.
- **CI / Release / Docs workflows** — set `UV_NO_SYNC=1` at the
  workflow level. Recent uv versions (~0.5) re-derive the project's
  default dependency set on every `uv run` invocation and remove
  anything the prior `uv sync --extra ...` step installed; without
  this env var the gate failed before reaching the build step, the
  Pages deploy could not find `mkdocs`, and the trusted-publisher
  release step never got the chance to run. The local gate (which
  invokes the binaries directly out of `.venv/bin/`) was unaffected,
  which is how the issue went unnoticed.

## [1.0.0] — 2026-05-10

The "ready for 1.0" release. All ten standout features from
`designs/standout.md` shipped (`0.2.0`), plus six post-`0.2` waves
(9–13b) addressing the Wave-8 audit gaps. Public surface is
considered stable from this release; subsequent breaking changes
will go through the standard semver deprecation cycle.

### Added

- `harness.privacy.PresidioDetector` (Wave 13b #1) under the new
  `[privacy-ml]` extra (`presidio-analyzer>=2.2`). Wraps Presidio's
  `AnalyzerEngine` behind the existing `Detector` Protocol; broader
  recognizers than the regex/entropy pack (people's names,
  international phone numbers, addresses, IBAN, etc.). Lazy-imports
  the SDK; the constructor raises `ImportError` with a clear
  `[privacy-ml]` install hint if the extra isn't present.
  `harness.privacy.build_pii_pack()` returns a pre-configured
  outbound-only pack mirroring `PII_PACK`'s posture.
- DAP `pause` request now actually pauses (Wave 13b #16) — sets a
  flag the runner's `break_on` consults; the next runner invocation
  stops. Editor's pause button works.
- DAP `next` / `stepIn` / `stepOut` requests now drive distinct
  semantics (Wave 13b #15) via `step_mode` flags on `DapAdapter`
  that `break_on` reads. `next` and `stepIn` step over a tool call;
  `stepOut` runs to the next assistant message.
- DAP `evaluate` opt-in arbitrary expressions (Wave 13b #17) — the
  editor passes `allowEvaluate: true` in launch arguments to enable
  REPL-equivalent expression evaluation against `ctx`. Default
  remains the restricted variables-view names. Routes through the
  new `harness.debug.repl.evaluate_in_context` helper, sharing the
  REPL's code path.
- Eager per-block speculator cancellation (Wave 13b #2) — when
  `max_speculations == 1` and `observe()` sees a non-matching call,
  the lone pending speculation is cancelled immediately instead of
  waiting for `cancel_unobserved` at stream-end. For
  `max_speculations > 1`, the stream-end policy is preserved
  (correctness with multiple pending requires policy that's not
  worth the complexity).

### Added

- `harness.streaming` module (Wave 13a #9) — `TextDelta`,
  `ToolUseStart`, `ToolUseEnd`, `MessageEnd` event types and a
  `runtime_checkable` `StreamingRunner` Protocol. Runners that
  implement `run_stream(agent, messages) -> AsyncIterator[StreamEvent]`
  satisfy the protocol structurally.
- `AnthropicRunner.run_stream()` — parallel method to `__call__()`
  that yields `TextDelta` per text-delta event, `ToolUseStart` at
  each `content_block_stop` for `tool_use` (after speculator.observe,
  before dispatch), `ToolUseEnd` after dispatch, and exactly one
  terminal `MessageEnd`. Non-streaming `__call__` is untouched per
  the advisor recommendation: parallel methods, not refactor.
- `Orchestrator.run_stream(agent, messages)` — async generator that
  delegates to the runner's stream after wrapping in
  `session_scope` + `span_scope` (when telemetry is configured) and
  emitting `SessionStart` / `SessionEnd` lifecycle hooks. Raises
  `TypeError` if the runner doesn't implement `StreamingRunner`.
- Top-level re-exports of `MessageEnd`, `StreamEvent`,
  `StreamingRunner`, `TextDelta`, `ToolUseEnd`, `ToolUseStart` from
  `harness`.
- `harness.prompts.ImageRef` (Wave 12 #7) and `attach_image(path|url, ...)`
  helper. New `image` `ContentBlock.type` carries an `ImageRef` (base64
  bytes or URL + media_type). `AnthropicRunner` translates to
  `{"type":"image","source":{...}}` (both base64 and URL). The
  OpenAI-compat path translates to `image_url` parts (data URLs for
  inline base64). User messages with images become list-shaped
  `content` arrays so text + image mixing works in both vendor
  formats.
- `attach_file(file_id="file_...")` (Wave 12 #8) — references an
  Anthropic Files API document by id. `AnthropicRunner` translates
  to a `{"type":"document","source":{"type":"file","file_id":...}}`
  block. Path-based `attach_file(path=...)` continues to inline text
  as before. OpenAI-compat surfaces file_ids as text placeholders.
- `TelemetryEvent` gains `trace_id` / `span_id` / `parent_span_id`
  fields (Wave 11 #11) — propagated via the `Telemetry` recorder's
  new `session_scope()` and `span_scope()` async context managers
  using `contextvars`. The orchestrator opens both per `run()`; the
  dispatcher opens a `span_scope` per `dispatch()`. Concurrent
  dispatches across `asyncio.gather` get distinct span_ids.
- `OpenTelemetrySink` promotes the new correlation IDs as
  `harness.trace_id` / `harness.span_id` / `harness.parent_span_id`
  attributes on the OTel event (Wave 11 #10) — events grouped by
  harness session correlate in Jaeger / Tempo / Honeycomb. Full span-
  tree synthesis is documented as deferred (would need a custom
  `IdGenerator` to round-trip the harness IDs faithfully).
- `tests/debug/test_dap_cli.py` (Wave 11 #18) — subprocess-driven
  end-to-end test of `harness debug --dap`. Validates the
  `connect_read_pipe` / `connect_write_pipe` plumbing the CLI uses
  for real editor integrations.
- Coverage tooling (Wave 11 #20) — `pytest-cov` is now in `[dev]`
  extras; CI runs `pytest --cov=harness` and gates on a configurable
  threshold (`fail_under = 85` in `pyproject.toml`, currently
  reporting **89%**).
- `harness.runner.anthropic.CacheBreakpointLimitExceeded` (Wave 10 #12) —
  raised when a request would exceed Anthropic's 4-cache-breakpoint cap,
  surfacing the failure at the harness boundary rather than as an
  opaque API 400.
- `timeout_s` kwarg on `AnthropicRunner` and `OpenAICompatRunner`
  (Wave 10 #6) — wraps the SDK call in `asyncio.wait_for`. Default
  `None` (no timeout). Per-iteration, not per-call.
- `harness.hooks.PauseTurn` and `harness.hooks.Refusal` events (Wave
  10 #4) — `AnthropicRunner` now emits these instead of raising on
  `pause_turn` / `refusal` stop reasons; the partial assistant message
  is returned so callers can resume / inspect.
- `OpenAICompatRunner` now surfaces emitted `tool_call`s to
  `speculator.observe()` and calls `cancel_unobserved()` before
  dispatch (Wave 10 #3) — feature parity with `AnthropicRunner`'s
  Wave 6 cancellation timing.

### Changed

- Both runners honor `HookDecision.replacement` (Wave 10 #5) —
  `PreToolUse` replacement short-circuits dispatch with the supplied
  `ToolResult` (id patched to the model's call id); `PostToolUse`
  replacement rewrites the dispatched result before it's sent back to
  the model. Pre-Wave-10 only `block` was honored.


## [0.2.0] — 2026-05-09

The "ten standout features" milestone. Every item from `designs/standout.md`
ships, plus runnable examples, per-event speculator cancellation, a DAP
server for IDE integration, and a docs site.

### Added

- **Wave 1** — counterfactual replay (`harness.replay.counterfactual`),
  behavioral contracts (`harness.contracts`), tool/agent fuzzing
  (`harness.fuzz`), causal attribution via leave-one-out ablation
  (`harness.attribute`), and cross-provider differential evaluation
  (`harness.replay.diff_eval`).
- **Wave 2** — prompt-prefix-drift watcher (`harness.cache`), privacy
  boundary with regex + entropy detectors (`harness.privacy`),
  plan-as-contract enforcement (`harness.plan`), and an interactive
  REPL debugger (`harness.debug`).
- **Wave 3** — speculative tool execution
  (`harness.speculate.Speculator` + `LastCallPredictor`,
  `SequencePredictor`).
- **Wave 4** — `OpenTelemetrySink` (`[otel]` extra),
  plan inference from past sessions
  (`harness.plan.infer_plan_from_records`),
  cross-session predictor (`harness.speculate.CrossSessionPredictor`),
  and `OpenAICompatRunner` speculator wiring.
- **Wave 5** — 13 runnable examples (one per module), each smoke-tested
  in CI under `tests/examples/`.
- **Wave 6** — per-event speculator cancellation in `AnthropicRunner`
  (`Speculator.observe()` + `cancel_unobserved()`); unmatched
  speculations cancel at stream-end before dispatch begins.
- **Wave 7** — DAP server (`harness debug --dap`) so IDEs (VS Code,
  neovim-dap, Emacs dap-mode) can drive the same replay-based debug
  session as the REPL. `DapAdapter` runs a concurrent message loop and
  orchestrator session; breakpoint inspection is non-blocking.
- **Wave 8** — MkDocs docs site under `docs/`, top-level re-exports of
  `DapAdapter` / `DapProtocolError`, narrowed two `Any` annotations in
  `harness.fuzz.runner`, added a 200-message-history orchestrator
  stress test, and pruned the deferred-items list to the honest
  remainder.

### Changed

- `harness.runner.protocols.SpeculatorProtocol` gains `observe()` and
  `cancel_unobserved()` lifecycle methods (Wave 6). Existing
  `Speculator` and test-stub implementers updated; structural
  compatibility preserved.
- `Speculator` internal pending state moved from `tuple[ToolCall, Task]`
  to a `_Pending` dataclass with an `observed: bool` flag.
- `tests/runner/fakes.FakeMessage` gains an optional
  `events: list | None` field; when `None`, `_FakeStream.__aiter__`
  auto-derives one `content_block_stop` per content entry, so
  pre-Wave-6 tests work unchanged.

### Fixed

- **`OpenTelemetrySink` emitted `None`-valued attributes** (Wave 5
  surfaced via examples). The OTel SDK rejects `None` and logs a
  warning; we now skip `None` values explicitly.
- **`harness.attribute.similarity` mypy errors with `[attribute]`
  installed** (Wave 5). The `# type: ignore[import-not-found]`
  suppressions on the lazy `sentence_transformers` and `numpy` imports
  became `unused-ignore` errors when those packages were installed.
  Combined to `[import-not-found, unused-ignore]` so mypy is happy in
  both states.
- **Hook-handler exceptions propagating through speculation** (Wave 3).
  `Speculator._dispatch_via_hooks` now wraps execution in a
  `try`/`except`; exceptions become `is_error=True` `ToolResult`s so a
  buggy hook in the speculative path doesn't crash the runner.
- **`PostAssistantMessage` not firing on Anthropic runner**
  (Wave 1+2 integration). The runner now emits the event after every
  assistant turn produced via the SDK or via a debugger mutation.
- **Privacy boundary missed `tool_use.arguments` and `tool_result.content`**
  (Wave 2 integration). The recursive scan now walks both fields with
  a depth cap.

## [0.0.1] — 2026-04 (initial scaffold)

The MVP: tools + dispatcher (`harness.tools`), prompts and content
blocks (`harness.prompts`), typed lifecycle hooks (`harness.hooks`),
allow/deny policies (`harness.policy`), agents and orchestrator
(`harness.agents`), pluggable runners (`harness.runner` —
`EchoRunner`/`CannedRunner` ship; `AnthropicRunner` and
`OpenAICompatRunner` opt-in via extras), pluggable telemetry sinks
(`harness.telemetry`), session memory (`harness.memory`), filesystem
sandbox (`harness.sandbox`), deterministic replay
(`harness.replay.ReplayRunner`).

[Unreleased]: https://github.com/dr-gareth-roberts/harness-engineering/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/dr-gareth-roberts/harness-engineering/releases/tag/v1.0.2
[1.0.1]: https://github.com/dr-gareth-roberts/harness-engineering/releases/tag/v1.0.1
[1.0.0]: https://github.com/dr-gareth-roberts/harness-engineering/releases/tag/v1.0.0
[0.2.0]: https://github.com/dr-gareth-roberts/harness-engineering/releases/tag/v0.2.0
[0.0.1]: https://github.com/dr-gareth-roberts/harness-engineering/releases/tag/v0.0.1
