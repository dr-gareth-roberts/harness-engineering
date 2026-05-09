# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/dr-gareth-roberts/harness-engineering/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dr-gareth-roberts/harness-engineering/releases/tag/v0.2.0
[0.0.1]: https://github.com/dr-gareth-roberts/harness-engineering/releases/tag/v0.0.1
