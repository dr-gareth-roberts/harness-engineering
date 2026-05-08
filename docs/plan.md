# Path to 1.0 — forward plan

[`progress.md`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/progress.md)
is the backward log of work shipped so far.
This page is the forward plan — what's left between today (`0.2.0`,
all 10 standout features shipped, Waves 5–8 cleaned up) and a real
`1.0` release.

The plan is built on the 28 known gaps tracked at the end of Wave 8.
Each gap has a number; the wave-level approach below references those
numbers so individual items are auditable. Effort estimates are
calibrated against the cadence of Waves 5–8 (one developer-day ≈ 1
focused session with iteration).

| Wave | Theme | Effort | Items addressed |
| --- | --- | --- | --- |
| 9  | CI/CD + governance + housekeeping | 1.5–2 days | #21, #22, #23, #24, #25, #26, #27, #28 |
| 10 | Vendor runner parity + robustness | ~3 days | #3, #4, #5, #6, #12 |
| 11 | Deeper observability + verification | ~3 days | #10, #11, #18, #19, #20 |
| 12 | Modality + Files API | ~1 day | #7, #8 |
| 13a | Streaming output | ~2 days | #9 |
| 13b | Speculator/privacy/DAP polish | ~3 days | #1, #2, #15, #16, #17 |

Two items (#13 namespace `getattr`, #14 `FlowControlMixin`) are
**persistent-monitoring**, not new waves — see the bottom of this
page.

Total: **~13–15 developer-days** from `0.2.0` to a defensible `1.0`.

---

## Wave 9 — CI/CD + governance + housekeeping

### Goal
Move from "tests pass on my machine" to "the project's gate runs on
every PR, the package is installable from PyPI, the docs are reachable
on the web, and the contributor flow is documented." Mostly mechanical
work with high signal-to-noise.

### Items

| # | Item | Approach |
| --- | --- | --- |
| 22 | No CI/CD | `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`, `mypy --strict src/harness`, `pytest`, `mkdocs build --strict`, `uv build`. Matrix on Python 3.11 / 3.12 / 3.13. Caches `.venv` between runs. Fails the PR on any red gate. |
| 28 | No PyPI publishing | `.github/workflows/release.yml` triggered on tag `v*`. Builds wheel + sdist, runs the gate, publishes to PyPI via OIDC trusted publishing (no API token in repo). README install instructions update from "GitHub" to `pip install harness-engineering`. |
| 23 | No GitHub Pages deployment | `.github/workflows/docs.yml` builds the MkDocs site on push to `main`, deploys via `peaceiris/actions-gh-pages`. Pages source set to the `gh-pages` branch. README links to the public URL. |
| 24 | No CHANGELOG.md | `CHANGELOG.md` at repo root following Keep-a-Changelog format. Pre-fill with `0.2.0` (everything in `progress.md` distilled to user-facing one-liners) and `0.0.1` (the MVP). Future PRs add entries under `Unreleased`. |
| 25 | No CONTRIBUTING.md | `CONTRIBUTING.md` at repo root. Sections: dev setup (`uv sync --extra dev`), running the gate (`pytest`, `mypy`, `ruff`), running the docs (`uv sync --extra docs && uv run mkdocs serve`), commit conventions (imperative mood, no emoji, conventional-commits-style scope), PR expectations (focused, reviewable, gated). |
| 26 | No SECURITY.md | `SECURITY.md` at repo root. Covers: scope of the package's privacy/secret detection, how to report issues responsibly (security@... — adjust to your org's address), embargo expectations, supported versions. Also documents the arbitrary-expression `inspect` surface in the debug REPL as a known but bounded risk (only reachable when a breakpoint hits in an opt-in debug session). |
| 27 | `progress.md` is ~700 lines | Rotate Waves 3–7 to `docs/waves/wave-{3..7}.md`, mirroring the existing pattern where Waves 1–2 are archived. `progress.md` keeps Wave 8 inline + the status-snapshot table + cross-cutting decisions. Update mkdocs nav and link from `roadmap.md`. |
| 21 | Pre-existing mypy errors in `tests/` | Run `mypy src tests` and triage. Most failures are `func-returns-value` on lambda hooks (`lambda e: list.append(...) or None` is the workaround pattern; cleanest fix is wrapping in a real function). Add `mypy --strict src tests` to the CI gate once the count is zero. |

### Acceptance

- A PR with a deliberate failing test or mypy error is rejected by CI before any reviewer looks at it.
- `pip install harness-engineering` (or `uv add`) installs `0.2.0` from PyPI.
- The docs are at `https://dr-gareth-roberts.github.io/harness-engineering/` (or wherever Pages is configured) and update on each `main` push.
- A new contributor can clone, run `uv sync --extra dev`, run `pytest` and `mkdocs serve`, and have everything Just Work, with explicit instructions for it.
- Security disclosures have a documented intake path.
- `progress.md` is back under 200 lines.
- `mypy --strict src tests` is clean and CI-gated.

### Sequencing

CI first (#22), then the rest can land in any order. Release flow (#28)
needs CI green; Pages (#23) needs the site building cleanly (already
true). Test mypy (#21) is the only item that might surface a real bug
— budget for it accordingly.

---

## Wave 10 — Vendor runner parity + robustness

### Goal
Bring `OpenAICompatRunner` to feature-parity with `AnthropicRunner` on
the speculator surface, and make both runners safe to use under
production conditions (transient errors, slow upstream, hooks that
want to rewrite messages). Closes the "Anthropic is the privileged
runner" implicit asymmetry.

### Items

| # | Item | Approach |
| --- | --- | --- |
| 3 | `OpenAICompatRunner` is not event-aware | OpenAI's chat-completions stream emits `chunk.choices[0].delta.tool_calls` incrementally. Refactor the runner from "await full response" to "iterate stream, accumulate tool-call args by index, surface each completed tool call to `speculator.observe()` at the chunk where its `index`'s args finish JSON-parsing." After stream end, call `cancel_unobserved`. Fakes (`tests/runner/fakes_openai.py`) extend with chunk-iteration like `FakeAsyncAnthropic` did in Wave 6. New tests mirror the AnthropicRunner Wave 6 set. |
| 4 | `pause_turn` / `refusal` raise | `pause_turn`: capture the partial assistant message, surface a `PauseTurn` event, return a special marker for the orchestrator to decide whether to re-invoke with the unpaused continuation. `refusal`: surface the refusal text via a new `Refusal` event, return the (refusal-only) assistant message normally so callers can inspect `is_refusal`. Both need new content-block flags or events; opt for events to avoid breaking `Message.content` consumers. |
| 5 | `HookDecision.replacement` ignored | Honor it in the runner's `PreToolUse` handling: when a hook returns `HookDecision(replacement=ToolResult(...))`, skip dispatch and use the replacement directly (with the model's `tool_use.id` patched in, mirroring the speculator-hit path). Ditto for `PostToolUse` rewriting the result before it's sent back to the model. Update the policy module's docs to advertise the capability. |
| 6 | No retry/backoff or per-call timeouts | Add `retry: RetryPolicy = ExponentialBackoff(max_attempts=3, base=0.5, jitter=True)` and `timeout_s: float | None = None` kwargs to both vendor runners. Wrap the SDK call in `asyncio.wait_for` and a retry loop that knows about the SDK's typed exceptions (rate-limit, server-error, timeout). Defaults: 3 attempts on transient errors, no timeout (caller opts in). Tests use injected sleep + flaky fake to verify backoff timing without real network. |
| 12 | Cache-breakpoint cap not enforced | Walk the request before the SDK call; count `cache_control` markers; raise `ValueError("Anthropic caps cache breakpoints at 4; got N")` early. Surfaces the failure at the harness boundary instead of the API boundary. Add a docstring example showing how `compact()` solves it. |

### Acceptance

- `OpenAICompatRunner` accepts a `Speculator` and the test suite has the same observe/cancel-unobserved coverage Wave 6 added for Anthropic.
- An agent paused on `pause_turn` can be resumed by re-invoking the runner with the partial message; refusals are surfaced as a typed event.
- A hook can rewrite both inbound (model's tool call → injected result) and outbound (real result → sanitized result) and the runner honors it.
- A flaky upstream (50% rate-limit error) returns success after retry; a stalled upstream raises `TimeoutError` after the configured budget. Tests pin both.
- Building a request with five `cache_control` markers raises a clear `ValueError` at construction time, not at the API call.

### Sequencing

#3 is the heaviest (~1.5 days). The others can run in parallel — they
touch independent code paths in the runners. Wave 6's pattern
(extract Pending dataclass, surface protocol method, runner refactor,
fake extension) is a good template for #3.

---

## Wave 11 — Deeper observability + verification

### Goal
Telemetry tells you the *shape* of a run, not just the events; the
tests cover the surfaces that vendor fakes can't reach; coverage gaps
become visible.

### Items

| # | Item | Approach |
| --- | --- | --- |
| 11 | No correlation IDs in `TelemetryEvent` | Add `trace_id: str` and `span_id: str | None` (parent) to `TelemetryEvent`. `Telemetry` (the recorder) issues a fresh `trace_id` per session, a fresh `span_id` per orchestrator turn / dispatch / speculation, and threads them through. Schema bump documented in `harness.telemetry`'s docstring. JSONL/Memory sinks ignore unknown fields gracefully — this is additive. |
| 10 | OTel sink has no span nesting | With #11 in place, `OpenTelemetrySink` builds an actual span tree: orchestrator → assistant turn → tool dispatch / speculation. Each event's `span_id` becomes the OTel span; `trace_id` ties them. Existing flat-event behavior stays as a fallback when correlation IDs are absent. |
| 18 | `harness debug --dap` has no end-to-end test | New `tests/debug/test_dap_cli.py`: spawn `uv run harness debug --dap <session>` as a subprocess, write framed DAP requests to its stdin, read framed responses from its stdout, assert the full launch → break → continue → terminated flow round-trips. Use `asyncio.create_subprocess_exec` so the test can interleave reads/writes. |
| 19 | No real-vendor integration tests | Cassette pattern: record one Anthropic and one OpenAI session against the real API (one-time, gated on `RECORD=1` env var), commit the recordings. CI replays them via a request-mocking layer (the same `FakeAsyncAnthropic` extended to read recorded responses). Catches SDK shape drift without paying for API calls every CI run. |
| 20 | No coverage tooling | Add `pytest-cov` to `[dev]` extras. CI runs `pytest --cov=harness --cov-report=term --cov-fail-under=85`. Coverage for `harness.runner.anthropic` likely lower than 85% given test coverage is fakes-only; raise threshold incrementally. Coverage doesn't hide the integration gap (#19), it surfaces it. |

### Acceptance

- A `harness.telemetry.MemorySink` after a session contains events with `trace_id` correlated; an OTel sink renders the same session as a parent span with child spans for each turn and tool call.
- `tests/debug/test_dap_cli.py` exercises the real `harness debug --dap` process end-to-end and is part of CI.
- Cassette-replay tests for both vendor runners are in CI; SDK breaking changes surface here, not in production.
- `pytest --cov` reports a number; CI fails on regression below the configured threshold.

### Sequencing

#11 unlocks #10 — do correlation IDs first. #18 and #20 are small and
independent; #19 is half-day-ish but needs care to keep recordings out
of the test suite's runtime path (replay should be fast).

---

## Wave 12 — Modality + streaming

### Goal
The orchestrator can do more than text-in/text-out: image inputs,
file references via the Anthropic Files API, and partial-output
streaming so callers see tokens as they arrive.

### Items

| # | Item | Approach |
| --- | --- | --- |
| 7 | No vision content blocks | Add `image` block type to `ContentBlock`: `image: ImageRef \| None = None` where `ImageRef` carries `source: Literal["base64", "url"] \| str` plus `media_type` and the data. `_translate_block_in` in both runners translates to vendor shapes. Tests pin round-trip of an inline base64 image + a URL-referenced image through both runners. |
| 8 | No Files API integration in AnthropicRunner | Replace the `<file path=...>` text inlining with a real `file` content block when the path resolves to an Anthropic Files API ID. Add `harness.prompts.upload_file(client, path) → file_id` helper that uploads via the client and returns the ID; `attach_file(file_id=...)` already exists, just plumb the ID through the translator. Old text-inlining stays as a fallback for non-Files-API paths. |
| 9 | No streaming text output | New `Orchestrator.run_stream(...)` that yields a stream of partial-message events: `TextDelta(text)`, `ToolUseStart(call)`, `ToolUseEnd(result)`, `MessageEnd(final)`. Built on the same event-iteration the runners already do (Wave 6 unblocked this). Existing `run()` becomes a thin wrapper that consumes the stream and returns the final message. CLI gets a `--stream` mode that prints deltas as they arrive. |

### Acceptance

- An agent can be passed a user message containing an inline image; the runner sends it to the model; the model's reply references the image. Tests pin the round-trip.
- `attach_file(file_id="file_...")` produces an Anthropic-API request with a `file` content block, not text inlining. Old path still works for `attach_file(path=...)` of small files.
- `async for event in orchestrator.run_stream(agent, messages):` yields `TextDelta` events as the model generates, with the same final `MessageEnd` as `run()`. The CLI's interactive mode shows the model thinking in real time.

### Sequencing

#7 first (smallest, unblocks vision examples). #9 needs Wave 6's
event iteration in place (already done) — it's an additive Orchestrator
method. #8 depends on the user wanting to upload, so it's gated on a
deliberate API-using example for a smoke test (the Files API has rate
limits and is not free). All three can land in the same wave but
should be three separate commits.

---

## Wave 13 — Speculator + privacy + DAP polish

### Goal
The two long-deferred items (ML privacy, eager per-block speculator
cancellation) ship, and the DAP surface gains the few features that
distinguish "spec-compliant subset" from "actually useful in the
editor."

### Items

| # | Item | Approach |
| --- | --- | --- |
| 1 | ML-based privacy detection | New `harness.privacy.PresidioDetector` under `[privacy-ml]` extra (`presidio-analyzer>=2.2`). Wraps Presidio's `AnalyzerEngine` behind the existing `Detector` protocol — `detect(text) -> list[Match]`. Pre-built `PRESIDIO_PII_PACK` mirrors `PII_PACK`'s shape but runs on Presidio recognizers. Ships docs comparing regex+entropy vs ML on the same corpus so users can pick. |
| 2 | Eager per-block speculator cancellation | When `max_speculations == 1` and a `tool_use` block arrives that doesn't match the lone pending speculation, cancel it immediately inside `observe()` instead of waiting for `cancel_unobserved`. For `max_speculations > 1`, keep the stream-end policy (correctness with multiple pending requires more bookkeeping than the win is worth). New test: 10s slow handler, single speculation, miss-shaped first block — assert handler is cancelled within ms of the first observe. |
| 15 | DAP `next` / `stepIn` / `stepOut` are aliases for continue | Re-frame the unit of "step." For agent trajectories, the natural granularities are: turn (assistant message produced), tool call (one `tool_use` dispatched), tool result (one tool returns). Map: `next` = step over a tool call (run the dispatch, pause before the next model call), `stepIn` = step into the tool's handler (set a temporary breakpoint for the next `PreToolUse`), `stepOut` = run to the next assistant message. Implementing requires a `step_mode` flag on `DebugRunner` and a one-shot break predicate. |
| 16 | DAP `pause` is a no-op | Implement: set a flag that the next `break_on` check (which fires before each `Runner` invocation) treats as "always break." Fires `stopped` with `reason=pause`. Editor's "pause" button starts working. |
| 17 | DAP `evaluate` is limited | Bring the DAP `evaluate` request to parity with the REPL's `inspect` command by routing through the same expression-resolution path the REPL uses, gated behind a launch-arg `allowEvaluate: true`. Default off (matches today's safety profile); explicit opt-in for users who want the REPL's full power in the IDE. Documented as an explicit security trade-off in the launch-config docs, mirroring the same opt-in posture the existing REPL inspect command has. |

### Acceptance

- An agent under `PrivacyBoundary([PresidioDetector(PRESIDIO_PII_PACK), ...])` redacts the same things as the regex pack plus Presidio's broader recognizers (people's names, addresses, etc.). Comparison harness is part of the test suite.
- A speculator with `max_speculations=1` predicting the wrong tool sees its pending task cancelled within milliseconds of the model emitting the actual call, not at end-of-stream. Pinned by a timing test.
- VS Code's "step over" / "step into" / "step out" buttons drive distinct, useful semantics in `harness debug --dap`. The `pause` button works.
- Editor users who set `"allowEvaluate": true` in their launch config can resolve arbitrary expressions over `ctx`; the default flow remains restricted to the variables-view names.

### Sequencing

#1 is independent. #2 needs care to not break the existing stream-end
policy when `max_speculations > 1`; the safe pattern is to make the
eager-cancel only fire on the single-spec case. #15 / #16 / #17 share
infrastructure (one-shot breakpoint policy on `DebugRunner`); land
them together.

---

## Persistent monitoring (not a wave)

These items don't get fixed; they get watched.

| # | Item | Watch |
| --- | --- | --- |
| 13 | `argparse.Namespace` `getattr` in `harness.debug.cli` | If a future flag is added to the parser and a programmatic test forgets it, the test will pass for the wrong reason. Mitigation: any new CLI flag added to the parser also gets added to the `argparse.Namespace(...)` constructions in the existing CLI tests. |
| 14 | DAP write-pipe uses `asyncio.streams.FlowControlMixin` | This is a known-flaky pattern across Python versions. Re-verify on each new Python release. Add a CI matrix entry for Python 3.13 (and the next 3.14) so a regression is caught the day it surfaces. |

---

## After 1.0 — speculative

Items that aren't in the 28-list but are worth flagging:

- **Multi-agent orchestration patterns**: `Orchestrator.run_parallel` exists; richer patterns (sub-agent spawning, supervisor/worker, voting) are unbuilt.
- **Memory backends beyond file**: SQLite, S3, Postgres. Each is a single-class implementation against the existing `MemoryStore` protocol.
- **Cookbook recipes**: short, narrative tutorials for common workflows (build-your-first-agent, redact-and-evaluate, debug-a-bad-trajectory). Examples are reference; cookbook is teaching.
- **Comparison with other agent libraries**: a docs page that places `harness-engineering` on the spectrum of LangChain / DSPy / AutoGen / CrewAI. Picky readers want this before they adopt.
- **Public benchmark suite**: a small set of recorded sessions for which the package's diff-eval matrix is published and trackable across versions.

These are the "1.x roadmap" — useful follow-ups that don't gate `1.0`.
