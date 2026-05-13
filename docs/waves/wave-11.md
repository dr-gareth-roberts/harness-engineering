## Wave 11 — Deeper observability + verification

### Goal
Telemetry tells you the *shape* of a run, not just the events; tests
cover the surfaces that vendor fakes can't reach; coverage gaps become
visible.

### Status
Shipped on `feature/wave-11-observability`. Four gaps cleared (#10,
#11, #18, #20); #19 (cassette pattern for vendor SDKs) deferred — it
needs real-API recordings (no API keys available in this environment),
and the existing `FakeAsync*` infrastructure already provides
scripted-response replay for SDK-shape testing.

### What landed

| # | Item | Implementation |
| --- | --- | --- |
| 11 | Correlation IDs | `TelemetryEvent` gains `trace_id` / `span_id` / `parent_span_id` (all optional). `Telemetry` recorder gets two `asynccontextmanager` APIs: `session_scope()` mints a 32-hex `trace_id` (OTel-compatible 128-bit), `span_scope()` mints a 16-hex `span_id` (OTel-compatible 64-bit) and snapshots the previous span as `parent_span_id`. State lives in `contextvars`, so nested + concurrent spans (think `asyncio.gather` over tool dispatches) get distinct IDs without manual threading. `Orchestrator.run()` opens session+span scopes; `Dispatcher.dispatch()` opens a nested span scope per call. Existing tests pass without modification because all three fields default to `None` and the recorder only fills them in when a scope is open. |
| 10 | OTel attribute promotion | `OpenTelemetrySink` removes correlation IDs from the reserved-fields list so `harness.trace_id` / `harness.span_id` / `harness.parent_span_id` ride as flat attributes on every emitted OTel event. Users can group / filter by `harness.trace_id` in Jaeger / Tempo / Honeycomb without the sink needing to synthesize spans itself. **Full span-tree synthesis is documented as deferred**: `tracer.start_span` calls the configured `IdGenerator` to mint the span_id, ignoring whatever harness span_id we hand it — round-tripping the harness IDs faithfully needs a custom `IdGenerator` (or lower-level span construction APIs) and isn't a short add. The conservative attribute path lands the data without lying about the structure. |
| 18 | DAP CLI subprocess test | `tests/debug/test_dap_cli.py` spawns the real `harness debug --dap` process via `asyncio.create_subprocess_exec`, writes framed DAP requests to its stdin, reads framed responses from its stdout, and asserts the full launch → setBreakpoints → configurationDone → launch → break → continue → terminated → disconnect flow round-trips. Validates the `connect_read_pipe` / `connect_write_pipe` plumbing the CLI uses for real editor integrations. ~0.4s wall-clock when warm. |
| 20 | Coverage tooling | `pytest-cov>=5` added to `[dev]` extras. `pyproject.toml` gets a `[tool.coverage.run]` (branch coverage on, source `src/harness`) + `[tool.coverage.report]` (`fail_under = 85`, sensible exclusions) section. CI runs `pytest --cov=harness`; below threshold = red. Current run reports **89% branch coverage** with all extras installed. |
| 19 | Cassette pattern | **Deferred** to a wave with API keys available. The `FakeAsync*` fakes already cover scripted-response replay; what's missing is the *recording* step against the real API, which requires credentials this environment doesn't have. Documented in the wave entry; format design lands when recordings can. |

### Tests added

| File | Count | Coverage |
| --- | --- | --- |
| `tests/telemetry/test_correlation.py` | 8 | emit-without-scope leaves IDs None; `session_scope` attaches trace_id; caller-supplied trace_id; reset after exit; `span_scope` attaches span_id and inherits trace_id; nested span_scope records parent; concurrent sibling tasks don't collide; orchestrator + dispatcher integration (full trace + parent linkage). |
| `tests/telemetry/test_otel.py` | +2 | correlation IDs ride as attributes; nested span_scope records `harness.parent_span_id`. |
| `tests/debug/test_dap_cli.py` | 2 | end-to-end DAP CLI subprocess flow; session record JSON validation. |

12 new tests, **522 total** (was 510). Coverage gate: 85% threshold,
89% actual.

### Verification gate

```
ruff check                       — clean
ruff format --check             — 171 files clean
mypy --strict src tests         — clean (157 source files)
pytest --cov=harness            — 521 passed + 1 skipped, 89% coverage (gate 85%)
mkdocs build --strict           — clean (~1s)
uv build                         — wheel + sdist build cleanly
```

### Deferred from this wave

- **Full OTel span-tree synthesis (#10 deeper)** — needs a custom
  `IdGenerator` or `Span` construction APIs to round-trip harness
  span_ids faithfully. The current attribute-promotion lands the
  correlation data without the structural lie.
- **Cassette pattern for vendor SDK shape drift (#19)** — needs
  real-API recordings. Schedule it for a wave with credentials.

### Commits

```
*  feat(telemetry): trace_id + span_id + parent_span_id correlation
*  feat(telemetry): OpenTelemetrySink promotes correlation IDs as attributes
*  test(debug): subprocess-driven harness debug --dap end-to-end test
*  feat(coverage): pytest-cov + 85% threshold gate in CI
*  docs: progress.md log of Wave 11
```
