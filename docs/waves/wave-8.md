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
docs/plan.md                     # forward plan from 0.2.0 to 1.0
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
a5fc470  docs: MkDocs scaffold + per-module API ref + [docs] extra
3b91549  docs: progress.md log of Wave 8
5660131  docs: forward plan from 0.2.0 to 1.0 (docs/plan.md)
```
