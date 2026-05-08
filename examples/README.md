# Examples

Each file in this directory demonstrates one harness module end-to-end.
The convention every example follows:

- **No real API calls.** Use `EchoRunner`, `CannedRunner`, or a small
  inline fake. The two existing exceptions are `anthropic_runner.py`
  (gated on `ANTHROPIC_API_KEY` — only run if you set it) and the
  in-process smoke `end_to_end.py` which already runs without a key.
- **`async def main() -> int`** as the entry point. Returns the exit
  code. Top-level: `if __name__ == "__main__": raise SystemExit(asyncio.run(main()))`.
- **Module docstring with a "Run with:" line** so readers can copy-paste.
- **Print a transcript** so the example's behavior is visible without
  reading the code. The smoke tests in `tests/examples/` rely on this.
- **Vendor-neutral.** No `claude-*` or `gpt-*` model names where the
  example doesn't need them; pass `model="demo-model"` and let the
  fake runner ignore it.
- **Self-contained.** No external network, no environment variables
  (except the one explicit gate above), no fixture files.

| File | Module / feature | Highlight |
| ---- | ---------------- | --------- |
| `end_to_end.py` | Core (tools / hooks / policy / agents) | The 60-second tour — every base module wired together with a fake runner. |
| `anthropic_runner.py` | `harness.runner.anthropic` | Real API smoke test, gated on `ANTHROPIC_API_KEY`. |
| `contracts.py` | `harness.contracts` | `Never` / `Always` patterns; same DFA powers runtime `attach_contracts` and offline `check`. |
| `counterfactual.py` | `harness.replay.counterfactual` | Mutate a recorded `SessionRecord` and continue from the divergence point. |
| `diff_eval.py` | `harness.replay.diff_eval` | Run the same cases against multiple runners; `DiffMatrix` surfaces dissenters. |
| `fuzz.py` | `harness.fuzz` | Hypothesis-driven fuzzing of a tool surface. |
| `attribute.py` | `harness.attribute` | Leave-one-out ablation over a session to attribute outputs to inputs. |
| `cache.py` | `harness.cache` | Prefix-drift watcher fingerprints + `audit` over a fingerprint store. |
| `privacy.py` | `harness.privacy` | `PrivacyBoundary` redacting PII before it crosses the runner boundary. |
| `plan.py` | `harness.plan` | `PlanGuardedRunner` enforcing a plan; `PlanViolation` on deviation. |
| `plan_inference.py` | `harness.plan.infer` | Mine a `Plan` from past `SessionRecord`s. |
| `debug.py` | `harness.debug` | `DebugRunner` programmatic breakpoint with mutation + resume. |
| `speculate.py` | `harness.speculate` | `Speculator` with `LastCallPredictor`; wall-clock parallelism demo. |
| `cross_session.py` | `harness.speculate.cross_session` | `CrossSessionPredictor` loaded from a `MemoryStore`. |
| `otel.py` | `harness.telemetry.otel` | `OpenTelemetrySink` adding events to the current span (uses an in-memory exporter). |

## Run them

```bash
# All of them, one at a time:
for f in examples/*.py; do echo "=== $f ==="; uv run python "$f" || break; done

# Or just one:
uv run python examples/contracts.py
```

## Test them

Every example is also exercised by `tests/examples/test_examples_run.py`,
which imports each module and calls its `main()`. If you change an
example, run that test to confirm the rewrite still completes:

```bash
uv run pytest tests/examples -q
```

The smoke test treats a non-zero exit code or an unhandled exception
from `main()` as a failure. Add a new example by dropping a
`whatever.py` here that follows the conventions above plus a one-line
entry in `tests/examples/test_examples_run.py`'s `EXAMPLES` list.
