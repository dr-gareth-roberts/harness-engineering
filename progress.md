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
| Wave 9 | CI/CD + governance + housekeeping                  | shipped | (current — see below)                            |

**Status: 10 of 10 standout features shipped, plus Waves 5–9 polish.**
The forward plan from `0.2.0` to `1.0` lives in
[`docs/plan.md`](docs/plan.md): five waves (9 through 13), ~13–15
developer-days, every gap from the Wave 8 audit assigned to a wave.
Wave 9 just shipped (8 of those gaps cleared).

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


## Wave 9 — CI/CD + governance + housekeeping

### Goal
Move from "tests pass on my machine" to "the project's gate runs on
every PR, the package is installable from PyPI, the docs are reachable
on the web, and the contributor flow is documented." Mostly mechanical
work with high signal-to-noise — the first wave executed against the
forward plan in [`docs/plan.md`](docs/plan.md).

### Status
Shipped on `feature/wave-9-cicd-governance`. Eight items from the
28-gap audit cleared in one pass:
#21 (mypy on tests), #22 (CI), #23 (Pages), #24 (CHANGELOG), #25
(CONTRIBUTING), #26 (SECURITY), #27 (rotate progress.md), #28
(release / PyPI).

### Governance docs

| File | Contents |
| --- | --- |
| `CHANGELOG.md` | Keep-a-Changelog format. `0.2.0` entry distills Waves 1–8 into user-visible one-liners; `0.0.1` covers the MVP scaffold; `[Unreleased]` opens for Wave 10+ work. |
| `CONTRIBUTING.md` | dev setup (`uv sync --extra dev`), running the gate locally, building docs, commit conventions (imperative mood, no emoji, conventional-commits-style), PR expectations, releasing flow + one-time GitHub setup (PyPI trusted publisher + Pages source). |
| `SECURITY.md` | scope of the package's privacy/secret detection, responsible disclosure flow (GitHub Security Advisory preferred), 5-day-acknowledge / 14-day-triage / 30-day-patch SLOs, supported versions table, threat model summary, what the policy explicitly does *not* cover. |

### CI workflows

| Workflow | Trigger | What |
| --- | --- | --- |
| `.github/workflows/ci.yml` | PR + push to main | Matrix Python 3.11 / 3.12 / 3.13. Runs ruff check, ruff format check, **mypy --strict src tests**, pytest, mkdocs build --strict, uv build. Cancel-in-progress on rapid pushes. uv version pinned. |
| `.github/workflows/release.yml` | Tag `v*` | Re-run gate against the tagged commit, build wheel + sdist, publish to PyPI via OIDC trusted publishing. `pypi` environment matches the trusted-publisher config. No API token in the repo. |
| `.github/workflows/docs.yml` | Push to main | Build MkDocs site, upload as Pages artifact, deploy via `actions/deploy-pages@v4`. Requires Pages source = "GitHub Actions" (one-time setting). |

One-time GitHub-side setup is documented in `CONTRIBUTING.md` ("Releasing"
section): PyPI trusted publisher, Pages source switch.

### Housekeeping — progress.md rotation

Waves 3–7 (and now 8) rotated to `docs/waves/wave-{3..8}.md`, mirroring
the existing pattern where Waves 1–2 were already archived. progress.md
goes from ~898 lines (everything inline) to ~150 lines (status snapshot
+ cross-cutting decisions + current wave). The status snapshot table
gains archive-link rows for Waves 3–8.

### Hardening — `mypy --strict src tests` clean

The 62 mypy errors in `tests/` (pre-existing across Waves 1–8) cleared
to **zero**. Three categories:

| Category | Count | Fix |
| --- | --- | --- |
| `Tool.handler` variance: tests pass `Callable[[MySpecificModel], ...]` where the type alias declared `Callable[[BaseModel], ...]` | 19 | Widen `ToolHandler` to `Callable[[Any], ...]` — runtime validation via `input_model.model_validate(...)` is the actual contract; type-laxity matches reality. Documented inline. |
| `lambda e: list.append(e) or None` triggering `func-returns-value` on the LHS of `or` | 8 | Drop the redundant `or None` — `append` returns None implicitly, satisfying hook handler signature. |
| Stale `# type: ignore` comments mypy now flags as unused | 6 | Remove. |
| Other (missing type args, missing isinstance narrowings, missing imports, narrow `direction: Literal[...]`, etc.) | 29 | Surgical fixes per call site. |

The CI gate now runs `mypy --strict src tests` (was `src/harness` only).

### Verification

```
uv build                                 — wheel + sdist build cleanly
uv run mkdocs build --strict             — clean, ~1s
uv run ruff check                        — clean
uv run ruff format --check               — 171 files clean
uv run mypy --strict src tests           — clean (156 source files)
uv run pytest -q                          — 495 passed
```

### Commits

```
*  chore(governance): CHANGELOG + CONTRIBUTING + SECURITY
*  ci: PR/push gate + tag-driven PyPI release + docs deploy workflows
*  chore(progress): rotate Waves 3-8 to docs/waves/, keep prelude inline
*  chore(types): mypy --strict src tests clean (62 errors -> 0)
*  docs: progress.md log of Wave 9
```
