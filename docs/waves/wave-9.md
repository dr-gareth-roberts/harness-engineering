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
