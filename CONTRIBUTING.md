# Contributing

Thanks for your interest. The package is small and self-contained, so
the contributor flow stays light.

## Dev setup

We use [`uv`](https://github.com/astral-sh/uv) for environment and
dependency management. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone and sync:

```bash
git clone https://github.com/dr-gareth-roberts/harness-engineering
cd harness-engineering
uv sync --extra dev
```

To work on a feature that touches an optional extra
(`anthropic`, `openai-compat`, `otel`, `fuzz`, `attribute`), add the
extra to the sync:

```bash
uv sync --extra dev --extra anthropic --extra otel
```

## Running the gate

The same checks CI runs, available locally:

```bash
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy --strict src/harness
uv run pytest
```

Optional but encouraged before opening a PR:

```bash
uv run mypy --strict src tests              # extends the gate to test code
uv run mkdocs build --strict                # docs build (needs --extra docs)
uv build                                     # wheel + sdist build
```

The local gate is the source of truth — if these pass and CI fails,
that's a CI bug, not yours.

## Running the docs

```bash
uv sync --extra docs
uv run mkdocs serve
```

Browse to `http://localhost:8000`. Hot-reload on file save.

## Commit conventions

- **Imperative mood**: `feat(speculate): add observe + cancel_unobserved` — not `Added` or `Adds`.
- **Scope in parens** when one applies: `feat(runner)`, `fix(privacy)`, `docs(examples)`, `test(dap)`, `chore`.
- **One concern per commit**. Tests that go with a feature commit live in the same commit; an unrelated typo fix lives in its own.
- **No emoji** in commit messages or in code (the project's existing convention; see global instructions in the repo).
- **Co-authored-by**: include any reviewers or pair-programming partners with `Co-Authored-By: Name <email>` in the trailer.

## Pull requests

- **One concern per PR**. If your branch grew to multiple unrelated features, split it.
- **Run the gate locally first**. The gate is fast (< 1 minute on a warm cache); CI shouldn't be how you discover a typo.
- **Update `progress.md`** if your change spans a wave or warrants a decision log entry. Otherwise, the commit message is enough.
- **Update `CHANGELOG.md`** under `[Unreleased]` with a one-line user-visible summary, if the change has user-visible impact.
- **Update docstrings, not just code**. Several modules (`harness.runner`, `harness.speculate`, `harness.debug`) have detailed module docstrings that drive the docs site via `mkdocstrings`. Keep them in sync.
- **Don't add dependencies casually**. Every new dependency is one more thing that can break. If you need one, justify it in the PR description.

## Project conventions

- **Python 3.11+**, type hints everywhere, `mypy --strict` clean on `src/`.
- **Line length 100** (`ruff format` enforces).
- **Double quotes in Python**, single quotes in TS/JS (no TS/JS in this repo).
- **Functional style where appropriate**, immutability by default, dependency injection over globals.
- **Explicit error handling**. No silent failures, no `except: pass` (use `contextlib.suppress` if you really must).
- **Test behavior, not implementation**. Mock at boundaries (APIs, databases), not internal code.

## Code review checklist

A handful of rules earned from repeated audit findings. Apply them when
reviewing your own PRs.

### Remove what doesn't earn its keep

A kwarg, parameter, or attribute that the implementation ignores is a bug,
not a "reserved for future use." Either honor it or remove it.

The 1.0.3 audit caught four instances of dead kwargs (`PrivacyBoundary.on_detect`,
`OpenTelemetrySink.tracer`, `derive_plan.plan_schema`, `Dispatcher.dispatch.start`).
Each was removed or wired up. New code should not introduce more.

This discipline keeps the public surface honest: every documented argument
either changes behavior or fails loudly, and users can't build on behavior
that doesn't exist. When in doubt, delete — re-adding a kwarg later is a
minor release; removing a kwarg that someone depended on is a major one.

## Releasing (maintainer-only)

1. Update `CHANGELOG.md`: move `[Unreleased]` items under a new version heading, set the date.
2. Bump `version` in `pyproject.toml` and `__version__` in `src/harness/__init__.py`.
3. Commit with `chore: release vX.Y.Z`, tag with `git tag vX.Y.Z`, push tag.
4. The release workflow (`.github/workflows/release.yml`) runs the gate, builds the wheel + sdist, and publishes to PyPI via OIDC trusted publishing — **no API token in the repo**.
5. The docs workflow deploys the new docs site on the same `main` push.

### One-time GitHub setup (maintainer-only)

These need to be configured once in repo settings, after the first
landing of the workflow files:

- **PyPI trusted publisher** — at <https://pypi.org/manage/account/publishing/>, configure a trusted publisher for the project name `harness-engineering` with: owner `dr-gareth-roberts`, repository `harness-engineering`, workflow `release.yml`, environment `pypi` (the workflow declares this environment).
- **GitHub Pages source** — Settings → Pages → Source: GitHub Actions. The first push to `main` after `docs.yml` lands deploys the site.

## Reporting bugs and security issues

- **Bugs**: open a GitHub issue with a minimal repro.
- **Security**: see [SECURITY.md](SECURITY.md). Do **not** open a public issue for security issues.

## License

By contributing, you agree your contributions are licensed under
[Apache-2.0](LICENSE), the project's license.
