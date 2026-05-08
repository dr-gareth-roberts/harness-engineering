# Security policy

## Scope

`harness-engineering` is a library, not a service. Its security
posture is shaped by what you wire it to:

- **Tool dispatch** runs handlers you write — the library validates
  inputs against your Pydantic schemas, but it doesn't sandbox the
  handler. If a handler shells out, that's your sandbox to manage
  (see `harness.sandbox.safe_subprocess_run` for a starting point).
- **Vendor runners** (`AnthropicRunner`, `OpenAICompatRunner`) hold
  the API keys you supply. The library never logs them or sends them
  off-machine; SDK clients you pass in are used as-is.
- **Privacy boundary** (`harness.privacy.PrivacyBoundary`) is opt-in
  and best-effort. The detectors that ship are regex + entropy; they
  catch common patterns (high-entropy strings, structured PII) but
  are not a substitute for a real DLP system. Run with caution on
  high-stakes data.
- **Debug REPL** (`harness.debug.DebugRunner` interactive mode) lets
  the operator resolve arbitrary Python expressions over the paused
  context. This is intentional — it's a debugger — but the surface
  exists. The DAP adapter (`harness debug --dap`) does **not**
  expose this by default; it's restricted to a fixed set of variable
  names. Don't run the interactive REPL in untrusted contexts.

## Reporting a vulnerability

If you find a security issue, please report it privately:

- **GitHub Security Advisory**: <https://github.com/dr-gareth-roberts/harness-engineering/security/advisories/new>
  (preferred — the report stays embargoed until disclosure).
- **Email**: maintainer's email on the GitHub profile, with `[security]` in the subject line.

Please include:

1. A description of the issue and its impact.
2. A minimal reproduction.
3. Any suggested mitigation, if you have one.

We aim to:

- **Acknowledge** within 5 working days.
- **Triage** (confirm, scope, assign severity) within 14 days.
- **Patch and release** within 30 days for high-severity issues; less
  urgent issues are scheduled into the normal wave cadence (see
  [`progress.md`](progress.md) and [`docs/plan.md`](docs/plan.md)).

We coordinate disclosure: by default we hold off public disclosure
until a patched release is out, modulo a 90-day fallback if no fix
materializes.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.2.x | ✅ current — security fixes via patch releases |
| 0.1.x | ❌ |
| 0.0.x | ❌ — pre-release scaffolding |

Once `1.0` ships, we'll backport security fixes to the previous minor
version line for 6 months.

## Dependency risk

The package keeps a deliberately small dependency footprint:

- **Required**: `pydantic>=2.6`.
- **Optional extras**, each off by default and only pulled when the user
  opts in: `anthropic`, `openai`, `opentelemetry-api`,
  `opentelemetry-sdk`, `hypothesis`, `sentence-transformers`,
  `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`.

We don't pin dependency versions tighter than `>=` minor in
`pyproject.toml`; if a dependency announces a security advisory, the
fix is to upgrade your environment, not us.

## What this policy does not cover

- **Issues in your tool handlers** — those are application-level
  concerns. The library gives you `Pydantic` validation,
  `harness.sandbox` for filesystem/process scoping, and the policy
  module for tool-call gating; using them is up to you.
- **Issues in the LLM provider's API or model behavior** — report
  those to the provider directly.
- **Operational secrets in your own code** — if you commit API keys,
  that's a key-rotation issue, not a library bug.

## Threat model summary

For the curious, the explicit threat model the library is designed
against is in [`docs/architecture.md`](docs/architecture.md). The
short version: we trust the model to not be adversarial, we trust the
operator to not be adversarial, we don't trust the model's inputs
(hence the privacy boundary), and we don't trust tool inputs (hence
Pydantic validation + the sandbox primitives).
