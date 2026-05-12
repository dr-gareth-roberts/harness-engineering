# FAQ

Common questions, common pitfalls, and "why does X behave that way?"
answers. Skim the headings; each answer is short.

## Getting started

### Do I need an API key to try the library?

No. The base install ships `EchoRunner` and `CannedRunner` which
return deterministic responses without any model. The
[Quickstart](quickstart.md) builds a working agent on
`CannedRunner` first, then upgrades to Anthropic if you want.

### Why is `pip install harness-engineering-toolkit` so small?

Because the base dependency is `pydantic>=2.6`. Vendor SDKs and
heavy libraries (Anthropic, OpenAI, Hypothesis, sentence-transformers,
OpenTelemetry, Presidio) are *opt-in extras*. If you need Anthropic,
`pip install 'harness-engineering-toolkit[anthropic]'`. If you need privacy
ML, `[privacy-ml]`. The full list is on the [Home](index.md) page.

### Which runner should I use?

| Situation | Runner |
|---|---|
| Tests / demos / docs | `CannedRunner` (scripted replies) or `EchoRunner` (echoes input) |
| Production with Anthropic Claude | `AnthropicRunner` (`[anthropic]`) |
| Production with OpenAI / Mistral / Together / Groq | `OpenAICompatRunner` (`[openai-compat]`) |
| Local model via Ollama / vLLM / llama.cpp / LM Studio | `OpenAICompatRunner` with `base_url=` pointed at your server |
| Replay a recorded session | `ReplayRunner.from_record(...)` |
| Wrap any of the above with breakpoints | `DebugRunner(real_runner, ...)` |
| Wrap any of the above with a plan-as-contract | `PlanGuardedRunner(real_runner, plan)` |
| Wrap any of the above with PII redaction | `PrivacyBoundary(...).wrap(real_runner)` |

The wrapper runners compose freely. The model layer stays oblivious.

## Tools and dispatching

### Why does my tool handler raise but the model still gets a response?

`Dispatcher.dispatch` catches handler exceptions and surfaces them
as `ToolResult(is_error=True, content=str(exc))`. The model sees
the error message and can decide what to do (retry, ask for
clarification, give up). If you want the exception to propagate
out of `dispatch`, that's currently not the contract — file an issue
if you have a use case.

### Can my handler be async?

Yes. The dispatcher detects awaitable returns (`inspect.isawaitable`)
and awaits them. Sync handlers work too. Don't mix awaitable and
non-awaitable returns from the same handler — the type system
catches that.

### How do I make a tool's input optional?

Standard Pydantic — make the field optional with a default:

```python
class SearchIn(BaseModel):
    query: str
    limit: int = 10
```

The model sees the schema with `limit` defaulted; calls without
`limit` validate fine.

### What's the `idempotent=True` flag for?

The speculator (`harness.speculate`) only pre-executes tools marked
idempotent. The flag is a *promise* by the tool author: re-running
this tool with the same arguments is observably equivalent to
running it once. Read-only tools (search, look up, fetch) qualify;
side-effecting tools (send_email, delete_record) don't.

If you mark a side-effecting tool idempotent, speculative miss runs
cause silent duplicate side effects. Mark a tool idempotent only
if you can defend that promise.

## Hooks and policies

### When should I use a hook vs a tool?

| Need | Use |
|---|---|
| The model can call this and read the result | Tool |
| Something fires *around* a tool call (audit, blocking, replacement) | `PreToolUse` / `PostToolUse` hook |
| Run code at the start / end of a session | `SessionStart` / `SessionEnd` hook |
| Observe every assistant message produced | `PostAssistantMessage` hook |
| Allow / deny tools by name or argument shape | `harness.policy` (which attaches as a `PreToolUse` hook) |

### Can a hook modify what the model sees?

Yes. Returning `HookDecision(replacement=ToolResult(...))` from a
`PreToolUse` hook short-circuits dispatch with the supplied result;
from a `PostToolUse` hook, it rewrites the result before it goes
back to the model. Typical use: redact a secret in the result
before the model retains it.

### Can hooks be async?

Yes. `HookRunner.register` accepts both sync and async callables.

## Privacy

### Regex+entropy or Presidio?

Both. Pick by what you're catching:

- **Regex+entropy** (built-in, zero deps): bounded shapes you know
  ahead of time — SSN, AWS keys, GitHub tokens, high-entropy
  strings. Predictable, reproducible, ~zero overhead.
- **Presidio** (`[privacy-ml]`, ~50ms/scan + spaCy model): broad
  PII shapes you can't enumerate — names, addresses, phone numbers,
  dates of birth. Looser semantics; some false positives.

The [Cookbook recipe](cookbook/redact-pii.md) shows both side by
side.

### What direction is the boundary scanning?

It depends on the pack — the defaults match each pack's threat model:

- `SECRET_PACK` (AWS / Anthropic / GitHub / Stripe tokens) defaults
  to `direction="both"`, `action="block"`. Secrets crossing the
  boundary in *either* direction is almost always a bug, so the
  pack stops the call rather than silently redacting.
- `PII_PACK` (US SSN / phone / email) defaults to
  `direction="outbound"`, `action="redact"`. PII is more often a
  content issue than a security one; redaction keeps the model
  usable.
- `HIPAA_PACK` (MRN / NPI / ICD-10) defaults to
  `direction="outbound"`, `action="redact"` (ICD-10 is
  `action="audit"` because legitimate clinical text contains it).

Each detector exposes `direction` (`"outbound"` / `"inbound"` /
`"both"`) and `action` (`"block"` / `"redact"` / `"audit"`); flip
them if your threat model differs.

## Replay and testing

### Do replayed sessions hit the real API?

No. `ReplayRunner.from_record(record)` is input-blind: it returns
the recorded assistant messages in order, with no API call and no
tool dispatch of its own. Tool handlers only fire if the recorded
messages contain `tool_use` blocks *and* you wrap ReplayRunner in
something that actually dispatches them — a bare Orchestrator does
not. The recorded `tool_result` messages are part of the playback,
so the model side of the trajectory replays faithfully without
needing the handlers to run again.

### Can I run pytest without the optional extras installed?

Yes. The test suite uses `pytest.importorskip` for tests that
require optional extras. CI runs with all extras installed for
maximal coverage; local "I just changed `harness.tools`" runs work
fine on the base install + `[dev]`.

### Why does `diff_eval` show different tool_use IDs across providers?

Anthropic generates `toolu_01...`, OpenAI generates `call_...`.
`diff_eval` and `compare_sessions` ignore IDs in their comparison so
cross-provider verdicts stay meaningful. The IDs are still on the
recorded `SessionRecord`s.

## Debugging

### `harness debug --dap` — what does the editor see?

A pseudo-source called `trajectory` with one line per assistant
turn. DAP line N maps to `ctx.turn_index == N - 1`. Setting a
breakpoint at line 3 pauses just before producing the 3rd
assistant turn. The Variables view shows `turn_index`,
`message_count`, `last_call.name`, `last_call.arguments`,
`pending_mutation.role`. The Source view fetches the synthesized
trajectory.

### Can I run arbitrary Python over `ctx` from the editor's debug console?

Off by default. Set `allowEvaluate: true` in the launch arguments
to enable. Same security trade-off as the REPL's `inspect` command
— this is a debugger; only reachable in an opt-in debug session.

### `step_in` doesn't go into the tool handler — why?

Today, `step_in` uses the same per-turn `step_over` semantics: the
runner resumes from the current breakpoint and pauses again at the
next break opportunity (typically the next iteration of the
tool-use loop). A finer "step into the tool handler" granularity
needs a one-shot pre-tool-use breakpoint that the DebugRunner
doesn't yet expose — tracked as a follow-up. `next`, `pause`, and
`stepOut` all behave per-turn for the same reason.

## Speculator

### When does speculation actually help?

When your tool handlers have non-trivial latency (DB lookup,
external API, file I/O) and the next tool call is predictable
from history. The speculator runs the predicted call concurrently
with model generation; on hit, the runner skips the handler
runtime entirely. On miss, the speculation is cancelled — at
stream-end at the latest, eagerly inside `observe()` for
`max_speculations=1`.

If your handlers are sub-millisecond, speculation overhead exceeds
the win. Don't speculate for the sake of it.

### Does the speculator run my non-idempotent tools?

No. By default, only `Tool.idempotent=True` tools are eligible.
Override with `Speculator(..., only_idempotent=False)` if you've
audited every tool's side-effect profile and explicitly want
non-idempotent speculation.

### Speculator + streaming — do they work together?

Yes. The speculator API (`begin` / `observe` / `cancel_unobserved`
/ `try_resolve` / `end`) is honored in both `Runner.__call__` and
`run_stream`. The streaming path preserves the same cancellation
timing — eager per-block cancellation when
`max_speculations == 1`, end-of-stream cleanup otherwise.

## Telemetry

### Does `OpenTelemetrySink` create OTel spans?

No. It emits each `TelemetryEvent` as a flat `Event` on whatever
span is currently active in the OTel context. Wrap your call in a
real span (FastAPI middleware, instrumented HTTP client, etc.) so
the events have a parent. When no instrumented caller is active,
`add_event` is a no-op on OTel's `NonRecordingSpan` — silent loss,
by design. Synthesizing OTel spans from harness events would
require a custom `IdGenerator`; tracked as deferred.

### How do I correlate events from one session?

The `Telemetry` recorder auto-generates `trace_id` per session and
`span_id` per turn / dispatch / speculation via `contextvars`. Each
`TelemetryEvent` carries them. Group / filter on `harness.trace_id`
in your backend.

### Can I propagate an upstream `trace_id`?

Yes:

```python
async with telemetry.session_scope(trace_id=request.headers["x-trace-id"]):
    await orchestrator.run(...)
```

The orchestrator's auto-opened scope respects the supplied ID.

## Performance

### What's the per-event overhead?

Single-digit microseconds for a `MemorySink` emit; the bulk of
`Telemetry.emit` is Pydantic field validation. Use `NullSink` (the
default if you don't pass one) to skip emission entirely.

### Why does the first agent run feel slow?

If you're using an extra that loads ML models (`[attribute]`,
`[privacy-ml]`), the first scan loads the model (~50MB-1GB). Subsequent
scans are warm. Pre-construct the detector / similarity at startup
if cold-start latency matters.

### Can I run the orchestrator in a synchronous context?

Wrap with `asyncio.run(...)`. The orchestrator and runners are
async-first. There's no sync facade today.

## Operations

### How do I cut a release?

Tag and push:

```bash
git tag v1.2.3
git push origin v1.2.3
```

`.github/workflows/release.yml` runs the gate, builds wheel + sdist,
publishes to PyPI via OIDC trusted publishing. No API token in the
repo. See [`CONTRIBUTING.md`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/CONTRIBUTING.md)
for the one-time PyPI / GitHub setup.

### Where do I report a security issue?

Don't open a public issue. See
[`SECURITY.md`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/SECURITY.md):
GitHub Security Advisory or email the maintainer with `[security]`
in the subject.

### What Python versions are supported?

3.11, 3.12, 3.13. CI runs the full matrix. 3.10 and below are
unsupported (the codebase uses `X | None` types and other 3.11+
syntax extensively).

## Where to read next

- [**Quickstart**](quickstart.md), [**Cookbook**](cookbook/index.md)
  — concrete code paths.
- [**Architecture**](architecture.md), [**Comparison**](comparison.md)
  — the design model and where it fits relative to alternatives.
- Module reference — start at [`tools`](modules/tools.md) and walk
  the nav.
