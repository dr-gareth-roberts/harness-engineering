# Docs-fix plan — post-Codex audit (2026-05-11)

## Context

The docs overhaul shipped in `da2607d` produced a docs site that
**reads well but lies in the code examples.** A Codex audit
(session `019e159d`, log `/tmp/codex-docs-review.out`) cross-checked
every cookbook recipe + module page against `src/harness/` and
found ~12 distinct issues. The root cause is the same in every
case: I wrote the examples from memory of the codebase rather than
opening the source. Several issues are *one wrong kwarg name*;
several are *wrong constructor shape entirely*; one is *wrong
direction/action on a pre-built privacy pack*. All of them would
fail at import or call time for an evaluator who copy-pastes.

This plan is the **fix work**: catalog every issue, decide
"fix the docs to match source" vs "fix the source to match docs"
per issue, then sequence the edits and the long-term CI gating
that stops this drift in the future.

The plan is **read-and-approve first, edit second.** Nothing
shipped from this branch yet; the deliverable here is the plan
itself. Once approved, the editing happens in a follow-up wave.

## Issue catalog

Each issue is `[severity][type] file:line — finding`.

Severity: **P0** (first-run examples fail), **P1** (cookbook /
module examples fail), **P2** (consistency / staleness, no failure
but trust erosion), **P3** (wording).

Type: `API` (signature wrong), `STALE` (was true once),
`POLICY` (defaults differ from docs claim), `WORDING` (true but
dismissive), `DOC-COMMENT` (source docstring lies).

### P0 — first-run failures

1. **[P0][API] docs/quickstart.md:33-43, docs/index.md:53-56** —
   imports `text`, `PostToolUse`, `PreToolUse` from top-level
   `harness`. Top-level `__init__.py` exports `Message` but not
   `text`, `PreToolUse`, `PostToolUse`, etc. ([src/harness/__init__.py:93-175](src/harness/__init__.py),
   [src/harness/prompts/__init__.py:1-23](src/harness/prompts/__init__.py),
   [src/harness/hooks/__init__.py:1-29](src/harness/hooks/__init__.py)).
   Decision needed: **(A)** add these to top-level so the docs
   work as written, or **(B)** change the docs to import from
   submodules.

### P1 — cookbook / module examples fail

2. **[P1][API] docs/cookbook/replay-evaluation.md:47-100** — uses
   `EvalCase` from top-level, calls the `run_eval` helper with
   wrong kwargs, calls `matrix.write_html()` where the real name
   is `report_html()`, accesses outlier fields that don't exist,
   and constructs `RewriteTurn` with `turn_index=..., new_text=...`
   kwargs which don't match the real dataclass. Multiple wrong
   signatures in one recipe ([src/harness/replay/harness.py:46-51](src/harness/replay/harness.py),
   [src/harness/replay/diff_eval.py:49-68](src/harness/replay/diff_eval.py),
   [src/harness/replay/diff_eval.py:134](src/harness/replay/diff_eval.py),
   [src/harness/replay/counterfactual.py:39-49](src/harness/replay/counterfactual.py)).

3. **[P1][API] docs/modules/memory.md:20-34** — the `Session`
   constructor is documented as `Session(store=..., agent=...)`.
   Real constructor is `Session(orchestrator, agent, store)`.
   `session.run` should be `session.send`. `session.id` should
   be `session.session_id`
   ([src/harness/memory/session.py:26-61](src/harness/memory/session.py)).

4. **[P1][API] docs/modules/plan.md:3-37** — uses
   `PlannedToolCall(name=..., required_args=...)` and
   `PlanGuardedRunner(..., mode=...)`. Source uses
   `PlannedToolCall(tool_name=..., arguments_match=...,
   arguments_regex=...)`; `mode` belongs on `Plan`, not on
   `PlanGuardedRunner` ([src/harness/plan/plan.py:40-105](src/harness/plan/plan.py),
   [src/harness/plan/guard.py:48-50](src/harness/plan/guard.py)).

5. **[P1][API] docs/modules/sandbox.md:20-52** — sandbox docs are
   "mostly wrong." `PathScope(roots=..., policy=...)` doesn't
   exist. There's no `resolve()` method. `safe_subprocess_run` is
   async; I documented it as sync. `scrub_env` takes `allow_keys`,
   not `keep` ([src/harness/sandbox/paths.py:22-75](src/harness/sandbox/paths.py),
   [src/harness/sandbox/process.py:16-58](src/harness/sandbox/process.py)).

6. **[P1][API] docs/cookbook/fuzz-a-tool.md:87-106,
   docs/modules/attribute.md:24-31** — `fuzz_agent` needs
   `tool_name`. `harness_property` needs `dispatcher` and `tool`
   kwargs. Attribution takes `target_message_index`, not `target`
   or `target_match` ([src/harness/fuzz/runner.py:152-162](src/harness/fuzz/runner.py),
   [src/harness/fuzz/decorators.py:25-31](src/harness/fuzz/decorators.py),
   [src/harness/attribute/ablation.py:275-285](src/harness/attribute/ablation.py)).

7. **[P1][API] docs/cli.md:83-85,
   docs/cookbook/cache-and-speculate.md:47-49** — `harness
   cache-audit` is documented as `path --window-hours N`; real
   invocation is `--store PATH --since 24h` ([src/harness/cache/cli.py:43-64](src/harness/cache/cli.py)).

### P2 — consistency / staleness

8. **[P2][STALE] SECURITY.md:53-62** — says "0.2.x is current,
   Once 1.0 ships..." We shipped 1.0.0 on 2026-05-10
   ([pyproject.toml:5-12](pyproject.toml),
   [CHANGELOG.md:10-16](CHANGELOG.md)).

9. **[P2][POLICY] docs/faq.md:122-128, docs/cookbook/redact-pii.md:63-80** —
   says pre-built packs default outbound-only redact.
   `SECRET_PACK` is actually `direction="both"` and `action="block"`
   ([src/harness/privacy/packs.py:24-53](src/harness/privacy/packs.py)).
   Decision needed: **(A)** change docs to describe the
   actual posture, **(B)** change `SECRET_PACK` to match the
   documented posture (redact, outbound).

10. **[P2][DOC-COMMENT] src/harness/debug/dap.py:25-41, 320-345** —
    module docstring + nearby comments still say
    `next`/`stepIn`/`stepOut` are "treated as continue," but Wave
    13b's `_step_mode` flag wired distinct semantics. Source-side
    docstring fix, not a docs/ fix (but the contradiction
    surfaces to FAQ readers).

11. **[P2][API] docs/modules/replay.md:54-56** — claims
    `ReplayRunner` "re-runs your tool handlers by default."
    Source: `ReplayRunner` is input-blind and returns recorded
    assistant messages — handlers do NOT re-run
    ([src/harness/replay/runner.py:14-47](src/harness/replay/runner.py)).

12. **[P2][API] docs/modules/fuzz.md:56-59** — claims the
    strategy bridge handles "primitives + Optional + Literal +
    list[X] + basic dict." Source: bridge explicitly **excludes**
    lists/dicts without overrides
    ([src/harness/fuzz/strategies.py:3-13](src/harness/fuzz/strategies.py)).

13. **[P2][API] docs/modules/attribute.md:57-59** — "target
    matched as substring of the assistant response by default;
    pass `target_match=...`" — invented. The real API takes
    `target_message_index` ([src/harness/attribute/ablation.py:275-285](src/harness/attribute/ablation.py)).

### P3 — wording

14. **[P3][WORDING] docs/comparison.md:36-55** — "Pick LangChain
    … if you don't mind giving up type safety" is too
    dismissive. Modern LangChain has structured output via
    Pydantic, LangGraph durability, LangSmith observability.

15. **[P3][WORDING] docs/comparison.md:81-90** — AutoGen
    paragraph uses `UserProxyAgent` / `GroupChatManager`
    terminology where current AutoGen emphasizes
    `AgentChat` + agents + teams.

16. **[P3][WORDING] docs/comparison.md:138-141** — claims
    "cassette-replay" as a strength when the cassette tests are
    actually deferred in the roadmap
    ([docs/roadmap.md:39-41](docs/roadmap.md)).

17. **[P3][WORDING] docs/faq.md:95, docs/roadmap.md:7-33** —
    internal "Wave 10 #5" language readable to maintainers,
    noise to evaluators. Strip for the public-facing pages.

## Decisions needed before edits

For each decision, the answers determine which file gets edited
(docs vs source) and the shape of the edit. Asking the user once
keeps the work coherent.

### D1 — Top-level prompts/hooks exports (#1)
**Option A:** Add `text`, `PreToolUse`, `PostToolUse`,
`PostAssistantMessage`, `SessionStart`, `SessionEnd`,
`PromptSubmit`, `Stop` to `harness/__init__.py`. Quickstart
"just works" with `from harness import text`. Bigger top-level
namespace (already broad — Codex flagged this as a smell).
**Option B:** Change docs to `from harness.prompts import text`
and `from harness.hooks import PostToolUse`. Smaller top-level,
more import lines per example.

**Recommendation: A.** The cost of a slightly broader top-level
is paid once; evaluators see the cost of B on every example.

### D2 — `SECRET_PACK` direction/action (#9)
**Option A:** Change docs to match source: "`SECRET_PACK`
defaults to `direction='both'` + `action='block'` — secrets are
treated as fatal, in either direction. Use `PII_PACK` (outbound
redact) for less-critical PII." Honest about the design
intent.
**Option B:** Change `SECRET_PACK` to `direction='outbound'`,
`action='redact'` to match the docs. Aligns with the
"don't-leak" mental model the docs describe.

**Recommendation: A.** The source's choice is defensible (real
secrets warrant blocking), and changing it would be a behavior
change for existing users.

### D3 — `Session` constructor (#3)
**Option A:** Change docs to use real signature
`Session(orchestrator, agent, store)`. Real method `send()`,
real field `session_id`.
**Option B:** Refactor `Session` to take kwargs
`(store=, agent=)`. Friendlier construction. Breaking change.

**Recommendation: A.** Source ships at 1.0.0; breaking
construction is not a doc-fix.

### D4 — CI gating against future drift
**Option A:** `pytest-codeblocks` (or
`pytest-doctest-codeblocks`). Run every fenced code block in
docs through Python. Catches API drift mechanically.
**Option B:** Treat `examples/*.py` as canonical demos, embed
into docs via mkdocs-material's `--8<--` snippet syntax. Docs
contain the same code that runs in `tests/examples/`.
**Option C:** Both: A for inline snippets in docs/,
B for the canonical end-to-end example per module.

**Recommendation: C.** A alone leaves the cookbook recipes
fragile because they're contrived; B alone leaves module pages
with snippets that aren't tested. C is one extra dependency
(`pytest-codeblocks`) + a small mkdocs plugin install +
careful authoring.

## Fix plan (post-decisions)

Sequence after the 4 decisions land:

### Phase 1 — Source canonicality (1 commit)

Touch source for items where source is the canonical truth:

1. (D1=A) Re-export `text`, hook event types into
   `harness/__init__.py`. Add to `__all__`. Verify
   `pytest --cov=harness` stays green.
2. (#10 / D-side) Update `src/harness/debug/dap.py` module
   docstring + inline comments at lines 25-41 and 320-345 to
   describe the actual Wave 13b step semantics.

### Phase 2 — Docs surgery (1 commit per file group)

Each commit fixes one file or one tightly-coupled group:

3. **`docs/quickstart.md`, `docs/index.md`** — fix imports per
   D1; verify each fenced block parses with `python -c` ad-hoc.
4. **`docs/modules/memory.md`** — rewrite the `Session` example
   to match the real constructor + methods.
5. **`docs/modules/plan.md`, `docs/modules/sandbox.md`,
   `docs/modules/replay.md`, `docs/modules/fuzz.md`,
   `docs/modules/attribute.md`** — five module pages with API
   corrections. Each gets its example rewritten against the
   source. Single commit if they're all editorial-only; split
   if any require deeper rework.
6. **`docs/cookbook/replay-evaluation.md`,
   `docs/cookbook/fuzz-a-tool.md`,
   `docs/cookbook/cache-and-speculate.md`** — three cookbook
   recipes with the bulk of the wrong APIs. Same per-file pass.
7. **`docs/cli.md`** — `cache-audit` invocation.
8. **`docs/faq.md`** — privacy pack posture, replay handler
   side effects, DAP step semantics alignment.
9. **`docs/comparison.md`** — soften LangChain/AutoGen
   wording; remove "cassette-replay" claim from strengths;
   align with current alternatives' state per Codex's
   verified-2026-05-11 cite list.
10. **`docs/roadmap.md`, `docs/faq.md`** — strip "Wave 10 #5"
    style internal language. Replace with feature names.
11. **`SECURITY.md`** — bump versioning table to 1.0.x, drop
    "Once 1.0 ships" framing.

### Phase 3 — CI gating (1 commit, D4=C)

12. Add `pytest-codeblocks` (or `pytest-markdown-docs`) to
    `[dev]` extras. Configure to run on `docs/**/*.md`.
    Verify it catches the original errors (commit a deliberate
    bad snippet, watch CI fail, revert).
13. Wire selected cookbook code blocks to pull from
    `examples/*.py` via mkdocs-material `--8<--` syntax. The
    canonical example becomes the only source of truth.

### Phase 4 — Verification (1 commit)

14. Full local gate: `ruff`, `ruff format`, `mypy --strict`,
    `pytest --cov` including the new docs-codeblock tests,
    `mkdocs build --strict`, `uv build`. Confirm everything
    Codex flagged is now resolved by re-running an audit-style
    check.
15. Update `CHANGELOG.md` `[Unreleased]` with the doc-fix
    summary; bump to `1.0.1` if user wants a patch release
    (recommended — the breaking-trust nature of the previous
    1.0.0 docs warrants a fix release).

## Effort estimate

| Phase | Files touched | Effort |
|---|---|---|
| Decisions (D1–D4) | — | 10 min user decisions |
| Phase 1 (source) | 2 files | 30 min |
| Phase 2 (docs surgery) | ~10 files | 2-3 hours |
| Phase 3 (CI gating) | pyproject.toml + ci.yml + 1 mkdocs change | 1 hour |
| Phase 4 (verification + release) | progress.md, CHANGELOG.md, version bump | 30 min |

Total: ~4-5 hours focused work after the 4 decisions land.

## Why not just dive in

I already dove in once. The Codex audit caught ~12 issues
because the dive was from memory, not from source. The plan
above:

1. Audits source per-issue before any edit (no inventing
   APIs).
2. Surfaces real choices (D1–D4) instead of papering over
   them.
3. Wires up CI gating so the next docs change can't drift the
   same way.
4. Sequences fixes so cross-page consistency holds (e.g., D1
   determines whether quickstart/index/memory/plan share
   import lines).

The user asked for "a very detailed plan" — this is it.
Approval on the four decisions unblocks the work.

## Open questions for the user

1. **D1: top-level exports — A or B?**
2. **D2: `SECRET_PACK` posture — A (change docs) or B (change source)?**
3. **D3: `Session` constructor — A or B?** (Strong recommend A.)
4. **D4: CI gating — A, B, or C?** (Strong recommend C.)
5. **Patch-release shape** — ship as `1.0.1` (docs-fix patch) or
   roll into next `1.x` minor? My instinct: `1.0.1`. The wrong
   docs landed in the same release as `1.0.0`; the fix
   deserves the same release cadence so anyone reading the
   1.0.0 docs sees the 1.0.1 note within a day.
