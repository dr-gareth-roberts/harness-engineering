## Wave 12 — Modality + Files API

### Goal
Bring multimodal input — vision (images via base64 or URL) and the
Anthropic Files API — into the harness primitives. Streaming output
(#9) was originally in this wave's scope; deferred per advisor review
to its own dedicated wave (Wave 13a) because it's the heaviest single
item in the 28-gap plan and a tail-of-wave squeeze risks regressions
in the 510 existing tests around `AnthropicRunner.__call__`.

### Status
Shipped on `feature/wave-12-modality-streaming`. Two gaps cleared
(#7, #8); #9 deferred to Wave 13a.

### What landed

| # | Item | Implementation |
| --- | --- | --- |
| 7 | Vision content blocks | New `harness.prompts.ImageRef` Pydantic model with `source: Literal["base64", "url"]`, `media_type`, `data`. `ContentBlock` gains `image: ImageRef \| None` and the literal `"image"` to its `BlockType`. `harness.prompts.attach_image(path=... \| url=..., media_type=...)` builds them — base64-encoding from disk and auto-inferring `media_type` from the extension when path-based; URL mode requires explicit `media_type`. Both runners translate: `AnthropicRunner` to `{"type":"image","source":{"type":"base64"\|"url","media_type":...,"data":...}}`; `OpenAICompatRunner` to `{"type":"image_url","image_url":{"url":...}}` parts (data URLs for inline base64). User messages with images become list-shaped `content` arrays so text + image mixing works in both vendor formats. Pre-Wave-12 text-only user messages stay string-shaped (back-compat). |
| 8 | Anthropic Files API integration | `ContentBlock` gains `file_id: str \| None`. `attach_file(file_id="file_...")` builds a file block referencing an Anthropic Files API document by id. `AnthropicRunner` translates to `{"type":"document","source":{"type":"file","file_id":...}}`; the path-based mode keeps the historical text-inlining behavior. `OpenAICompatRunner` surfaces `file_id` as a `<file file_id=...>` text placeholder (no native equivalent). The upload helper (`upload_file(client, path) → file_id`) is **deferred** — needs API keys for an end-to-end smoke; users can call `client.beta.files.upload(...)` directly today. |
| 9 | Streaming output | **Deferred to Wave 13a.** Per advisor review: streaming touches the `AnthropicRunner.__call__` hot path (tool-use loop, speculator begin/end, hook ordering, cache cap, timeout, replacement honoring) — a tail-of-wave attempt risks regressions across 500+ tests. The dedicated wave will add new event types (TextDelta / ToolUseStart / ToolUseEnd / MessageEnd), a `StreamingRunner` Protocol, `Orchestrator.run_stream()`, and a CLI `--stream` mode with explicit speculator-during-stream tests. |

### Tests added

| File | Count | Coverage |
| --- | --- | --- |
| `tests/prompts/test_image_and_files.py` | 16 | `attach_image` URL-mode + path-mode (with media_type inference, unrecognized-extension error, mutual exclusion); `attach_file(file_id=...)` block shape + mutual exclusion vs path; Anthropic translation for both image source modes + file_id document blocks + path-based fallback; OpenAI-compat data-URL conversion + URL passthrough + text-only string-shape preservation + file_id text fallback. |

16 new tests, **537 total** (was 521; +1 stress test surfaced earlier).
Coverage stays at **89%** with the 85% threshold.

### Verification gate

```
ruff check                       — clean
ruff format --check             — 174 files clean
mypy --strict src tests         — clean (159 source files)
pytest --cov=harness            — 537 passed, 1 skipped, 89% coverage
mkdocs build --strict           — clean
uv build                         — wheel + sdist build cleanly
```

### Deferred from this wave

- **Streaming output (#9)** — moved to Wave 13a. Estimated ~2 days
  in its own wave: new event types, `StreamingRunner` Protocol,
  `Orchestrator.run_stream()`, CLI `--stream` mode, speculator-
  during-stream targeted test.
- **Files API upload helper (`upload_file`)** — needs real API keys
  for end-to-end smoke. Users can call `client.beta.files.upload`
  directly via the SDK today; the harness side handles the resulting
  `file_id` correctly.

### Commits

```
*  chore(progress): rotate Wave 11 to docs/waves/
*  feat(prompts): vision content blocks + Anthropic Files API attachments
*  feat(runner): translate image and file_id blocks in both vendor runners
*  docs: CHANGELOG + plan.md split + progress.md log of Wave 12
```
