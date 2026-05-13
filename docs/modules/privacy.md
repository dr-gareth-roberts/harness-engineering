# `harness.privacy`

`PrivacyBoundary(detectors).wrap(real_runner)` returns a runner that
scans every text fragment, tool argument, tool result content, and
image/file metadata for secrets / PII. `RegexDetector` +
`EntropyDetector` ship with pre-built `SECRET_PACK`, `PII_PACK`, and
`HIPAA_PACK`. `PresidioDetector` (Wave 13b, under `[privacy-ml]`) adds
NLP-backed PII detection. Per-detector `direction`
(outbound/inbound/both) and `action` (redact/block/audit). Audit
events never carry the matched value.

## When to reach for this

- You read user-submitted text and don't want it leaking to your
  model provider's logs.
- You want a uniform redaction layer across vendor runners — wrap
  the boundary once, switch runners freely.
- You want to detect PII the regex pack misses (names, addresses,
  international phone numbers) — enable Presidio.

## Quick example

<!-- reason: illustrative; AnthropicRunner needs the [anthropic] extra and references undefined dispatcher / hooks -->
<!--pytest.mark.skip-->
```python
from harness import (
    AnthropicRunner, PrivacyBoundary, PII_PACK, SECRET_PACK,
)

inner = AnthropicRunner(dispatcher, hooks)
boundary = PrivacyBoundary(detectors=[*SECRET_PACK, *PII_PACK])
runner = boundary.wrap(inner)

# Use `runner` like any other Runner. The model never sees raw PII.
```

Switch to NLP-backed:

<!-- reason: shell example, not executed in the codeblock gate -->
<!--pytest.mark.skip-->
```bash
uv add 'harness-engineering-toolkit[privacy-ml]'
```

<!-- reason: illustrative; build_pii_pack() needs the [privacy-ml] extra and references undefined SECRET_PACK -->
<!--pytest.mark.skip-->
```python
from harness.privacy import build_pii_pack
boundary = PrivacyBoundary(detectors=[*SECRET_PACK, *build_pii_pack()])
```

## What the boundary scans (and what it doesn't)

The boundary is a **string-level** detector pipeline. It scans every
string it can find on a message; it does not decode image bytes or
read file bodies.

**Scanned by default:**

- `text` block content (the `text` field).
- `tool_use.arguments` — walked recursively to find string leaves,
  capped at four levels deep (deeper subtrees are stringified and
  flat-scanned).
- `tool_result.content` — same recursive walk.
- **Image block metadata**: the URL (when `image.source == "url"`)
  and the `media_type` field.
- **File block metadata**: the `file_id` and `path` fields.
- Dict keys used in audit-event `location` paths are sanitized
  separately so secret-shaped keys don't leak into the audit trail.

**Not scanned:**

- **Base64-encoded image content.** Reading text out of an inline
  image requires OCR, which is out of scope. The base64 bytes pass
  through the boundary untouched.
- **File body content.** When a `file_id` references a vendor Files
  API document, the body is fetched by the runner *after* the
  boundary and is never visible to the detector pipeline.

If you need image-text or file-body scanning, run a pre-pass over
the source material yourself and materialize the extracted text into
a `text` block before calling the runner. A common pattern:

<!-- reason: illustrative; references non-existent receipt.png and an undefined run_ocr() helper -->
<!--pytest.mark.skip-->
```python
from harness.prompts import attach_image
# Construct the image block for the model to see…
img_block = attach_image(path="receipt.png")
# …and a sibling text block carrying OCR output for the boundary to scan.
ocr_text = run_ocr("receipt.png")  # your OCR of choice
messages = [
    Message(role="user", content=[img_block, ContentBlock(type="text", text=ocr_text)]),
]
```

The boundary then redacts/blocks/audits on the OCR text exactly as it
does for any other text fragment.

## Gotchas

- **The boundary scans recursively** — text content, tool arguments,
  tool result content. A handler returning a 10MB JSON blob will
  get every leaf string scanned. Depth is capped to prevent
  runaway, but cost is real.
- **Redaction is destructive.** Once scrubbed, the original isn't
  in the model's context. If you need the original later, thread
  it through your handler directly, not through the model.
- **Direction matters.** Pre-built packs default to outbound-only
  (don't leak to provider). For inbound (don't trust the model's
  return), set `direction="inbound"` or `"both"`.
- **Presidio cold-start** is ~2s on first scan (loads spaCy model).
  Pre-construct the detector at startup if first-request latency
  matters.

## Related

- [Cookbook: Redact PII](../cookbook/redact-pii.md) — extended walkthrough.
- [`examples/privacy.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/privacy.py) — runnable demo.
- [`harness.telemetry`](telemetry.md) — pipe audit events to OTel / JSONL.

## API reference

::: harness.privacy
