# `harness.privacy`

`PrivacyBoundary(detectors).wrap(real_runner)` returns a runner that
scans every text fragment, tool argument, and tool result content
recursively for secrets / PII. `RegexDetector` + `EntropyDetector`
ship with pre-built `SECRET_PACK`, `PII_PACK`, and `HIPAA_PACK`.
`PresidioDetector` (Wave 13b, under `[privacy-ml]`) adds
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

```bash
uv add 'harness-engineering[privacy-ml]'
```

```python
from harness.privacy import build_pii_pack
boundary = PrivacyBoundary(detectors=[*SECRET_PACK, *build_pii_pack()])
```

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
