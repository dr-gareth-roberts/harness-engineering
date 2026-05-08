# `harness.privacy`

`PrivacyBoundary(detectors).wrap(real_runner)` returns a runner that
scans every text fragment, tool argument, and tool result content
recursively for secrets / PII. `RegexDetector` + `EntropyDetector`
ship with pre-built `SECRET_PACK`, `PII_PACK`, and `HIPAA_PACK`.
Per-detector `direction` (outbound/inbound/both) and `action`
(redact/block/audit). Audit events never carry the matched value.

::: harness.privacy
