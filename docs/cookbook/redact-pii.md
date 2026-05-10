# Redact PII before sending to a model

## Problem

Your agent reads user-submitted text that might contain emails, SSNs,
API keys, credit cards. You don't want any of that to land in your
model provider's logs (or worse, the model's context).

## Solution sketch

Wrap your runner with a `PrivacyBoundary`. The boundary scans every
text fragment, tool argument, and tool result *recursively* on its
way out (to the model) and on its way back. Each detection is either
**redacted** in place, **blocked** with a typed exception, or
**audited** silently to a sink — your choice per-detector.

Two detector flavors ship out of the box:

- `RegexDetector` for known shapes (SSN, AWS keys, GitHub tokens, etc.).
- `EntropyDetector` for high-entropy strings that "look like" secrets
  but don't match a known pattern.

For broader NLP-backed PII (people's names, international phone
numbers, addresses), the `[privacy-ml]` extra adds `PresidioDetector`.

## Working code

```python
import asyncio

from pydantic import BaseModel

from harness import (
    AnthropicRunner,
    Dispatcher,
    HookRunner,
    Orchestrator,
    PII_PACK,
    PrivacyBoundary,
    SECRET_PACK,
    SubAgent,
    Tool,
    text,
)


class LookupIn(BaseModel):
    query: str


def lookup(args: LookupIn) -> str:
    return "(no result)"


dispatcher = Dispatcher(
    [Tool(name="lookup", description="Search records.", input_model=LookupIn, handler=lookup)]
)
hooks = HookRunner()

# 1. Build the inner runner (the one that actually talks to the model).
inner = AnthropicRunner(dispatcher, hooks)

# 2. Wrap it with a PrivacyBoundary configured with the pre-built
#    detector packs. Both packs default to outbound-only redact —
#    the cleanest posture for "don't leak user data to the provider."
boundary = PrivacyBoundary(detectors=[*SECRET_PACK, *PII_PACK])
runner = boundary.wrap(inner)

# 3. Drive normally. The orchestrator sees the wrapped runner; the
#    inner AnthropicRunner only sees redacted text.
orchestrator = Orchestrator(dispatcher, hooks, runner)
agent = SubAgent(name="bot", system_prompt="", model="claude-opus-4-7", allowed_tools=["lookup"])

asyncio.run(
    orchestrator.run(
        agent,
        [text("user", "My email is alice@example.com and SSN 123-45-6789.")],
    )
)
# The model never sees the email or SSN — it sees `[REDACTED]` markers.
```

## Switch to NLP-backed detection

```bash
uv add 'harness-engineering[privacy-ml]'
```

```python
from harness.privacy import build_pii_pack

# Replace the regex PII pack with Presidio's NLP recognizers.
boundary = PrivacyBoundary(detectors=[*SECRET_PACK, *build_pii_pack()])
```

`build_pii_pack()` returns an outbound-only redact pack covering
PERSON, EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, US_DRIVER_LICENSE,
US_PASSPORT, CREDIT_CARD, IBAN_CODE, DATE_TIME, LOCATION,
IP_ADDRESS. Costs ~50ms per scan after the spaCy model warms up.

## Switch to "block on detection"

If you want the boundary to *fail* rather than silently redact:

```python
from harness.privacy import RegexDetector

api_key = RegexDetector(
    name="aws_key",
    pattern=r"AKIA[0-9A-Z]{16}",
    action="block",  # raises PrivacyViolation instead of redacting
)
boundary = PrivacyBoundary(detectors=[api_key])
```

`PrivacyViolation` is a typed exception you can catch upstream.

## Audit-only mode (don't redact, just record)

For pre-production / observability:

```python
from harness.telemetry import JSONLSink

audit = JSONLSink("./privacy.jsonl")
boundary = PrivacyBoundary(
    detectors=[RegexDetector("ssn", r"\d{3}-\d{2}-\d{4}", action="audit")],
    audit_sink=audit.emit,
)
```

Audit events never carry the matched value — only the detector name,
location, and the action that fired. The text passes through unchanged.

## Gotchas

- **Boundary scans recursively** — text content, tool arguments, tool
  result content. If you have a deep nested dict, every leaf string
  gets scanned. Depth is capped to prevent runaway, but a tool that
  returns a 10MB JSON blob will still cost time.
- **Redaction is destructive** — once a text fragment has been
  scrubbed, the original is not in the model's context. If you need
  the original later (e.g., to insert into a database), thread it
  through your handler directly, not through the model.
- **Direction matters**. Pre-built packs are outbound-only by default
  ("don't leak to provider"). For inbound scanning ("don't trust
  whatever the model returns"), set `direction="inbound"` or
  `"both"`.
- **Presidio cold-start** is ~2s on first scan (loads spaCy model).
  Pre-construct the `PresidioDetector` at startup if latency on
  first request matters.

## Related

- [`harness.privacy`](../modules/privacy.md) — module reference.
- [`examples/privacy.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/privacy.py) — runnable end-to-end demo.
- [Cookbook: Observability](observability.md) — pipe audit events to OTel.
