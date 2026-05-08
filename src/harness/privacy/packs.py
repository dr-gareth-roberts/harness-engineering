"""Pre-built detector packs.

Curated, named regexes for common secret/PII shapes. These are heuristic
*starters*: covering them is better than nothing, but a real deployment
will want to extend them based on its own threat model and false-positive
budget.

Each detector carries a sane default `action`:

- `SECRET_PACK` defaults to `block` — secrets crossing the prompt boundary
  is almost always a bug.
- `PII_PACK` defaults to `redact` — PII is more often a content issue than
  a security issue; redaction keeps the model usable.
- `HIPAA_PACK` ships a small starter set; expand for real PHI workloads.
"""

from __future__ import annotations

from harness.privacy.detectors import RegexDetector

# ---------------------------------------------------------------------------
# Secrets

SECRET_PACK: list[RegexDetector] = [
    # AWS access key IDs are uppercase 20-char tokens prefixed with AKIA.
    RegexDetector(
        "aws_access_key",
        r"\bAKIA[A-Z0-9]{16}\b",
        direction="both",
        action="block",
    ),
    # Anthropic API keys: `sk-ant-` then >=40 base64-ish chars.
    RegexDetector(
        "anthropic_api_key",
        r"\bsk-ant-[A-Za-z0-9_-]{40,}\b",
        direction="both",
        action="block",
    ),
    # GitHub fine-grained / classic / OAuth / user-server tokens.
    RegexDetector(
        "github_token",
        r"\bgh[opsu]_[A-Za-z0-9]{36,}\b",
        direction="both",
        action="block",
    ),
    # Stripe live/test keys (secret, publishable, restricted).
    RegexDetector(
        "stripe_key",
        r"\b(?:sk|pk|rk)_(?:test|live)_[A-Za-z0-9]{24,}\b",
        direction="both",
        action="block",
    ),
]


# ---------------------------------------------------------------------------
# PII (US-centric starters; extend per locale)

PII_PACK: list[RegexDetector] = [
    RegexDetector(
        "us_ssn",
        r"\b\d{3}-\d{2}-\d{4}\b",
        direction="outbound",
        action="redact",
    ),
    RegexDetector(
        "us_phone",
        r"\b\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        direction="outbound",
        action="redact",
    ),
    RegexDetector(
        "email",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        direction="outbound",
        action="redact",
    ),
]


# ---------------------------------------------------------------------------
# HIPAA-flavoured PHI starters
#
# Realistic PHI scrubbing is a much larger project (Presidio etc.). What ships
# here is a *starter set* matching common identifier shapes — extend per
# institution.

HIPAA_PACK: list[RegexDetector] = [
    # Medical-record-number style identifiers, common shapes.
    RegexDetector(
        "us_mrn",
        r"\bMRN[:#\s-]?\d{6,10}\b",
        direction="outbound",
        action="redact",
    ),
    # National Provider Identifier (10 digits).
    RegexDetector(
        "us_npi",
        r"\bNPI[:#\s-]?\d{10}\b",
        direction="outbound",
        action="redact",
    ),
    # ICD-10 diagnostic codes — flagged for audit only by default; legitimate
    # clinical text will contain them, so blocking is too aggressive.
    RegexDetector(
        "icd10_code",
        r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b",
        direction="outbound",
        action="audit",
    ),
]
