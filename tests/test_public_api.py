"""Doc/source parity for the top-level `harness` namespace.

Two invariants:

1. Every name documented as `from harness import X` (across `docs/` and
   `README.md`) resolves on the `harness` module — either eagerly or
   through the lazy `__getattr__` fallback. Vendor-extra names
   (`AnthropicRunner`, `OpenAICompatRunner`, `OpenTelemetrySink`) are
   permitted to raise `ImportError` when the matching extra isn't
   installed; that's the contract, not a regression.

2. Every name in `harness.__all__` resolves the same way. This is the
   forward-going guard — when a maintainer adds a name to `__all__`,
   they must also wire the import.

The parser only handles `from harness import …` (single-line and
parenthesized multi-line). It deliberately doesn't try to parse
arbitrary Python — keeping the regex tight is the point.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import harness

REPO_ROOT = Path(__file__).resolve().parent.parent

DOC_PATHS = [
    REPO_ROOT / "docs" / "index.md",
    REPO_ROOT / "docs" / "quickstart.md",
    REPO_ROOT / "README.md",
    *sorted((REPO_ROOT / "docs" / "modules").glob("*.md")),
    *sorted((REPO_ROOT / "docs" / "cookbook").glob("*.md")),
]

# Names whose import path requires an optional extra; missing extras
# surface as `ImportError` at the lazy `__getattr__` boundary.
OPTIONAL_EXTRA_NAMES: frozenset[str] = frozenset(
    {
        "AnthropicRunner",
        "OpenAICompatRunner",
        "OpenTelemetrySink",
    }
)

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SIMPLE = re.compile(r"^\s*from\s+harness\s+import\s+(.+?)\s*$", re.MULTILINE)
_MULTI_OPEN = re.compile(r"from\s+harness\s+import\s*\(")


def _extract_names(text: str) -> set[str]:
    """Return identifiers imported via `from harness import …` in `text`.

    Handles both single-line `from harness import A, B` and the
    parenthesized multi-line variant. Ignores aliasing keywords.
    """
    names: set[str] = set()
    for match in _SIMPLE.finditer(text):
        rhs = match.group(1)
        if rhs.startswith("("):
            # Parenthesized form starts here; the multi-line scan below
            # will pick it up. Skip to avoid double counting the open paren.
            continue
        for token in _NAME.findall(rhs):
            if token != "as":
                names.add(token)
    for match in _MULTI_OPEN.finditer(text):
        index = match.end()
        depth = 1
        while index < len(text) and depth > 0:
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        block = text[match.end() : index - 1]
        for token in _NAME.findall(block):
            if token != "as":
                names.add(token)
    return names


def _collect_documented_names() -> set[str]:
    """Scan every doc path and union the `from harness import …` symbols."""
    documented: set[str] = set()
    for path in DOC_PATHS:
        if not path.exists():
            continue
        documented |= _extract_names(path.read_text(encoding="utf-8"))
    return documented


def _resolve(name: str) -> object:
    """Return `getattr(harness, name)` with optional-extra tolerance.

    For vendor-extra names, an `ImportError` is the documented contract
    (the lazy `__getattr__` re-raises the underlying SDK import failure
    with an actionable install hint). Surfacing it here as a `pytest.skip`
    keeps the suite passing on base installs while still flagging the
    name as part of the public surface.
    """
    try:
        return getattr(harness, name)
    except ImportError as exc:
        if name in OPTIONAL_EXTRA_NAMES:
            pytest.skip(f"{name} requires an optional extra: {exc}")
        raise


DOCUMENTED_NAMES = sorted(_collect_documented_names())
ALL_NAMES = sorted(harness.__all__)


@pytest.mark.parametrize("name", DOCUMENTED_NAMES)
def test_documented_name_is_resolvable(name: str) -> None:
    """Every `from harness import X` in the docs must resolve on `harness`."""
    assert _resolve(name) is not None


@pytest.mark.parametrize("name", ALL_NAMES)
def test_all_name_is_resolvable(name: str) -> None:
    """Every name in `harness.__all__` must resolve via `getattr`."""
    assert _resolve(name) is not None


def test_documented_names_are_in_all() -> None:
    """Forward-going guard: no doc drift sneaking in a new name.

    Any name a user can `from harness import` per the docs must also be
    declared in `__all__`. Catches the inverse direction of the gap that
    M2.3 fixed.
    """
    missing = sorted(set(DOCUMENTED_NAMES) - set(harness.__all__))
    assert not missing, (
        f"Names documented as `from harness import …` but absent from `harness.__all__`: {missing}"
    )


def test_documented_names_collection_is_nonempty() -> None:
    """Sanity: the doc scan actually found something.

    A silent zero-match (e.g. someone moved the docs directory) would
    make the parametrized tests above pass vacuously.
    """
    assert DOCUMENTED_NAMES, "No `from harness import …` names extracted from docs"
