"""Smoke-test every example file under `examples/`.

Each example exposes an `async def main() -> int`. We import the module,
call `main()`, and assert it returned 0. Stdout is captured but not
asserted on — the goal is "did this finish without raising," not "did
the output exactly match." If you want richer per-example checks, add
them to that example's own test file (e.g. `tests/contracts/test_*`)
rather than here.

Adding a new example: drop the file in `examples/`, then append a
`("module_name", "marker")` tuple to `EXAMPLES` below. The marker is a
short string that should appear in the example's printed output —
catches accidental "main runs but does nothing" regressions.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

# (module_name, expected_stdout_substring). The substring is informational —
# it's what the example should print if it actually ran. If you change the
# example, update the marker.
EXAMPLES: list[tuple[str, str]] = [
    # Module name (under examples/) → a short string that should appear in
    # stdout when the example runs successfully.
    ("end_to_end", "final assistant message"),
    ("contracts", "contract"),
    ("cache", "drift"),
    ("privacy", "REDACTED"),
    ("otel", "events on span"),
    ("debug", "breakpoint"),
]


def _load_example(name: str):
    """Import an examples/*.py module by its file name (no extension)."""
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    spec = importlib.util.spec_from_file_location(f"_example_{name}", examples_dir / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("name", "marker"), EXAMPLES)
def test_example_main_runs_to_completion(
    name: str,
    marker: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_example(name)
    assert hasattr(module, "main"), f"examples/{name}.py must define `async def main()`"
    rc = asyncio.run(module.main())
    assert rc == 0, f"examples/{name}.py main() returned {rc}, expected 0"

    captured = capsys.readouterr()
    assert marker in captured.out, (
        f"examples/{name}.py output did not contain expected marker {marker!r}; "
        f"got: {captured.out[:300]!r}"
    )
