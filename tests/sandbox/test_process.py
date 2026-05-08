from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from harness.sandbox import (
    DEFAULT_ALLOWED_ENV_KEYS,
    SubprocessTimeout,
    safe_subprocess_run,
    scrub_env,
)

# ---------------------------------------------------------------------------
# scrub_env


def test_scrub_env_keeps_only_allowed_keys() -> None:
    base = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "secret", "HOME": "/root"}
    out = scrub_env(base=base)
    assert out == {"PATH": "/usr/bin", "HOME": "/root"}


def test_scrub_env_extra_overrides_base() -> None:
    base = {"PATH": "/usr/bin", "HOME": "/root"}
    out = scrub_env(base=base, extra={"PATH": "/override", "X": "y"})
    assert out == {"PATH": "/override", "HOME": "/root", "X": "y"}


def test_scrub_env_does_not_mutate_base() -> None:
    base = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "secret"}
    snapshot = dict(base)
    scrub_env(base=base, extra={"X": "y"})
    assert base == snapshot


def test_scrub_env_default_set_excludes_secrets() -> None:
    # The default allow-list does not include any of the common secret-bearing
    # env-var names. This is the contract that protects child processes.
    suspect = {"ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "OPENAI_API_KEY"}
    assert suspect.isdisjoint(DEFAULT_ALLOWED_ENV_KEYS)


# ---------------------------------------------------------------------------
# safe_subprocess_run


async def test_runs_simple_command() -> None:
    result = await safe_subprocess_run([sys.executable, "-c", "print('hi')"])
    assert result.returncode == 0
    assert result.stdout == "hi\n"
    assert result.stderr == ""
    assert result.duration_ms >= 0


async def test_captures_nonzero_exit_without_raising() -> None:
    result = await safe_subprocess_run(
        [sys.executable, "-c", "import sys; sys.exit(7)"]
    )
    assert result.returncode == 7


async def test_timeout_raises_subprocess_timeout() -> None:
    start = time.perf_counter()
    with pytest.raises(SubprocessTimeout, match="exceeded timeout"):
        await safe_subprocess_run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.2,
        )
    elapsed = time.perf_counter() - start
    # Must return well before the 5s sleep would complete.
    assert elapsed < 3.0


async def test_default_env_strips_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    result = await safe_subprocess_run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('ANTHROPIC_API_KEY', '<unset>'))",
        ]
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "<unset>"


async def test_extra_env_passes_through() -> None:
    result = await safe_subprocess_run(
        [sys.executable, "-c", "import os; print(os.environ.get('X', '<unset>'))"],
        env=scrub_env(extra={"X": "y"}),
    )
    assert result.stdout.strip() == "y"


async def test_cwd_is_respected(tmp_path: Path) -> None:
    result = await safe_subprocess_run(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        cwd=tmp_path,
    )
    # On macOS /tmp is symlinked to /private/tmp — resolve both sides for compare.
    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


async def test_stdin_passes_through() -> None:
    result = await safe_subprocess_run(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"],
        stdin=b"hello\n",
    )
    assert result.stdout == "HELLO\n"


async def test_explicit_env_does_not_default_scrub() -> None:
    # Passing an explicit env replaces the default scrubbed view; if a caller
    # passes os.environ directly, secrets flow through (their choice).
    result = await safe_subprocess_run(
        [sys.executable, "-c", "import os; print(os.environ.get('X', '<unset>'))"],
        env={"PATH": os.environ.get("PATH", ""), "X": "yes"},
    )
    assert result.stdout.strip() == "yes"
