from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
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
    result = await safe_subprocess_run([sys.executable, "-c", "import sys; sys.exit(7)"])
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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="start_new_session / killpg are POSIX-only; see process.py docstring.",
)
async def test_timeout_kills_double_forked_grandchildren(tmp_path: Path) -> None:
    """A misbehaving program that backgrounds a sleeper must not survive timeout.

    The shell launches a long sleeper in the background (`&`), records its
    PID, then sleeps itself. On timeout the direct child gets killed, but
    without `start_new_session=True` + `killpg`, the backgrounded sleeper
    inherits PID 1 / the test process and lingers. This test asserts that
    after the timeout the grandchild PID is gone.
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on this platform")

    pidfile = tmp_path / "grandchild.pid"
    # Background a long sleeper, write its PID, then block. The direct child
    # (bash) will be killed on timeout; without process-group cleanup the
    # backgrounded `sleep 30` survives.
    script = f"sleep 30 & echo $! > {pidfile}; sleep 30"

    with pytest.raises(SubprocessTimeout):
        await safe_subprocess_run([bash, "-c", script], timeout=0.5)

    # Give the kernel a moment to deliver SIGKILL across the group.
    await asyncio.sleep(0.3)

    assert pidfile.exists(), "shell did not record the grandchild PID"
    grandchild_pid = int(pidfile.read_text().strip())

    try:
        # `kill 0` probes existence without delivering a signal.
        # ProcessLookupError ⇒ dead (what we want).
        os.kill(grandchild_pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False

    if alive:
        # The fix regressed — clean up the orphan we just leaked so this test
        # process tree doesn't carry it for ~30s, then fail loudly.
        with contextlib.suppress(ProcessLookupError):
            os.kill(grandchild_pid, signal.SIGKILL)
        pytest.fail(
            f"grandchild PID {grandchild_pid} survived timeout — "
            "start_new_session/killpg cleanup did not reach the process group."
        )
