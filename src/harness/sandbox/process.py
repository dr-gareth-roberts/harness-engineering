from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ALLOWED_ENV_KEYS: frozenset[str] = frozenset(
    {"PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL"}
)


def scrub_env(
    base: Mapping[str, str] | None = None,
    *,
    allow_keys: Iterable[str] = DEFAULT_ALLOWED_ENV_KEYS,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Filter `base` (defaults to `os.environ`) down to `allow_keys`,
    then merge `extra` on top. `extra` overrides `base`.

    Useful for handing a tight environment to a child process: by default
    the output contains only `PATH`, `HOME`, `TMPDIR`, `TMP`, `TEMP`, `LANG`,
    and `LC_ALL` — so secrets in `os.environ` (`ANTHROPIC_API_KEY`,
    `AWS_*`, `GITHUB_TOKEN`, etc.) do not leak into a child by default.
    """
    src: Mapping[str, str] = base if base is not None else os.environ
    allow_set = set(allow_keys)
    result = {k: v for k, v in src.items() if k in allow_set}
    if extra is not None:
        result.update(extra)
    return result


class SubprocessTimeout(TimeoutError):
    """Raised by `safe_subprocess_run` when the child exceeds `timeout` seconds."""


@dataclass(frozen=True)
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float


async def safe_subprocess_run(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: float = 30.0,
    stdin: bytes | None = None,
    text_decode: str = "utf-8",
) -> SubprocessResult:
    """Run `cmd` directly (no shell), with a scrubbed env and a wall-clock timeout.

    `cmd` is a list of arguments — the underlying call uses
    `asyncio.create_subprocess_exec`, which spawns the program directly
    with no shell layer. Quoting games and command-injection in `cmd` are
    therefore not a risk. `env` defaults to `scrub_env()` — pass an
    explicit dict to override. On timeout the child is killed and
    `SubprocessTimeout` is raised; the kill+wait pattern is tested on
    POSIX (Linux/macOS); Windows event-loop variants are not exhaustively
    verified.
    """
    actual_env: Mapping[str, str] = env if env is not None else scrub_env()
    actual_cwd = str(cwd) if cwd is not None else None

    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=dict(actual_env),
        cwd=actual_cwd,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(input=stdin), timeout=timeout)
    except TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        raise SubprocessTimeout(f"subprocess {cmd[0]!r} exceeded timeout of {timeout}s") from exc

    duration_ms = (time.perf_counter() - start) * 1000.0
    return SubprocessResult(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout_b.decode(text_decode, errors="replace"),
        stderr=stderr_b.decode(text_decode, errors="replace"),
        duration_ms=duration_ms,
    )
