from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.hooks.events import PreToolUse
from harness.sandbox import PathDenied, PathPolicy, PathScope
from harness.tools import ToolCall


def test_allow_list_passes_path_inside_prefix(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    inside = tmp_path / "a" / "b.txt"
    assert scope.is_allowed(inside) is True


def test_allow_list_rejects_path_outside(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    assert scope.is_allowed("/etc/passwd") is False


def test_deny_list_overrides_allow(tmp_path: Path) -> None:
    secret = tmp_path / "secret"
    scope = PathScope.of(allow=[tmp_path], deny=[secret])
    assert scope.is_allowed(secret / "file") is False
    assert scope.is_allowed(tmp_path / "ok.txt") is True


def test_empty_allow_list_means_unrestricted(tmp_path: Path) -> None:
    scope = PathScope.of(deny=[tmp_path / "secret"])
    assert scope.is_allowed("/anything/anywhere") is True
    assert scope.is_allowed(tmp_path / "secret" / "x") is False


def test_dotdot_traversal_is_rejected(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    escape = tmp_path / ".." / ".." / "etc" / "passwd"
    assert scope.is_allowed(escape) is False


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    inside = tmp_path / "allowed"
    inside.mkdir()
    link = inside / "escape"
    os.symlink(target, link)
    scope = PathScope.of(allow=[inside])

    # Direct symlink → resolves outside `inside` → rejected.
    assert scope.is_allowed(link) is False
    # Path through the symlink → also rejected.
    assert scope.is_allowed(link / "file.txt") is False


def test_validate_returns_resolved_path(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    out = scope.validate(tmp_path / "a" / "b.txt")
    assert out.is_absolute()


def test_validate_raises_path_denied(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    with pytest.raises(PathDenied):
        scope.validate("/etc/passwd")


# ---------------------------------------------------------------------------
# PathPolicy


def _event(name: str, **arguments: object) -> PreToolUse:
    return PreToolUse(call=ToolCall(name=name, arguments=dict(arguments)))


def test_policy_blocks_out_of_scope_path(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    policy = PathPolicy.of(scope, ["read_file"])

    decision = policy(_event("read_file", path="/etc/passwd"))
    assert decision is not None
    assert decision.block is True
    assert "outside the allowed scope" in (decision.reason or "")


def test_policy_passes_in_scope_path(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    policy = PathPolicy.of(scope, ["read_file"])
    assert policy(_event("read_file", path=str(tmp_path / "ok"))) is None


def test_policy_ignores_unlisted_tool(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    policy = PathPolicy.of(scope, ["read_file"])
    assert policy(_event("write_file", path="/etc/passwd")) is None


def test_policy_skips_missing_arg_keys(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    policy = PathPolicy.of(scope, ["read_file"], arg_keys=("source", "destination"))
    # Neither key present → no decision.
    assert policy(_event("read_file", other="x")) is None


def test_policy_checks_multiple_arg_keys(tmp_path: Path) -> None:
    scope = PathScope.of(allow=[tmp_path])
    policy = PathPolicy.of(scope, ["copy"], arg_keys=("source", "destination"))

    # source ok, destination outside → blocked.
    decision = policy(_event("copy", source=str(tmp_path / "a"), destination="/etc/passwd"))
    assert decision is not None
    assert decision.block is True
    assert "destination" in (decision.reason or "")
