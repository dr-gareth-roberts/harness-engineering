from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from harness.hooks.events import HookDecision, PreToolUse


class PathDenied(ValueError):
    """Raised by PathScope.validate when a path is outside the allowed scope."""


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class PathScope:
    """A pair of allow / deny path-prefix lists.

    Symlink-resolved: both the argument and the configured prefixes are
    passed through `Path.resolve(strict=False)` before containment checks,
    so `../escape` and symlinks pointing outside the allow set are caught.

    **Advisory, not enforced.** Between `is_allowed()` returning True and a
    caller actually opening the path, a concurrent symlink swap (TOCTOU)
    can redirect the operation. Use OS-level isolation (sandbox, namespaces,
    seccomp) if real safety matters.

    Empty `allow_prefixes` means **everything is allowed**, subject to
    `deny_prefixes`. To deny by default, configure an explicit
    `deny_prefixes=("/",)` or pass an allow-list of an unreachable root.
    """

    allow_prefixes: tuple[Path, ...] = field(default_factory=tuple)
    deny_prefixes: tuple[Path, ...] = field(default_factory=tuple)

    @classmethod
    def of(
        cls,
        *,
        allow: Iterable[str | Path] = (),
        deny: Iterable[str | Path] = (),
    ) -> PathScope:
        return cls(
            allow_prefixes=tuple(Path(p).resolve(strict=False) for p in allow),
            deny_prefixes=tuple(Path(p).resolve(strict=False) for p in deny),
        )

    def _resolve(self, path: str | Path) -> Path | None:
        try:
            return Path(path).resolve(strict=False)
        except (OSError, RuntimeError):
            return None

    def is_allowed(self, path: str | Path) -> bool:
        resolved = self._resolve(path)
        if resolved is None:
            return False
        if any(_is_within(resolved, d) for d in self.deny_prefixes):
            return False
        if not self.allow_prefixes:
            return True
        return any(_is_within(resolved, a) for a in self.allow_prefixes)

    def validate(self, path: str | Path) -> Path:
        resolved = self._resolve(path)
        if resolved is None or not self.is_allowed(resolved):
            raise PathDenied(f"path {path!r} is outside the allowed scope")
        return resolved


@dataclass(frozen=True)
class PathPolicy:
    """A `harness.policy.Policy` that blocks `PreToolUse` events whose
    path-shaped arguments fall outside `scope`.

    The policy only inspects calls to tools listed in `tool_names`; calls
    to other tools pass through. For each listed tool, `arg_keys` is the
    list of argument names that carry path values — the default checks a
    single `path` key. Missing keys are skipped (no error).
    """

    scope: PathScope
    tool_names: frozenset[str]
    arg_keys: tuple[str, ...] = ("path",)

    @classmethod
    def of(
        cls,
        scope: PathScope,
        tool_names: Iterable[str],
        *,
        arg_keys: Iterable[str] = ("path",),
    ) -> PathPolicy:
        return cls(scope=scope, tool_names=frozenset(tool_names), arg_keys=tuple(arg_keys))

    def __call__(self, event: PreToolUse) -> HookDecision | None:
        if event.call.name not in self.tool_names:
            return None
        for key in self.arg_keys:
            value = event.call.arguments.get(key)
            if value is None:
                continue
            if not self.scope.is_allowed(value):
                return HookDecision(
                    block=True,
                    reason=(
                        f"path {value!r} on tool {event.call.name!r} arg {key!r} "
                        "is outside the allowed scope"
                    ),
                )
        return None
