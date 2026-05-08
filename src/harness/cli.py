"""harness — top-level CLI dispatcher.

Each feature module that wants a CLI surface ships a `cli.py` exposing
a `register(subparsers)` callable that registers its own argparse
subparser and `set_defaults(func=...)` to a callable taking the parsed
`Namespace`. The dispatcher loads each known subcommand module lazily
via `importlib.util.find_spec`, so a feature whose module hasn't yet
landed simply doesn't appear in `--help`.

Subcommand contract:
    `<feature>.cli.register(subparsers: argparse._SubParsersAction) -> None`
        - Adds one subparser via `subparsers.add_parser(name, help=...)`.
        - Calls `parser.set_defaults(func=<callable taking argparse.Namespace
          and returning int>)` on it.

Example: `src/harness/cache/cli.py` adds the `cache-audit` subcommand;
`src/harness/debug/cli.py` adds the `debug` subcommand.

Run with `uv run harness <command> [args]`. The console script is
declared under `[project.scripts]` in `pyproject.toml`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys

# Modules to probe for a `register(subparsers)` entry point. Order matters
# only for `--help` output — subcommand names are independent.
_SUBCOMMAND_MODULES: tuple[str, ...] = (
    "harness.cache.cli",
    "harness.debug.cli",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="harness-engineering toolkit CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    for module_name in _SUBCOMMAND_MODULES:
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            # The parent package itself isn't present (e.g. harness.cache
            # before #3 lands). Treat as "subcommand not installed."
            spec = None
        if spec is None:
            continue
        # `import_module` will surface real import failures (syntax errors,
        # import-time exceptions inside the module) — only the "doesn't
        # exist" branch is silently skipped above.
        module = importlib.import_module(module_name)
        register = getattr(module, "register", None)
        if register is None:
            raise RuntimeError(
                f"{module_name} loaded but exposes no `register(subparsers)`. "
                "Each subcommand module must define a `register` callable."
            )
        register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    result = func(args)
    return int(result) if result is not None else 0


if __name__ == "__main__":  # pragma: no cover - convenience shim
    sys.exit(main())
