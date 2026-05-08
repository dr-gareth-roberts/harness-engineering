"""Top-level `harness` CLI dispatcher.

A minimal argparse front-end. Each feature module registers its own
subcommand by exporting a `register(subparsers)` function — this module
imports the registrar lazily so missing optional dependencies don't break
`harness --help`.

Add a new subcommand by creating `src/harness/<feature>/cli.py` with a
`register(subparsers)` function and adding a one-line lazy import below.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="harness-engineering CLI",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Each subcommand registration is wrapped in a try/except so that an
    # ImportError in one feature (e.g. missing optional dependency) does
    # not break the rest of the CLI.
    try:
        from harness.debug.cli import register as register_debug

        register_debug(subparsers)
    except ImportError:  # pragma: no cover - only fires if module missing
        pass

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0

    result = func(args)
    return int(result) if result is not None else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
