"""`harness cache-audit` — wire `audit()` into the top-level CLI.

The top-level dispatcher in `harness.cli` discovers this module
automatically via `importlib.util.find_spec` and calls our `register`
function. Each subcommand owns its own argparse surface; we don't share
state with the dispatcher beyond the parser tree.

Usage:

    $ harness cache-audit --store ./.harness/fingerprints --since 24h
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from harness.cache.audit import DriftReport, audit
from harness.cache.store import FileFingerprintStore

_DURATION_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[hdmw])$")
_UNIT_HOURS = {"h": 1.0, "d": 24.0, "w": 24.0 * 7, "m": 1.0 / 60}


def _parse_duration(spec: str) -> float:
    """Parse `24h` / `7d` / `30m` / `2w` into a float number of hours.

    Kept tiny on purpose: no `dateutil` dep. If you need seconds or
    months, lift this into `_internal/duration.py` and share with #6.
    """
    match = _DURATION_RE.match(spec.strip())
    if match is None:
        raise ValueError(
            f"unrecognised duration {spec!r}; expected forms like '24h', '7d', '30m', '2w'"
        )
    value = float(match.group("value"))
    unit = match.group("unit")
    return value * _UNIT_HOURS[unit]


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "cache-audit",
        help="Audit prompt cache fingerprints (#3 prefix-drift watcher).",
        description=(
            "Walk a FileFingerprintStore and report which cache breakpoints "
            "have stayed stable and which have drifted. Drift events include "
            "a unified diff of the prompt segments on either side of the change."
        ),
    )
    parser.add_argument(
        "--store",
        required=True,
        help="Path to a FileFingerprintStore — either a directory holding "
        "`fingerprints.jsonl` or the .jsonl file itself.",
    )
    parser.add_argument(
        "--since",
        default="24h",
        help="Audit window (default: 24h). Forms: '24h', '7d', '30m', '2w'.",
    )
    parser.set_defaults(func=_cmd_cache_audit)


def _print_report(report: DriftReport) -> None:
    print(f"audit-time: {report.audited_at.isoformat()}")
    if report.stable_prefixes:
        joined = ", ".join(str(idx) for idx in report.stable_prefixes)
        print(f"stable breakpoints: {joined}")
    else:
        print("stable breakpoints: <none>")

    if not report.drift_events:
        print("no drift events in window.")
        return

    print(f"drift events: {len(report.drift_events)}")
    for event in report.drift_events:
        print()
        print(
            f"DRIFT detected at breakpoint {event.breakpoint_index} on {event.after_ts.isoformat()}"
        )
        print(f"  before: {event.before_hash[:12]}…  after: {event.after_hash[:12]}…")
        if event.hint is not None:
            print(f"  Hint: {event.hint}")
        print("  Diff (first 20 changed lines):")
        for line in event.diff.splitlines():
            print(f"    {line}")


def _cmd_cache_audit(args: argparse.Namespace) -> int:
    window_hours = _parse_duration(args.since)
    store = FileFingerprintStore(Path(args.store))
    report = asyncio.run(audit(store, window_hours=window_hours))
    _print_report(report)
    return 0
