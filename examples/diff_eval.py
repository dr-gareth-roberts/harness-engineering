"""Differential cross-runner evaluation - find the dissenter.

Run with: `uv run python examples/diff_eval.py`

`harness.replay.diff_eval` runs the same `EvalCase` set against
multiple runners in parallel and surfaces which cases were unanimous
and which had outliers. Internally each runner walks its case list
sequentially; the runner-level coroutines are gathered, so wall time
is roughly slowest_runner * len(cases).

This example wires three named runners against three small cases.
- alpha and beta share an identical reply list.
- gamma diverges on exactly one case.

The resulting `DiffMatrix` should report two unanimous cases (where
all three agree) and one with a single outlier (gamma). The example
also writes the HTML report to a temp file and reports its size.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from harness.agents import SubAgent
from harness.replay import EvalCase
from harness.replay.diff_eval import diff_eval
from harness.runner import CannedRunner


def _agent() -> SubAgent:
    return SubAgent(
        name="diff-demo",
        system_prompt="be helpful",
        model="demo-model",
    )


async def main() -> int:
    transcript: list[str] = []

    # Three single-prompt cases. Each runner answers each case in
    # sequence - the CannedRunner index advances per call.
    cases = [
        EvalCase(name="capital-of-france", prompts=["what is the capital of France?"]),
        EvalCase(name="capital-of-japan", prompts=["what is the capital of Japan?"]),
        EvalCase(name="capital-of-brazil", prompts=["what is the capital of Brazil?"]),
    ]

    # alpha and beta share a reply script - they agree on every case.
    # gamma diverges on the second case ("capital-of-japan") only.
    consensus_replies = ["Paris", "Tokyo", "Brasilia"]
    runners = {
        "alpha": CannedRunner(replies=list(consensus_replies)),
        "beta": CannedRunner(replies=list(consensus_replies)),
        "gamma": CannedRunner(replies=["Paris", "Kyoto", "Brasilia"]),
    }

    matrix = await diff_eval(cases, agent=_agent(), runners=runners)

    transcript.append("--- diff matrix ---")
    transcript.append(f"  cases: {[c.name for c in matrix.cases]}")
    transcript.append(f"  runners: {matrix.runner_names}")

    unanimous = matrix.unanimous()
    transcript.append("")
    transcript.append("--- unanimous cases (every runner agrees) ---")
    for case in unanimous:
        transcript.append(f"  {case.name}")

    outliers = matrix.outliers()
    transcript.append("")
    transcript.append("--- outlier rows (at least one dissenting runner) ---")
    if not outliers:
        transcript.append("  (none)")
    for outlier in outliers:
        transcript.append(
            f"  case={outlier.case.name!r} "
            f"dissenter={outlier.dissenting_runner!r} "
            f"consensus={outlier.consensus_runners}"
        )

    # Write the HTML report to a temp file. Show that the file is non-empty
    # without coupling the example to a checked-in fixture.
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "diff_report.html"
        matrix.report_html(report_path)
        size = report_path.stat().st_size
        transcript.append("")
        transcript.append("--- HTML report ---")
        transcript.append(f"  wrote {report_path.name} ({size} bytes)")

    print("\n".join(transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
