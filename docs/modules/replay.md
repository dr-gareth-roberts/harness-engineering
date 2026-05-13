# `harness.replay`

`ReplayRunner` for deterministic playback of recorded trajectories,
`run_eval` and `compare_sessions` for offline evaluation,
`counterfactual` for prefix-mutation + continuation, and `diff_eval`
for cross-provider differential matrices with HTML reports.

## When to reach for this

- You want to reproduce a bug from a recorded session deterministically.
- You want to evaluate prompt / tool changes against a batch of
  test cases.
- You want to compare the same prompts across providers (Anthropic
  vs OpenAI vs local) and see who disagrees.
- You want "what if I changed turn 3?" — counterfactual analysis.

## Quick example

<!-- reason: illustrative; needs session.json on disk -->
<!--pytest.mark.skip-->
```python
import asyncio
from pathlib import Path

from harness import (
    Dispatcher,
    HookRunner,
    Orchestrator,
    ReplayRunner,
    RewriteTurn,
    SessionRecord,
    counterfactual,
    text,
)

# 1. Replay a recorded session. ReplayRunner returns the recorded
#    assistant messages in order; it is input-blind, so wiring it
#    behind an Orchestrator gives a deterministic playback.
record = SessionRecord.model_validate_json(Path("session.json").read_text())
replay = ReplayRunner.from_record(record)
orchestrator = Orchestrator(Dispatcher([]), HookRunner(), replay)

# 2. Batch evaluation and cross-provider matrices live in the
#    cookbook — see Related below for `run_eval` and `diff_eval`.

# 3. Counterfactual: what if turn 2 had said "Be terse."?
mutation = RewriteTurn(index=2, new_message=text("user", "Be terse."))
forked = asyncio.run(
    counterfactual(record, mutation, replay, orchestrator)
)
```

The cookbook page goes deeper:
[Replay evaluation](../cookbook/replay-evaluation.md) walks
through batch evaluation and the cross-provider matrix end-to-end,
including the HTML report.

## Gotchas

- **Tool call IDs vary across providers.** Anthropic generates
  `toolu_01...`, OpenAI generates `call_...`. The comparison
  helpers normalize them out so cross-provider diffs stay
  meaningful.
- **`ReplayRunner` is input-blind.** It returns the recorded
  assistant messages in order regardless of the current dispatcher
  state, agent, or message list. Tool handlers run only if the
  recorded messages contain `tool_use` blocks *and* you wire
  ReplayRunner into something that dispatches them (the bare
  `Orchestrator` doesn't). For tests that should re-execute tool
  handlers, drive a real runner — replay is for canned playback.
- **`counterfactual` needs an orchestrator and a runner.** The
  orchestrator carries `dispatcher`, `hooks`, and `telemetry`;
  the runner you supply produces the new continuation after the
  mutated prefix. The orchestrator's own runner is ignored.
- **Cross-provider matrices run runners in parallel** via
  `asyncio.gather`; wall time tracks the slowest runner × cases.
  Consider running slow / expensive providers separately.
- **HTML reports are static.** Re-run to refresh.

## Related

- [Cookbook: Replay evaluation](../cookbook/replay-evaluation.md) — extended walkthrough.
- [`examples/counterfactual.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/counterfactual.py),
  [`examples/diff_eval.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/diff_eval.py)
- [`harness.memory`](memory.md) — `SessionRecord` is the input shape.
- [`harness.debug`](debug.md) — replay + breakpoints debug bad trajectories.

## API reference

::: harness.replay
