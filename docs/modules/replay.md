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

```python
import asyncio
from harness import (
    ReplayRunner,
    SessionRecord,
    EvalCase,
    counterfactual,
    RewriteTurn,
)

# 1. Replay a recorded session.
record = SessionRecord.model_validate_json(open("session.json").read())
runner = ReplayRunner.from_record(record)

# 2. Batch evaluation. See cookbook for `run_eval` and `diff_eval`
#    full call sites — they're imported the same way:
#       from harness import run_eval, diff_eval
#    cases = [EvalCase(name="t1", prompts=["..."]), ...]
#    result = await run_eval(runner=..., agent=..., cases=cases)

# 3. Counterfactual: what if turn 2 had said "be terse"?
mutation = RewriteTurn(turn_index=2, new_text="Be terse.")
forked = asyncio.run(counterfactual(record, mutation=mutation, runner=runner))
```

The cookbook page goes deeper:
[Replay evaluation](../cookbook/replay-evaluation.md) walks
through batch evaluation and the cross-provider matrix end-to-end,
including the HTML report.

## Gotchas

- **Tool call IDs vary across providers.** Anthropic generates
  `toolu_01...`, OpenAI generates `call_...`. The comparison
  helpers ignore IDs so cross-provider diffs stay meaningful.
- **`ReplayRunner` re-runs your tool handlers** by default — replay
  is exact, including dispatches. Mark tools idempotent if you
  want to skip side effects on replay.
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
