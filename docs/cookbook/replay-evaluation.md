# Replay a session for evaluation

## Problem

You have a recorded agent trajectory. You want to:

- Replay it deterministically (no API call) to reproduce a bug.
- Run a batch of test cases and grade the results.
- Compare the same prompts across providers (Anthropic vs OpenAI vs
  a local Ollama) to see where they disagree.

## Solution sketch

`harness.replay` ships three primitives, each layered on the next:

1. **`ReplayRunner`** — a `Runner` backed by a recorded `SessionRecord`.
   No API calls; deterministic; useful for tests, debugging, and
   counterfactual analysis.
2. **`run_eval`** — drives a list of `EvalCase`s through a runner and
   returns a structured report. Use any runner: real, replay,
   counterfactual.
3. **`diff_eval`** — runs each case across multiple runners
   simultaneously, then surfaces unanimous-agree vs outlier verdicts
   with an HTML report.

## Working code

### Replay a single recorded session

```python
import asyncio

from harness import Orchestrator, ReplayRunner, SessionRecord, SubAgent, text

# `SessionRecord.from_jsonl(...)` works the same way; `MemoryStore`
# is the production interface for fetching by session_id.
record = SessionRecord.model_validate_json(open("session.json").read())
runner = ReplayRunner.from_record(record)

orchestrator = Orchestrator(runner=runner, dispatcher=record.dispatcher_or_empty(), hooks=...)
asyncio.run(orchestrator.run(record.agent, record.messages[:2]))  # whatever prefix you want
```

### Batch evaluate

```python
from harness import EvalCase, run_eval

cases = [
    EvalCase(name="weather-berlin", prompts=["What's the weather in Berlin?"]),
    EvalCase(name="weather-tokyo", prompts=["What's the weather in Tokyo?"]),
    EvalCase(name="rude-prompt",   prompts=["Insult me."]),
]

result = await run_eval(
    runner=AnthropicRunner(...),
    agent=SubAgent(name="weather", system_prompt="", model="claude-opus-4-7", allowed_tools=["weather"]),
    cases=cases,
)
print(result.summary())  # cases x outcomes table
```

### Cross-provider differential matrix

```python
from harness import diff_eval, AnthropicRunner, OpenAICompatRunner

matrix = await diff_eval(
    cases=cases,
    runners={
        "anthropic":   AnthropicRunner(...),
        "openai":      OpenAICompatRunner(...),
        "local_llama": OpenAICompatRunner(..., base_url="http://localhost:11434/v1"),
    },
    agent=agent,
)

# Find the cases where exactly one runner disagrees with the others.
for outlier in matrix.outliers():
    print(f"{outlier.case_name}: {outlier.dissenter} dissents from {outlier.majority}")

# Render an HTML report for review.
matrix.write_html("./eval-report.html")
```

The HTML report shows per-case responses side-by-side, with cluster
detection highlighting the dissenter when 2 of 3 agree.

## Counterfactual: "what if I changed prompt X?"

Sometimes you want to mutate one turn in a recorded session and see
how the rest plays out:

```python
from harness import counterfactual, RewriteTurn

forked = await counterfactual(
    record,
    mutation=RewriteTurn(turn_index=2, new_text="Be more concise."),
    runner=AnthropicRunner(...),
)
# `forked` is the divergent SessionRecord starting at turn 2.
```

Mutations available: `RewriteTurn`, `ReplaceToolResult`, `InsertTurn`,
`DeleteTurn`. Compose them however you want.

## Gotchas

- **Tool call IDs vary across providers** — Anthropic generates
  `toolu_01...`, OpenAI generates `call_...`. `compare_sessions` and
  `diff_eval` ignore IDs in the comparison so cross-provider diffs
  stay meaningful.
- **Replay is exact** — if your test re-creates the same dispatcher,
  the replay runs through the same handler dispatches. If you want
  the model's text but not the side effects, mark tools idempotent
  and use `MemoryStore.snapshot` to fork.
- **`diff_eval` runs runners in parallel** via `asyncio.gather`.
  Wall time approx slowest runner times cases. If one runner is
  much slower, consider running it separately.
- **HTML reports are static** — they don't pull live data. Re-run
  to refresh.

## Related

- [`harness.replay`](../modules/replay.md) — module reference.
- [`examples/counterfactual.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/counterfactual.py),
  [`examples/diff_eval.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/diff_eval.py)
  — runnable end-to-end demos.
- [Cookbook: Debug a trajectory](debug-a-trajectory.md) — when
  replay reveals a bad turn, debug it interactively.
