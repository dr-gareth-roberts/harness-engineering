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
from pathlib import Path

from harness import (
    Dispatcher, HookRunner, Orchestrator, ReplayRunner,
    SessionRecord, text,
)

record = SessionRecord.model_validate_json(
    Path("session.json").read_text()
)
replay = ReplayRunner.from_record(record)

# ReplayRunner is input-blind: it returns the recorded assistant
# messages in order. Wire it behind an Orchestrator with whatever
# dispatcher / hooks you want for the test. The bare Orchestrator
# below does not re-dispatch tool calls — wire a real Dispatcher
# if your test needs handlers to fire.
orchestrator = Orchestrator(Dispatcher([]), HookRunner(), replay)
reply = asyncio.run(
    orchestrator.run(record.agent, [text("user", "any prompt")])
)
print(reply.content[0].text)
```

### Batch evaluate

```python
from harness import (
    AnthropicRunner, Dispatcher, EvalCase, HookRunner, Orchestrator,
    SubAgent, run_eval,
)

cases = [
    EvalCase(name="weather-berlin", prompts=["What's the weather in Berlin?"]),
    EvalCase(name="weather-tokyo",  prompts=["What's the weather in Tokyo?"]),
    EvalCase(name="rude-prompt",    prompts=["Insult me."]),
]

agent = SubAgent(
    name="weather",
    system_prompt="",
    model="claude-opus-4-7",
    allowed_tools=["weather"],
)
orchestrator = Orchestrator(
    dispatcher,                              # your Dispatcher
    HookRunner(),
    AnthropicRunner(dispatcher, HookRunner()),
)

results = await run_eval(cases, orchestrator=orchestrator, agent=agent)
for r in results:
    last = r.record.messages[-1]
    print(f"{r.case.name:20s} {r.duration_ms:6.0f}ms  {last.role}")
```

`run_eval` returns a `list[EvalResult]`; each element carries the
case, the produced `SessionRecord`, and a duration. There is no
built-in scorer — your grading is whatever assertion you want to run
over `r.record.messages`.

### Cross-provider differential matrix

```python
from harness import AnthropicRunner, OpenAICompatRunner, diff_eval

matrix = await diff_eval(
    cases,
    agent=agent,
    runners={
        "anthropic":   AnthropicRunner(dispatcher, HookRunner()),
        "openai":      OpenAICompatRunner(dispatcher, HookRunner()),
        "local_llama": OpenAICompatRunner(
            dispatcher, HookRunner(),
            base_url="http://localhost:11434/v1",
        ),
    },
    dispatcher=dispatcher,
)

# Each entry pairs one dissenting runner with the consensus cluster.
for outlier in matrix.outliers():
    print(
        f"{outlier.case.name}: {outlier.dissenting_runner} dissents from "
        f"{outlier.consensus_runners}"
    )

# Render a static HTML report for review.
matrix.report_html("./eval-report.html")
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
    RewriteTurn(index=2, new_message=text("user", "Be more concise.")),
    AnthropicRunner(dispatcher, HookRunner()),
    orchestrator,
)
# `forked` is a fresh SessionRecord whose messages are the original
# prefix up to and including the rewritten turn, plus one new
# continuation produced by the live runner above.
```

Mutations available: `RewriteTurn`, `ReplaceToolResult`, `InsertTurn`,
`DeleteTurn`. Each is a frozen dataclass with primitive fields, so
they serialize alongside the session and the counterfactual is
reproducible.

## Gotchas

- **Tool call IDs vary across providers** — Anthropic generates
  `toolu_01...`, OpenAI generates `call_...`. `compare_sessions` and
  `diff_eval` strip IDs in their comparisons so cross-provider diffs
  stay meaningful.
- **`ReplayRunner` is input-blind** — it returns the recorded
  assistant messages in order regardless of the inputs you feed
  into the orchestrator. Tool handlers fire only if the recorded
  messages contain tool_use blocks *and* you wire ReplayRunner into
  something that dispatches them (a bare Orchestrator does not).
- **`diff_eval` runs runners in parallel** via `asyncio.gather`.
  Wall time ≈ slowest_runner × len(cases). Exceptions per
  (case, runner) are captured so the matrix stays rectangular.
- **HTML reports are static** — they don't pull live data. Re-run
  to refresh.

## Related

- [`harness.replay`](../modules/replay.md) — module reference.
- [`examples/counterfactual.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/counterfactual.py),
  [`examples/diff_eval.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/diff_eval.py)
  — runnable end-to-end demos.
- [Cookbook: Debug a trajectory](debug-a-trajectory.md) — when
  replay reveals a bad turn, debug it interactively.
