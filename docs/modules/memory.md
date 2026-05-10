# `harness.memory`

`SessionRecord` (Pydantic, JSON-serializable trajectory snapshot),
the `MemoryStore` Protocol, `InMemoryStore` and `FileStore`
implementations, and a `Session` helper that snapshots after every
turn.

## When to reach for this

- You need to persist agent trajectories for later replay,
  evaluation, or audit.
- You want the snapshot to survive process restarts (`FileStore`)
  or stay in-memory for tests (`InMemoryStore`).
- You're building a custom backend (SQLite, S3, Postgres) — the
  `MemoryStore` Protocol is one async method per CRUD operation.

## Quick example

```python
from harness import FileStore, Session, SubAgent, text

store = FileStore("./sessions")  # one JSON file per session_id
session = Session(
    store=store,
    agent=SubAgent(name="bot", system_prompt="", model="claude", allowed_tools=[]),
)

# Run a turn — Session snapshots after each one.
await session.run(orchestrator, [text("user", "Hello")])
await session.run(orchestrator, [text("user", "Tell me more")])

# Later — load it back.
record = await store.load(session.id)
print(f"{len(record.messages)} messages")
```

## Gotchas

- **`SessionRecord.agent` is captured at session start.** If you
  mutate the agent between turns, the recorded snapshot stays at
  the original. By design — the agent is the agent definition, not
  the running state.
- **`InMemoryStore` doesn't persist** across process restarts. Use
  `FileStore` for anything you want to keep.
- **`FileStore` writes one file per session_id**. For high-cardinality
  workloads, consider a custom SQLite-backed store implementing
  `MemoryStore`.
- **`Session.run` snapshots after every turn** — fine for normal
  loops, expensive if the conversation is large and you're calling
  in a tight loop. Snapshot manually with `store.save(record)` if
  you need finer control.

## Related

- [`harness.replay`](replay.md) — `ReplayRunner.from_record(...)` works on any `SessionRecord`.
- [`harness.speculate`](speculate.md) — `CrossSessionPredictor` reads from a `MemoryStore`.
- [`examples/end_to_end.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/end_to_end.py)

## API reference

::: harness.memory
