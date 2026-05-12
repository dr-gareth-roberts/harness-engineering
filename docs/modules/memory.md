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
from harness import FileStore, Session, SubAgent

store = FileStore("./sessions")  # one JSON file per session_id
agent = SubAgent(name="bot", system_prompt="", model="canned", allowed_tools=[])
# `orchestrator` is your Orchestrator from harness.agents.
session = Session(orchestrator, agent, store)

# `send()` runs one turn: appends the user message, calls the
# orchestrator, appends the reply, snapshots to the store.
await session.send("Hello")
await session.send("Tell me more")

# Later — load it back. `load()` may return None if the id is unknown.
record = await store.load(session.session_id)
assert record is not None
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
- **`Session.send` snapshots after every call** — fine for normal
  loops, expensive if the conversation is large and you're calling
  in a tight loop. Snapshot manually with `store.save(record)` if
  you need finer control.
- **Single-writer per session_id.** Two `Session.restore(same_id)`
  instances racing `send()` is last-writer-wins. Treat each
  `Session` as owned by a single coroutine.

## Related

- [`harness.replay`](replay.md) — `ReplayRunner.from_record(...)` works on any `SessionRecord`.
- [`harness.speculate`](speculate.md) — `CrossSessionPredictor` reads from a `MemoryStore`.
- [`examples/end_to_end.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/end_to_end.py)

## API reference

::: harness.memory
