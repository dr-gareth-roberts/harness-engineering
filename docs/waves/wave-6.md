## Wave 6 — per-event speculator cancellation

### Goal
Free the speculation handler runtime that was being burned between
stream-end and the iteration's `finally:` block. Pre-Wave-6 the runner
called `await stream.get_final_message()` (waiting for the full
message) and only cancelled unmatched speculations at iteration end —
which fires *after* the model's emitted tool_use blocks have all been
dispatched. Pre-event, an unmatched 5-second speculation runs through
the entire dispatch phase. Post-Wave-6, the runner iterates stream
events as they arrive, surfaces each `tool_use` block to the speculator
via a new `observe()` call, then cancels everything still unobserved
the moment the stream ends — *before* dispatch.

### Status
Shipped on `feature/speculator-per-event`. Single-coherent refactor
done in main, no parallel agents — runner + fake + tests are too
tightly coupled to split.

### Cancellation timing — what we actually do

The user's prompt phrased this as "cancel pending tasks at
ContentBlockStopEvent for tool_use." The advisor flagged that as
*per-block* cancellation: with `max_speculations > 1`, deciding when
a speculation is "definitively dead" mid-stream requires policy that
isn't worth the complexity for MVP.

This wave cancels at *stream-end* (via `cancel_unobserved`) instead.
That captures the bulk of the win — the dispatch phase no longer
runs alongside burning speculation handlers — without the policy
complexity. End() (in the iteration's finally) still acts as a
final safety net.

The protocol is shaped so eager per-block cancellation could be
added later without breaking changes: `observe()` is called per-block,
so a future Speculator could implement eager cancellation in observe()
itself; today's implementation just marks the entry as observed and
defers cancellation to `cancel_unobserved`.

### Approach

**Protocol additions (`a39cfe0`, pre-step):**

`SpeculatorProtocol` gains two lifecycle methods:

```python
async def observe(self, call: ToolCall) -> None: ...
async def cancel_unobserved(self) -> None: ...
```

`Speculator` tracks observation per pending entry via a new `_Pending`
dataclass (replacing the old `tuple[ToolCall, Task]`). `observe(call)`
walks the pending list and marks the first unobserved match as
observed; `cancel_unobserved()` cancels and drains every pending entry
not marked observed. `try_resolve` and `end` are unchanged in shape but
read from the new dataclass.

Test stubs (`tests/runner/test_anthropic.py`,
`tests/runner/test_openai_compat.py`) get matching no-op
implementations so structural protocol compatibility holds. Existing
28 speculator tests + 42 runner tests still pass without modification.

**Runner refactor (single commit, in branch):**

`AnthropicRunner.__call__` now iterates the stream:

```python
async with self._client.messages.stream(**request) as stream:
    async for event in stream:
        if (
            self._speculator is not None
            and getattr(event, "type", None) == "content_block_stop"
            and getattr(getattr(event, "content_block", None), "type", None) == "tool_use"
        ):
            block = event.content_block
            await self._speculator.observe(
                ToolCall(name=block.name, arguments=dict(block.input), id=block.id)
            )
    response = await stream.get_final_message()

if self._speculator is not None:
    await self._speculator.cancel_unobserved()
```

`get_final_message()` after iteration mirrors the SDK's behavior —
`until_done()` is a no-op once the stream is consumed, the snapshot
stays accumulated.

**Fake extension:**

`tests/runner/fakes.FakeMessage` gains an optional `events: list[Any] | None`
field. When None (the default), `_FakeStream.__aiter__` auto-derives
one `FakeContentBlockStopEvent` per entry in `content` — zero-config
for existing tests. When set explicitly, tests can script specific
arrival orders (text-then-tool, multiple tools, scrambled-vs-content,
zero events, etc.). `get_final_message` returns the same `FakeMessage`
whether the stream was iterated first or not.

### Tests added

| File | Test | Pins |
|---|---|---|
| `tests/speculate/test_speculator.py` | `test_observe_marks_first_unobserved_matching_pending_spec` | observe records first unobserved match; cancel_unobserved leaves it alone |
| | `test_observe_with_no_match_is_a_noop` | observe with no matching pending is silent |
| | `test_cancel_unobserved_with_no_pending_is_noop` | safe to call when begin returned without launching anything |
| | `test_cancel_unobserved_runs_fast_when_handler_is_slow` | the perf claim — drain time vs handler runtime |
| | `test_observe_then_try_resolve_resolves_observed_spec` | full happy-path lifecycle |
| | `test_observe_claims_separate_entries_for_duplicate_calls` | two specs of the same shape stay distinct |
| `tests/runner/test_anthropic.py` | `test_runner_calls_observe_for_each_tool_use_block_in_stream` | observe fires per tool_use, in stream order |
| | `test_runner_does_not_observe_text_block_stop_events` | text blocks don't surface |
| | `test_runner_with_speculator_none_iterates_stream_without_error` | back-compat: no-speculator path still works with event iteration |
| | `test_runner_explicit_events_drive_observe_in_order` | order follows event arrival, not content list |
| | `test_unobserved_speculation_does_not_complete_when_dispatch_diverges` | runner-level correctness: unmatched speculation never reaches "done" |

11 new tests, 465 total (was 454).

### Verification gate

```
ruff check       — clean
ruff format     — 166 files already formatted
mypy --strict src/harness  — clean (79 source files)
pytest          — 465 passed
```

### Commits

```
a39cfe0  feat(speculate): add observe + cancel_unobserved to SpeculatorProtocol
5bbd0bf  feat(runner): per-event observe + cancel_unobserved in AnthropicRunner
3ee6bba  docs: progress.md log of Wave 6
```
