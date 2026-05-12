# `harness.attribute`

Causal provenance via leave-one-out ablation: re-runs a session with
each input chunk removed and ranks chunks by influence on a target
output. `JaccardSimilarity` and `LengthRatio` are zero-dep;
`EmbeddingSimilarity` (sentence-transformers) lives behind the
`[attribute]` extra.

## When to reach for this

- You want to know which part of a long input the model actually
  cared about.
- You're debugging a hallucination and want to find the prompt
  fragment that triggered it.
- You want to rank a retrieval system's chunks by their downstream
  influence, not their retrieval score.

## Quick example

```python
import asyncio
from harness import attribute, JaccardSimilarity

# `target_message_index` points at the assistant message whose
# content you want explained. Negative indices count from the end.
result = asyncio.run(attribute(
    record,
    target_message_index=-1,
    runner=runner,
    agent=agent,
    granularity="block",          # "message" | "block" | "sentence"
    similarity=JaccardSimilarity(),
))

print(f"called runner {result.actual_calls} times")
for chunk in result.top_k(3):
    print(f"{chunk.score:.3f}  {chunk.preview!r}")
```

Estimate the cost before paying for it:

```python
result = asyncio.run(attribute(
    record, -1, runner, agent,
    granularity="block",
    estimate_only=True,
))
print(f"would have made {result.estimated_calls} runner calls")
```

For embedding-based similarity:

```bash
uv add 'harness-engineering[attribute]'
```

```python
from harness.attribute import EmbeddingSimilarity
similarity = EmbeddingSimilarity(model_name="all-MiniLM-L6-v2")
```

## Gotchas

- **Leave-one-out is O(n) re-runs** where n is chunk count.
  Wall-time = chunk_count × runner_latency. Coarser granularity
  or `estimate_only=True` lets you budget before committing.
- **`JaccardSimilarity` is whitespace-tokenized lowercase.** Cheap
  and predictable, but doesn't capture semantic similarity.
  `EmbeddingSimilarity` does, at the cost of model loading +
  inference.
- **Target text is read from the record, not regenerated.** The
  original assistant response at `target_message_index` is the
  anchor; ablation runs are compared against it. We never re-run
  the original session.
- **Whatever `runner` does, ablation does N+1 times.** A real
  vendor runner will dispatch tool handlers per call; a
  `ReplayRunner` just returns canned assistant messages with no
  dispatch. Pair attribution with `ReplayRunner.from_record(...)`
  for cost-free re-runs over already-recorded outputs.

## Related

- [`examples/attribute.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/attribute.py)
- [`harness.replay`](replay.md) — pair attribution with replay for cost-free re-runs.

## API reference

::: harness.attribute
