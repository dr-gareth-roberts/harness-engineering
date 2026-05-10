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

result = asyncio.run(attribute(
    session=record,
    target="the secret ingredient is rosebud",   # the output you want explained
    runner=runner,
    agent=agent,
    granularity="block",                          # or "sentence", "paragraph"
    similarity=JaccardSimilarity(),
))

for chunk in result.top_k(3):
    print(f"{chunk.score:.3f}  {chunk.text!r}")
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
  Wall-time = chunk_count × runner_latency. Use coarser granularity
  or fewer chunks if the runner is expensive.
- **`JaccardSimilarity` is whitespace-tokenized lowercase.** Cheap
  and predictable, but doesn't capture semantic similarity.
  `EmbeddingSimilarity` does, at the cost of model loading +
  inference.
- **The `target` is matched as a substring of the assistant
  response by default.** Pass `target_match="exact"` if you want
  exact equality.
- **Ablation re-runs your tool handlers** unless `ReplayRunner` is
  the runner. For idempotent handlers this is fine; for side-
  effecting ones, use replay.

## Related

- [`examples/attribute.py`](https://github.com/dr-gareth-roberts/harness-engineering/blob/main/examples/attribute.py)
- [`harness.replay`](replay.md) — pair attribution with replay for cost-free re-runs.

## API reference

::: harness.attribute
