# `harness.attribute`

Causal provenance via leave-one-out ablation: re-runs a session with
each input chunk removed and ranks chunks by influence on a target
output. `JaccardSimilarity` and `LengthRatio` are zero-dep;
`EmbeddingSimilarity` (sentence-transformers) lives behind the
`[attribute]` extra.

::: harness.attribute
