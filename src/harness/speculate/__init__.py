"""Speculative tool execution (#5).

Pre-execute likely tool calls while the model is still generating its
response. On hit, the result is already cached — near-zero added latency.
Wrong predictions are cheap (one wasted tool call) and cancelled when
they don't match what the model actually asked for.

Public API:

    from harness.speculate import (
        Speculator, Predictor,
        LastCallPredictor, SequencePredictor,
        SpeculationLaunched, SpeculationHit, SpeculationMiss,
    )

    speculator = Speculator(
        predictor=LastCallPredictor(history_window=3),
        max_speculations=2,
        only_idempotent=True,
    )
    runner = AnthropicRunner(dispatcher, hooks, speculator=speculator)

The speculator only fires for tools marked `Tool.idempotent=True`. That
flag is a *promise* by the tool author; see `Speculator`'s class
docstring for the semantics.
"""

from harness.speculate.events import (
    SpeculationHit,
    SpeculationLaunched,
    SpeculationMiss,
)
from harness.speculate.predictor import (
    LastCallPredictor,
    Predictor,
    SequencePredictor,
)
from harness.speculate.speculator import Speculator

__all__ = [
    "LastCallPredictor",
    "Predictor",
    "SequencePredictor",
    "SpeculationHit",
    "SpeculationLaunched",
    "SpeculationMiss",
    "Speculator",
]
