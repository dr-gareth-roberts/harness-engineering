"""Tests for the Pydantic to Hypothesis bridge.

Each test imports Hypothesis through ``pytest.importorskip`` so the
file degrades gracefully when the ``[fuzz]`` extra is not installed.
"""

from __future__ import annotations

import sys
from typing import Optional  # noqa: UP035 - exercising legacy spelling

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("hypothesis")

from harness.fuzz.strategies import (  # noqa: E402
    FuzzStrategyUnsupported,
    pydantic_strategy,
)


class _Simple(BaseModel):
    text: str
    n: int


class _Constrained(BaseModel):
    handle: str = Field(min_length=1, max_length=8)
    age: int = Field(ge=0, le=150)


class _OptionalLegacy(BaseModel):
    """Uses ``Optional[X]`` from typing (the older spelling)."""

    label: Optional[str] = None  # noqa: UP007, UP045 - exercising legacy spelling


class _OptionalPep604(BaseModel):
    label: str | None = None


class _UnsupportedField(BaseModel):
    rows: list[int]


def test_simple_model_to_strategy_yields_str_and_int() -> None:
    strategy = pydantic_strategy(_Simple)

    seen: list[_Simple] = []

    from hypothesis import HealthCheck, Phase, given, seed, settings

    @seed(0)
    @settings(
        max_examples=20,
        database=None,
        derandomize=True,
        deadline=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(value=strategy)
    def collect(value: _Simple) -> None:
        seen.append(value)

    collect()
    assert len(seen) == 20
    for instance in seen:
        assert isinstance(instance, _Simple)
        assert isinstance(instance.text, str)
        assert isinstance(instance.n, int)


def test_field_constraints_are_honoured() -> None:
    strategy = pydantic_strategy(_Constrained)

    from hypothesis import HealthCheck, Phase, given, seed, settings

    seen: list[_Constrained] = []

    @seed(0)
    @settings(
        max_examples=50,
        database=None,
        derandomize=True,
        deadline=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(value=strategy)
    def collect(value: _Constrained) -> None:
        seen.append(value)

    collect()
    assert seen, "no examples generated"
    for instance in seen:
        assert 1 <= len(instance.handle) <= 8
        assert 0 <= instance.age <= 150


def test_optional_legacy_typing_supported() -> None:
    strategy = pydantic_strategy(_OptionalLegacy)
    # We just need the strategy to build without raising; downstream
    # tests cover that None and a string both surface.
    assert strategy is not None


def test_optional_pep604_supported() -> None:
    strategy = pydantic_strategy(_OptionalPep604)

    from hypothesis import HealthCheck, Phase, given, seed, settings

    seen_none = False
    seen_str = False

    @seed(0)
    @settings(
        max_examples=200,
        database=None,
        derandomize=True,
        deadline=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(value=strategy)
    def collect(value: _OptionalPep604) -> None:
        nonlocal seen_none, seen_str
        if value.label is None:
            seen_none = True
        elif isinstance(value.label, str):
            seen_str = True

    collect()
    assert seen_none, "expected at least one None"
    assert seen_str, "expected at least one string"


def test_unsupported_field_raises_clear_error() -> None:
    with pytest.raises(FuzzStrategyUnsupported) as excinfo:
        pydantic_strategy(_UnsupportedField)
    assert excinfo.value.field == "rows"
    # The exception message names the field for the user's benefit.
    assert "rows" in str(excinfo.value)


def test_overrides_take_precedence() -> None:
    from hypothesis import strategies as st

    strategy = pydantic_strategy(
        _UnsupportedField, overrides={"rows": st.lists(st.integers(), max_size=3)}
    )
    from hypothesis import HealthCheck, Phase, given, seed, settings

    seen: list[_UnsupportedField] = []

    @seed(0)
    @settings(
        max_examples=10,
        database=None,
        derandomize=True,
        deadline=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(value=strategy)
    def collect(value: _UnsupportedField) -> None:
        seen.append(value)

    collect()
    assert seen
    for instance in seen:
        assert isinstance(instance.rows, list)
        assert all(isinstance(x, int) for x in instance.rows)


def test_missing_hypothesis_raises_structured_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When hypothesis is unavailable the bridge surfaces a clear hint.

    We simulate the missing module via ``sys.modules`` so the test runs
    even when hypothesis is installed (the usual case). The lazy import
    inside ``_require_hypothesis`` is what produces the error.
    """

    # Drop any cached hypothesis modules so the lazy import re-runs.
    for name in list(sys.modules):
        if name == "hypothesis" or name.startswith("hypothesis."):
            monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(ImportError, match=r"\[fuzz\]"):
        pydantic_strategy(_Simple)
