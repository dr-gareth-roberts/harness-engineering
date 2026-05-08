"""Pydantic v2 to Hypothesis bridge.

The bridge intentionally covers a small but useful slice:

* primitive scalars: ``str``, ``int``, ``float``, ``bool``
* ``Optional[X]`` (both ``typing.Optional[X]`` and ``X | None``)
* the ``annotated_types`` constraints attached by ``Field`` for
  ``min_length`` / ``max_length`` / ``ge`` / ``le``

Anything outside that envelope (lists, nested models, unions of two
non-None types, ``Decimal``, ``datetime``, etc.) raises
:class:`FuzzStrategyUnsupported` with the field name and the offending
type so the caller knows exactly which field to register manually.

``hypothesis`` is imported lazily inside :func:`pydantic_strategy` so the
``harness.fuzz`` package stays importable when the ``[fuzz]`` extra is
not installed; the first call surfaces a clean error.
"""

from __future__ import annotations

import types
import typing
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy


_FUZZ_EXTRA_MESSAGE = (
    "harness.fuzz requires the `hypothesis` package. "
    "Install with: pip install 'harness-engineering[fuzz]'"
)


class FuzzStrategyUnsupported(Exception):
    """Raised when the bridge cannot synthesize a Hypothesis strategy.

    Attributes:
        field: The name of the field that could not be handled.
        annotation: The type annotation we did not know how to map.
    """

    def __init__(self, field: str, annotation: Any) -> None:
        super().__init__(
            f"cannot derive Hypothesis strategy for field {field!r} "
            f"with annotation {annotation!r}; "
            f"register a custom strategy via the `overrides` argument."
        )
        self.field = field
        self.annotation = annotation


def _require_hypothesis() -> Any:
    """Import hypothesis lazily; raise a structured error if absent."""

    try:
        import hypothesis  # noqa: F401
        from hypothesis import strategies as st
    except ImportError as exc:  # pragma: no cover - covered via monkeypatch
        raise ImportError(_FUZZ_EXTRA_MESSAGE) from exc
    return st


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    """Return ``(is_optional, inner)`` for ``Optional[X]`` / ``X | None``.

    Handles both ``typing.Union[X, None]`` and the PEP 604 ``X | None``
    syntax. For non-optional annotations returns ``(False, annotation)``.
    """

    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1 and type(None) in typing.get_args(annotation):
            return True, args[0]
    return False, annotation


def _constraint_value(metadata: list[Any], names: tuple[str, ...]) -> Any | None:
    """Pluck a constraint value out of ``annotated_types`` metadata.

    Pydantic stores ``Field(ge=..., min_length=...)`` constraints as
    objects in ``FieldInfo.metadata``. The objects expose the limit via
    differently named attributes depending on the constraint, so we try a
    short list of likely names.
    """

    for entry in metadata:
        for name in names:
            if hasattr(entry, name):
                return getattr(entry, name)
    return None


def _strategy_for_annotation(
    field_name: str,
    annotation: Any,
    metadata: list[Any],
    st: Any,
) -> SearchStrategy[Any]:
    is_optional, inner = _is_optional(annotation)
    if is_optional:
        inner_strategy = _strategy_for_annotation(field_name, inner, metadata, st)
        return cast("SearchStrategy[Any]", st.one_of(st.none(), inner_strategy))

    if inner is str:
        kwargs: dict[str, Any] = {}
        min_length = _constraint_value(metadata, ("min_length",))
        max_length = _constraint_value(metadata, ("max_length",))
        if min_length is not None:
            kwargs["min_size"] = int(min_length)
        if max_length is not None:
            kwargs["max_size"] = int(max_length)
        return cast("SearchStrategy[Any]", st.text(**kwargs))
    if inner is bool:
        return cast("SearchStrategy[Any]", st.booleans())
    # bool is a subclass of int in Python, so check int *after* bool.
    if inner is int:
        kwargs = {}
        ge = _constraint_value(metadata, ("ge", "gt"))
        le = _constraint_value(metadata, ("le", "lt"))
        if ge is not None:
            kwargs["min_value"] = int(ge)
        if le is not None:
            kwargs["max_value"] = int(le)
        return cast("SearchStrategy[Any]", st.integers(**kwargs))
    if inner is float:
        return cast(
            "SearchStrategy[Any]",
            st.floats(allow_nan=False, allow_infinity=False),
        )

    raise FuzzStrategyUnsupported(field_name, annotation)


def pydantic_strategy(
    model: type[BaseModel],
    *,
    overrides: dict[str, SearchStrategy[Any]] | None = None,
) -> SearchStrategy[BaseModel]:
    """Return a Hypothesis strategy that yields valid instances of ``model``.

    Walks ``model.model_fields`` and maps each field's annotation to a
    Hypothesis strategy. Fields listed in ``overrides`` take precedence —
    use this to plug strategies for types the bridge cannot synthesize.

    Raises:
        FuzzStrategyUnsupported: if any field is not supported by the
            bridge and is not present in ``overrides``.
        ImportError: if the ``[fuzz]`` extra is not installed.
    """

    st = _require_hypothesis()
    overrides = overrides or {}
    field_strategies: dict[str, SearchStrategy[Any]] = {}
    for name, field_info in model.model_fields.items():
        if name in overrides:
            field_strategies[name] = overrides[name]
            continue
        field_strategies[name] = _strategy_for_annotation(
            name,
            field_info.annotation,
            list(field_info.metadata or []),
            st,
        )

    def _build(values: dict[str, Any]) -> BaseModel:
        return model(**values)

    return cast(
        "SearchStrategy[BaseModel]",
        st.fixed_dictionaries(field_strategies).map(_build),
    )
