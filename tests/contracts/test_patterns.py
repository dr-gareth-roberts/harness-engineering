from __future__ import annotations

from harness.contracts import (
    Always,
    ArgMatches,
    Earlier,
    Eventually,
    HasToolUse,
    Never,
    RoleIs,
    TextMatches,
)
from harness.prompts import assistant_tool_use, text
from harness.tools import ToolCall


def _delete_prod() -> ToolCall:
    return ToolCall(name="delete", arguments={"table": "prod_users"}, id="c1")


def _delete_stage() -> ToolCall:
    return ToolCall(name="delete", arguments={"table": "stage_users"}, id="c2")


def _search_q() -> ToolCall:
    return ToolCall(name="search", arguments={"q": "x"}, id="cs")


def test_never_violated_on_first_match() -> None:
    state = Never(HasToolUse(name="delete") & ArgMatches(table=r"^prod_")).compile()

    benign = state.tick(text("user", "hi"))
    assert benign.violated is False

    benign2 = state.tick(assistant_tool_use(_delete_stage()))
    assert benign2.violated is False

    bad = state.tick(assistant_tool_use(_delete_prod()))
    assert bad.violated is True

    # finalize() does not re-raise.
    assert state.finalize().violated is False


def test_eventually_violates_only_at_end_when_never_seen() -> None:
    state = Eventually(HasToolUse(name="search")).compile()

    # Several non-matching messages: no per-message violation.
    for _ in range(3):
        assert state.tick(text("user", "hi")).violated is False
    # End of session without a match -> finalize violates.
    assert state.finalize().violated is True

    # Reset and feed a matching message before finalize -> satisfied.
    state.reset()
    state.tick(assistant_tool_use(_search_q()))
    assert state.finalize().violated is False


def test_always_predicate_first_miss_is_violation() -> None:
    state = Always(RoleIs("assistant")).compile()

    # Assistant messages all pass.
    assert state.tick(text("assistant", "ok")).violated is False
    assert state.tick(text("assistant", "still ok")).violated is False

    # First user message -> violation.
    out = state.tick(text("user", "boom"))
    assert out.violated is True

    # Subsequent messages don't re-emit violations.
    out2 = state.tick(text("user", "again"))
    assert out2.violated is False


def test_earlier_when_pattern_gates_check_on_trigger() -> None:
    # "If an assistant message starts with 'Answer:', some earlier assistant
    #  message must have used `search`."
    pat = Earlier(HasToolUse(name="search")).when(RoleIs("assistant") & TextMatches(r"^Answer:"))
    state = pat.compile()

    # User question -> not a trigger, no violation.
    assert state.tick(text("user", "what is the answer?")).violated is False

    # Assistant emits Answer: WITHOUT searching first -> violation.
    out = state.tick(text("assistant", "Answer: 42"))
    assert out.violated is True

    # New session: search first, then answer -> no violation.
    state.reset()
    assert state.tick(assistant_tool_use(_search_q())).violated is False
    assert state.tick(text("assistant", "Answer: 42")).violated is False


def test_pattern_composition_always_with_earlier_when() -> None:
    """`Always(Earlier(A).when(B))` — every trigger checked against history.

    Non-trigger messages pass vacuously. The first time B fires without A
    earlier, a violation is emitted.
    """
    pat = Always(
        Earlier(HasToolUse(name="search")).when(RoleIs("assistant") & TextMatches(r"^Answer:"))
    )
    state = pat.compile()

    # Many non-trigger messages: all pass.
    assert state.tick(text("user", "tell me about Mars")).violated is False
    assert state.tick(text("system", "be helpful")).violated is False

    # First trigger without preceding search -> violation.
    out = state.tick(text("assistant", "Answer: it's red"))
    assert out.violated is True

    # Reset and try compliant flow.
    state.reset()
    assert state.tick(assistant_tool_use(_search_q())).violated is False
    assert state.tick(text("assistant", "Answer: 42")).violated is False
    assert state.finalize().violated is False
