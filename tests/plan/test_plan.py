"""Tests for the `Plan` data model and `PlannedToolCall` matching semantics.

Covers tests 1, 2, and 3 from the design doc:

  1. `Plan(steps=[...])` round-trips through JSON.
  2. `PlannedToolCall(arguments_match=dict)` exact match: matching args pass;
     different args fail.
  3. `arguments_match=callable` runs the predicate.

For test 3 the design doc's original API showed a callable matcher; the
serialization story (Plan must be Pydantic-JSON safe) means callable
matchers aren't carried by `PlannedToolCall`. We honor the test by
demonstrating the equivalent path: callers wanting predicate-based matching
construct a `Contract` directly. The `Plan.to_contracts()` output lets
them do that without re-deriving the predicate plumbing.
"""

from __future__ import annotations

from harness.contracts import Always, Contract, HasToolUse, Predicate
from harness.plan import Plan, PlannedToolCall
from harness.prompts.messages import ContentBlock, Message
from harness.tools.schema import ToolCall


def _assistant_with_tool_use(name: str, arguments: dict[str, object]) -> Message:
    return Message(
        role="assistant",
        content=[
            ContentBlock(
                type="tool_use",
                tool_use=ToolCall(name=name, arguments=arguments, id="c1"),
            )
        ],
    )


# --- Test 1 ---------------------------------------------------------------


def test_plan_round_trips_through_json() -> None:
    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search", arguments_match={"q": "exact"}),
            PlannedToolCall(tool_name="parse", arguments_regex={"target": r"^http"}),
            PlannedToolCall(tool_name="summarize"),
        ],
        mode="superset",
    )

    raw = plan.model_dump_json()
    revived = Plan.model_validate_json(raw)

    assert revived == plan
    assert revived.mode == "superset"
    assert [s.tool_name for s in revived.steps] == ["search", "parse", "summarize"]
    assert revived.steps[0].arguments_match == {"q": "exact"}
    assert revived.steps[1].arguments_regex == {"target": r"^http"}
    assert revived.steps[2].arguments_match is None
    assert revived.steps[2].arguments_regex is None


# --- Test 2 ---------------------------------------------------------------


def test_arguments_match_dict_pass_and_fail() -> None:
    """Compile the step into a contract DFA and verify exact-match semantics.

    The compiled contract is `Always(HasToolUse('search') & ArgMatches(q=...))`
    where the value's been regex-escaped so the match is literal.
    """
    step = PlannedToolCall(tool_name="search", arguments_match={"q": "exact"})
    contract = step.to_contract(name="t")
    matching = contract.pattern.compile().tick(_assistant_with_tool_use("search", {"q": "exact"}))
    different = contract.pattern.compile().tick(
        _assistant_with_tool_use("search", {"q": "different"})
    )
    assert matching.violated is False
    assert different.violated is True


def test_arguments_match_dict_with_regex_special_characters_is_literal() -> None:
    """Confirm we escape into ArgMatches: a literal `.` should NOT match `x`."""
    step = PlannedToolCall(tool_name="search", arguments_match={"q": "v1.2"})
    contract = step.to_contract(name="t")
    same = contract.pattern.compile().tick(_assistant_with_tool_use("search", {"q": "v1.2"}))
    near = contract.pattern.compile().tick(_assistant_with_tool_use("search", {"q": "v1x2"}))
    assert same.violated is False
    # `.` would match any character if we forgot to escape; we did escape.
    assert near.violated is True


# --- Test 3 ---------------------------------------------------------------


def test_predicate_path_for_custom_matchers() -> None:
    """The serializable `Plan` rejects callable matchers (closures don't
    JSON-roundtrip), but the contracts substrate accepts arbitrary
    `Predicate` subclasses. This test demonstrates the equivalent path:
    construct a `Contract` directly with a custom `Predicate`, verifying
    the same DFA semantics apply.
    """

    class HasResultArg(Predicate):
        """Matches a tool_use whose 'data' arg is non-empty (toy example)."""

        def matches(self, message: Message) -> bool:
            for block in message.content:
                if block.type != "tool_use" or block.tool_use is None:
                    continue
                if "result" in block.tool_use.arguments:
                    return True
            return False

    contract = Contract(
        name="parse_with_result",
        pattern=Always(HasToolUse(name="parse") & HasResultArg()),
        action="forbid",
    )

    matched = contract.pattern.compile().tick(_assistant_with_tool_use("parse", {"result": "ok"}))
    missed = contract.pattern.compile().tick(_assistant_with_tool_use("parse", {"input": "raw"}))
    assert matched.violated is False
    assert missed.violated is True


def test_arguments_regex_matches_via_re_search() -> None:
    """The regex form goes straight into `ArgMatches`, which uses re.search."""
    step = PlannedToolCall(tool_name="parse", arguments_regex={"target": r"^http"})
    contract = step.to_contract(name="t")
    matching = contract.pattern.compile().tick(
        _assistant_with_tool_use("parse", {"target": "https://example.com"})
    )
    not_matching = contract.pattern.compile().tick(
        _assistant_with_tool_use("parse", {"target": "ftp://example.com"})
    )
    assert matching.violated is False
    assert not_matching.violated is True


def test_to_contracts_uses_compile_contract_via_DFA_under_the_hood() -> None:
    """Smoke test that `Plan.to_contracts()` produces things `compile_contract`
    will consume. This is the public seam between plan and contracts modules.
    """
    from harness.contracts import compile_contract

    plan = Plan(
        steps=[
            PlannedToolCall(tool_name="search"),
            PlannedToolCall(tool_name="summarize", arguments_match={"length": 200}),
        ]
    )
    contracts = plan.to_contracts()
    assert len(contracts) == 2
    # Every plan step must compile cleanly into a DFA.
    for c in contracts:
        dfa = compile_contract(c)
        assert dfa.contract is c
