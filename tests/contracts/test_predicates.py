from __future__ import annotations

from harness.contracts import ArgMatches, HasToolUse, RoleIs, TextMatches
from harness.prompts import assistant_tool_use, text
from harness.tools import ToolCall


def test_has_tool_use_matches_named_assistant_block() -> None:
    call = ToolCall(name="search", arguments={"q": "hi"}, id="c1")
    msg = assistant_tool_use(call)

    assert HasToolUse(name="search").matches(msg) is True
    assert HasToolUse(name="other").matches(msg) is False
    # `name=None` matches any tool_use.
    assert HasToolUse().matches(msg) is True
    # Plain text assistant message does not match.
    assert HasToolUse(name="search").matches(text("assistant", "hello")) is False
    # User message with same shape never matches HasToolUse.
    assert HasToolUse(name="search").matches(text("user", "hello")) is False


def test_arg_matches_against_call_arguments() -> None:
    call = ToolCall(name="delete", arguments={"table": "prod_users"}, id="c1")
    msg = assistant_tool_use(call)

    assert ArgMatches(table=r"^prod_").matches(msg) is True
    # Different prefix does not match.
    other = ToolCall(name="delete", arguments={"table": "stage_users"}, id="c2")
    assert ArgMatches(table=r"^prod_").matches(assistant_tool_use(other)) is False
    # Missing argument does not match.
    assert ArgMatches(missing=r".*").matches(msg) is False


def test_role_is_strict_match() -> None:
    user_msg = text("user", "hi")
    assistant_msg = text("assistant", "hi")
    system_msg = text("system", "hi")

    assert RoleIs("user").matches(user_msg) is True
    assert RoleIs("assistant").matches(user_msg) is False
    assert RoleIs("assistant").matches(assistant_msg) is True
    assert RoleIs("system").matches(system_msg) is True


def test_text_matches_runs_on_concatenated_text() -> None:
    msg = text("assistant", "Answer: forty-two")
    assert TextMatches(r"^Answer:").matches(msg) is True
    assert TextMatches(r"forty-two$").matches(msg) is True
    assert TextMatches(r"hello").matches(msg) is False
    # No text at all (e.g. tool_use only) does not match.
    call = ToolCall(name="x", arguments={})
    assert TextMatches(r".*").matches(assistant_tool_use(call)) is False


def test_and_or_compose_predicates() -> None:
    msg = text("assistant", "Answer: yes")
    p_and = RoleIs("assistant") & TextMatches(r"^Answer:")
    p_or = RoleIs("user") | TextMatches(r"^Answer:")

    assert p_and.matches(msg) is True
    # Neither side of `&` -> false.
    assert (RoleIs("user") & TextMatches(r"^Answer:")).matches(msg) is False
    assert p_or.matches(msg) is True
    # Both sides false -> false.
    assert (RoleIs("user") | TextMatches(r"^nope")).matches(msg) is False
