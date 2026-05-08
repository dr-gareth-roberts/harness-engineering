"""`derive_plan(...)` — call a planner agent through a runner and parse the
JSON it emits into a `Plan`.

The planner is expected to return an assistant message whose concatenated
text content is a JSON-encoded `Plan` (matching `Plan.model_json_schema()`).
Real callers pass `plan_schema=...` into the planner's system prompt so
the model knows the shape; this helper does not inject it — that's the
caller's responsibility because system prompt construction is too
project-specific to standardize.

The default `plan_schema` argument is `Plan.model_json_schema()`, returned
as a dict so callers can serialize it themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harness.plan.plan import Plan

if TYPE_CHECKING:
    from harness.agents.definition import SubAgent
    from harness.agents.orchestrator import Runner
    from harness.prompts.messages import Message


async def derive_plan(
    planner_agent: SubAgent,
    planner_runner: Runner,
    messages: list[Message],
    *,
    plan_schema: dict[str, Any] | None = None,
) -> Plan:
    """Invoke the planner runner and parse its output as a `Plan`.

    Args:
        planner_agent: The `SubAgent` configuration for the planner.
        planner_runner: A `Runner` (any callable matching the protocol)
            that drives the planner. In tests, this is typically a
            `ReplayRunner` or a `CannedRunner`-style fake that returns
            a known JSON string.
        messages: The prompt messages to feed the planner. The planner is
            expected to look at these and emit a structured plan.
        plan_schema: The JSON schema for `Plan`. Defaults to
            `Plan.model_json_schema()`. Callers may override to feed a
            tighter / annotated schema to the planner.

    Returns:
        A parsed `Plan` instance.

    Raises:
        ValueError: if the planner's output does not parse as a valid
            `Plan` JSON document.
    """
    # Argument is documented for callers that pass it into a system
    # prompt; the helper itself doesn't inject anything into messages
    # because each project's system-prompt construction is bespoke.
    _ = plan_schema if plan_schema is not None else Plan.model_json_schema()

    response = await planner_runner(planner_agent, messages)
    raw = _extract_text(response)
    if not raw:
        raise ValueError(
            "derive_plan: planner returned no text content; expected a JSON-encoded Plan"
        )
    try:
        return Plan.model_validate_json(raw)
    except Exception as exc:
        raise ValueError(f"derive_plan: planner output did not parse as Plan JSON: {exc}") from exc


def _extract_text(message: Message) -> str:
    """Concatenate every text block in the message into a single string.

    Models that wrap their response in a single text block are the common
    case; some emit a sequence of text deltas which we join with no
    separator (the model itself decides spacing).
    """
    parts: list[str] = []
    for block in message.content:
        if block.type == "text" and block.text:
            parts.append(block.text)
    return "".join(parts)
