"""Wire compiled contract DFAs into a `HookRunner`.

The same `DFA.tick` / `DFA.finalize` used by the offline `check` is used here.
Hook handlers synthesize `Message` objects from events so `RoleIs` / `TextMatches`
predicates apply uniformly to both runtime and offline evaluation.

Three actions:
    forbid : matched message returns `HookDecision(block=True)` and surfaces
             the violation through the decision's `reason`. The same DFA also
             keeps state for any subsequent `finalize` check.
    warn   : emits a `ContractWarning` telemetry event and continues.
    require: registers a `SessionEnd` handler that raises `ContractViolation`
             if the DFA's `finalize()` reports unmet.
"""

from __future__ import annotations

from typing import Literal

from harness.contracts.contract import Contract, ContractViolation, Violation
from harness.contracts.dfa import DFA, compile_contract
from harness.hooks.events import (
    HookDecision,
    PostToolUse,
    PreToolUse,
    PromptSubmit,
    SessionEnd,
    SessionStart,
)
from harness.hooks.runner import HookRunner
from harness.prompts.messages import ContentBlock, Message
from harness.telemetry.events import TelemetryEvent
from harness.telemetry.recorder import Telemetry


class ContractWarning(TelemetryEvent):
    """Telemetry event emitted when a `warn` contract matches at runtime."""

    kind: Literal["contract.warning"] = "contract.warning"
    contract: str
    message_index: int


def _msg_from_pre_tool_use(event: PreToolUse) -> Message:
    """Synthesize an assistant tool_use message from a `PreToolUse` event."""
    return Message(
        role="assistant",
        content=[ContentBlock(type="tool_use", tool_use=event.call)],
    )


def _msg_from_post_tool_use(event: PostToolUse) -> Message:
    """Synthesize a user tool_result message from a `PostToolUse` event."""
    return Message(
        role="user",
        content=[ContentBlock(type="tool_result", tool_result=event.result)],
    )


def _msg_from_prompt_submit(event: PromptSubmit) -> Message:
    """Synthesize a user text message from a `PromptSubmit` event."""
    return Message(
        role="user",
        content=[ContentBlock(type="text", text=event.prompt)],
    )


def attach_contracts(
    hooks: HookRunner,
    contracts: list[Contract],
    *,
    telemetry: Telemetry | None = None,
) -> list[DFA]:
    """Register handlers on `hooks` so each contract's DFA observes the session.

    Returns the list of compiled DFAs (one per contract) so callers can
    inspect state for tests / debugging. The DFAs are stateful and shared
    across the registered handlers — `Earlier(...).when(...)` etc. retain
    state for the lifetime of an `Orchestrator.run`. A `SessionStart` handler
    resets each DFA so a single `HookRunner` can drive multiple runs cleanly.

    Observable events at runtime (current scope):
        * `SessionStart` — DFAs reset.
        * `PromptSubmit` — synthesized as a user-text `Message`.
        * `PreToolUse`   — synthesized as an assistant `tool_use` `Message`.
                           This is where `forbid` contracts most commonly
                           block. Returns `HookDecision(block=True)` on match.
        * `PostToolUse`  — synthesized as a user `tool_result` `Message`.
        * `SessionEnd`   — DFAs finalize; unmet `require` contracts raise
                           `ContractViolation`.

    Not yet observed at runtime: assistant *text* messages. There is no
    `PostAssistantMessage` event in `harness.hooks` yet, so a contract such
    as `Never(RoleIs("assistant") & TextMatches(...))` will fire correctly
    against a recorded `SessionRecord` via `harness.contracts.check` but
    will NOT fire live until the corresponding hook event lands. Use the
    offline `check` for assistant-text invariants today.
    """
    dfas = [compile_contract(c) for c in contracts]

    # SessionStart: reset state so DFAs don't leak across runs.
    async def on_session_start(_event: SessionStart) -> None:
        for dfa in dfas:
            dfa.reset()

    hooks.register(SessionStart, on_session_start)

    # PromptSubmit: feed user text into every DFA. None of the patterns can
    # block at this stage (it's pre-tool-use), but warn / require still update.
    async def on_prompt_submit(event: PromptSubmit) -> HookDecision | None:
        message = _msg_from_prompt_submit(event)
        for dfa in dfas:
            violation = dfa.tick(message)
            if violation is not None:
                decision = await _react_to_violation(dfa.contract, violation, telemetry)
                if decision is not None and decision.block:
                    return decision
        return None

    hooks.register(PromptSubmit, on_prompt_submit)

    # PreToolUse: feed an assistant tool_use message; this is where forbid
    # contracts most commonly fire and where blocking has effect.
    async def on_pre_tool_use(event: PreToolUse) -> HookDecision | None:
        message = _msg_from_pre_tool_use(event)
        for dfa in dfas:
            violation = dfa.tick(message)
            if violation is not None:
                decision = await _react_to_violation(dfa.contract, violation, telemetry)
                if decision is not None and decision.block:
                    return decision
        return None

    hooks.register(PreToolUse, on_pre_tool_use)

    # PostToolUse: feed the tool_result so contracts that look at results
    # (Eventually(...) on a result-shaped predicate, etc.) can observe them.
    async def on_post_tool_use(event: PostToolUse) -> None:
        message = _msg_from_post_tool_use(event)
        for dfa in dfas:
            violation = dfa.tick(message)
            if violation is not None:
                # PostToolUse can't block — the call already ran. But we still
                # surface warn / forbid violations through telemetry.
                await _react_to_violation(dfa.contract, violation, telemetry)

    hooks.register(PostToolUse, on_post_tool_use)

    # SessionEnd: finalize() — `require` contracts raise here if unmet.
    async def on_session_end(_event: SessionEnd) -> None:
        for dfa in dfas:
            violation = dfa.finalize()
            if violation is None:
                continue
            if dfa.contract.action == "require":
                raise ContractViolation(violation)
            if dfa.contract.action == "warn" and telemetry is not None:
                await telemetry.emit(
                    ContractWarning(
                        contract=violation.contract,
                        message_index=violation.message_index,
                    )
                )

    hooks.register(SessionEnd, on_session_end)

    return dfas


async def _react_to_violation(
    contract: Contract,
    violation: Violation,
    telemetry: Telemetry | None,
) -> HookDecision | None:
    """Translate a DFA-emitted Violation into the right runtime side-effect."""
    if contract.action == "forbid":
        return HookDecision(
            block=True,
            reason=f"contract {violation.contract!r} forbids this action",
        )
    if contract.action == "warn":
        if telemetry is not None:
            await telemetry.emit(
                ContractWarning(
                    contract=violation.contract,
                    message_index=violation.message_index,
                )
            )
        return None
    # `require` mid-stream — the inner pattern (e.g. Always) hard-failed.
    # Treat as a hard violation: raise immediately so callers find out at the
    # earliest possible point. This still goes through ContractViolation so
    # tests can introspect the carrier `Violation`.
    raise ContractViolation(violation)
