from __future__ import annotations

from typing import Any

from harness.agents import SubAgent
from harness.prompts import Message, compact, summarize_compact, text


def test_keeps_last_n_non_system() -> None:
    messages = [text("user", str(i)) for i in range(10)]
    out = compact(messages, keep_last=3)
    assert [m.content[0].text for m in out] == ["7", "8", "9"]


def test_keeps_system_messages_by_default() -> None:
    messages = [
        text("system", "you are helpful"),
        *[text("user", str(i)) for i in range(5)],
    ]
    out = compact(messages, keep_last=2)
    assert [(m.role, m.content[0].text) for m in out] == [
        ("system", "you are helpful"),
        ("user", "3"),
        ("user", "4"),
    ]


def test_drops_system_when_flag_off() -> None:
    messages = [
        text("system", "sys"),
        text("user", "u"),
    ]
    out = compact(messages, keep_last=10, keep_system=False)
    assert [m.role for m in out] == ["user"]


def test_does_not_mutate_input() -> None:
    messages = [text("user", str(i)) for i in range(5)]
    snapshot = list(messages)
    compact(messages, keep_last=2)
    assert messages == snapshot


def test_keep_last_zero_drops_non_system() -> None:
    messages = [text("system", "s"), text("user", "u")]
    out = compact(messages, keep_last=0)
    assert [m.role for m in out] == ["system"]


# ---------------------------------------------------------------------------
# summarize_compact


async def test_summarize_compact_inserts_summary_and_keeps_tail() -> None:
    messages = [
        text("system", "system rule"),
        *[text("user", f"u{i}") for i in range(10)],
    ]
    runner_calls: list[tuple[SubAgent, list[Message]]] = []

    async def fake_runner(agent: SubAgent, msgs: list[Message]) -> Message:
        runner_calls.append((agent, msgs))
        return text("assistant", "TIDY-SUMMARY")

    out = await summarize_compact(messages, fake_runner, keep_last=3)

    assert len(runner_calls) == 1
    summarized_payload = runner_calls[0][1]
    # 7 dropped + 1 instruction prompt = 8 messages handed to the runner
    assert len(summarized_payload) == 8

    # System message + summary block + last 3 non-system
    assert [m.role for m in out] == ["system", "system", "user", "user", "user"]
    assert "TIDY-SUMMARY" in (out[1].content[0].text or "")
    assert [m.content[0].text for m in out[-3:]] == ["u7", "u8", "u9"]


async def test_summarize_compact_skips_runner_when_under_keep_last() -> None:
    messages = [text("user", str(i)) for i in range(3)]
    runner_calls: list[tuple[Any, Any]] = []

    async def fake_runner(agent: SubAgent, msgs: list[Message]) -> Message:  # pragma: no cover
        runner_calls.append((agent, msgs))
        return text("assistant", "x")

    out = await summarize_compact(messages, fake_runner, keep_last=8)

    assert runner_calls == []
    assert [m.content[0].text for m in out] == ["0", "1", "2"]


async def test_summarize_compact_drops_system_when_flag_off() -> None:
    messages = [
        text("system", "sys"),
        *[text("user", str(i)) for i in range(10)],
    ]

    async def fake_runner(agent: SubAgent, msgs: list[Message]) -> Message:
        return text("assistant", "S")

    out = await summarize_compact(messages, fake_runner, keep_last=2, keep_system=False)

    # No original system messages; only the synthetic summary + tail.
    assert [m.role for m in out] == ["system", "user", "user"]
    assert [m.content[0].text for m in out[-2:]] == ["8", "9"]
