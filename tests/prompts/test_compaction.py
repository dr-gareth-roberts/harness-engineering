from __future__ import annotations

from harness.prompts import compact, text


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
