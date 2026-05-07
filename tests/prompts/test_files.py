from __future__ import annotations

from pathlib import Path

import pytest

from harness.prompts.files import TRUNCATION_MARKER, attach_file


def test_reads_file_into_block(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello world")
    block = attach_file(p)
    assert block.type == "file"
    assert block.path == str(p)
    assert block.text == "hello world"


def test_truncates_at_max_bytes(tmp_path: Path) -> None:
    p = tmp_path / "big.txt"
    p.write_text("a" * 1000)
    block = attach_file(p, max_bytes=100)
    assert block.text is not None
    assert block.text.startswith("a" * 100)
    assert block.text.endswith(TRUNCATION_MARKER)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        attach_file(tmp_path / "nope.txt")


def test_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        attach_file(tmp_path)
