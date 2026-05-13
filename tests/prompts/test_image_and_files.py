"""Tests for image content blocks (#7) and Files API attachments (#8).

The translation layer is the load-bearing part — `attach_image` /
`attach_file(file_id=...)` build the right `ContentBlock` shape, and
each runner translates it to the right vendor-specific request shape.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from harness.prompts import (
    ContentBlock,
    ImageRef,
    Message,
    attach_file,
    attach_image,
    text,
)
from harness.runner.anthropic import _translate_in as anthropic_translate
from harness.runner.openai_compat import _translate_in as oa_translate

# ---------------------------------------------------------------------------
# attach_image — input modes


def test_attach_image_url_mode_requires_media_type() -> None:
    with pytest.raises(ValueError, match="requires media_type"):
        attach_image(url="https://example.com/cat.png")


def test_attach_image_url_mode_builds_url_image_ref() -> None:
    block = attach_image(url="https://example.com/cat.png", media_type="image/png")
    assert block.type == "image"
    assert block.image is not None
    assert block.image.source == "url"
    assert block.image.media_type == "image/png"
    assert block.image.data == "https://example.com/cat.png"


def test_attach_image_path_mode_base64_encodes_and_infers_media_type(tmp_path: Path) -> None:
    p = tmp_path / "pixel.png"
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8  # not a real PNG but has the magic header
    p.write_bytes(raw)

    block = attach_image(path=p)
    assert block.type == "image"
    assert block.image is not None
    assert block.image.source == "base64"
    assert block.image.media_type == "image/png"
    # Round-trip the data: decoding back should recover the bytes.
    assert base64.standard_b64decode(block.image.data) == raw


def test_attach_image_unrecognized_extension_requires_explicit_media_type(
    tmp_path: Path,
) -> None:
    p = tmp_path / "blob.unknown"
    p.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="cannot infer media_type"):
        attach_image(path=p)


def test_attach_image_rejects_both_or_neither_path_and_url() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        attach_image()
    with pytest.raises(ValueError, match="exactly one"):
        attach_image(path="/does/not/matter", url="https://example.com")


# ---------------------------------------------------------------------------
# attach_file — file_id mode


def test_attach_file_id_mode_builds_file_block_without_path_or_text() -> None:
    block = attach_file(file_id="file_abc123")
    assert block.type == "file"
    assert block.file_id == "file_abc123"
    assert block.path is None
    assert block.text is None


def test_attach_file_rejects_both_or_neither_path_and_id(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("hi")
    with pytest.raises(ValueError, match="exactly one"):
        attach_file()
    with pytest.raises(ValueError, match="exactly one"):
        attach_file(path=p, file_id="file_x")


def test_attach_file_path_mode_still_inlines_text_under_cap(tmp_path: Path) -> None:
    """The pre-Wave-12 historical behavior: small files inline their
    text. The new file_id mode is opt-in."""
    p = tmp_path / "note.txt"
    p.write_text("hello")
    block = attach_file(p)
    assert block.type == "file"
    assert block.file_id is None
    assert block.text == "hello"


# ---------------------------------------------------------------------------
# Anthropic translation


def test_anthropic_translates_base64_image_block() -> None:
    msg = Message(
        role="user",
        content=[
            text("user", "look at this").content[0],
            ContentBlock(
                type="image",
                image=ImageRef(source="base64", media_type="image/png", data="aGVsbG8="),
            ),
        ],
    )
    api_messages, _ = anthropic_translate([msg])
    assert len(api_messages) == 1
    blocks = api_messages[0]["content"]
    text_block, image_block = blocks
    assert text_block["type"] == "text"
    assert image_block["type"] == "image"
    assert image_block["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "aGVsbG8=",
    }


def test_anthropic_translates_url_image_block() -> None:
    msg = Message(
        role="user",
        content=[
            ContentBlock(
                type="image",
                image=ImageRef(
                    source="url",
                    media_type="image/jpeg",
                    data="https://example.com/cat.jpg",
                ),
            ),
        ],
    )
    api_messages, _ = anthropic_translate([msg])
    [api_msg] = api_messages
    [block] = api_msg["content"]
    assert block["type"] == "image"
    assert block["source"]["type"] == "url"
    assert block["source"]["data"] == "https://example.com/cat.jpg"


def test_anthropic_translates_file_id_to_document_block() -> None:
    """Wave 12 #8: file_id-bearing block becomes an Anthropic Files API
    document block, NOT a text inlining."""
    msg = Message(
        role="user",
        content=[ContentBlock(type="file", file_id="file_abc123")],
    )
    api_messages, _ = anthropic_translate([msg])
    [api_msg] = api_messages
    [block] = api_msg["content"]
    assert block["type"] == "document"
    assert block["source"] == {"type": "file", "file_id": "file_abc123"}


def test_anthropic_path_based_file_still_inlines_as_text(tmp_path: Path) -> None:
    """Pre-Wave-12 path-based attach_file behavior is preserved when no
    file_id is set — small files still inline."""
    p = tmp_path / "x.txt"
    p.write_text("inline content")
    msg = Message(role="user", content=[attach_file(p)])
    api_messages, _ = anthropic_translate([msg])
    [api_msg] = api_messages
    [block] = api_msg["content"]
    assert block["type"] == "text"
    assert "inline content" in block["text"]


# ---------------------------------------------------------------------------
# OpenAI-compat translation


def test_oa_translates_base64_image_to_data_url() -> None:
    """OpenAI vision parts use data URLs for inline images."""
    msg = Message(
        role="user",
        content=[
            ContentBlock(type="text", text="see attached"),
            ContentBlock(
                type="image",
                image=ImageRef(source="base64", media_type="image/png", data="aGVsbG8="),
            ),
        ],
    )
    api_messages = oa_translate([msg])
    [api_msg] = api_messages
    # When there are image parts, content becomes a list[part], not a string.
    parts = api_msg["content"]
    assert isinstance(parts, list)
    text_part, image_part = parts
    assert text_part == {"type": "text", "text": "see attached"}
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"] == "data:image/png;base64,aGVsbG8="


def test_oa_translates_url_image() -> None:
    msg = Message(
        role="user",
        content=[
            ContentBlock(
                type="image",
                image=ImageRef(
                    source="url",
                    media_type="image/jpeg",
                    data="https://example.com/cat.jpg",
                ),
            ),
        ],
    )
    api_messages = oa_translate([msg])
    [api_msg] = api_messages
    [part] = api_msg["content"]
    assert part == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.jpg"},
    }


def test_oa_text_only_message_stays_string_shaped() -> None:
    """Back-compat: a user message without images stays as a plain
    string, not a list-of-parts."""
    msg = Message(role="user", content=[ContentBlock(type="text", text="hi")])
    api_messages = oa_translate([msg])
    [api_msg] = api_messages
    assert api_msg["content"] == "hi"


def test_oa_file_id_falls_back_to_text_placeholder() -> None:
    """OpenAI doesn't have a direct Files API equivalent; we emit a
    text placeholder so the model at least sees the reference."""
    msg = Message(
        role="user",
        content=[ContentBlock(type="file", file_id="file_xyz")],
    )
    api_messages = oa_translate([msg])
    [api_msg] = api_messages
    assert "file_id=file_xyz" in api_msg["content"]
