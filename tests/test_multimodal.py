"""Tests for multimodal message support (ContentPart, image_message, text_content)."""

from __future__ import annotations

import base64
import os
import tempfile

import pytest

from selectools.types import ContentPart, Message, Role, image_message, text_content


class TestContentPart:
    def test_text_part(self):
        p = ContentPart(type="text", text="hello")
        assert p.type == "text"
        assert p.text == "hello"

    def test_image_url_part(self):
        p = ContentPart(type="image_url", image_url="https://example.com/img.png")
        assert p.image_url == "https://example.com/img.png"

    def test_image_base64_part(self):
        p = ContentPart(type="image_base64", image_base64="abc123", media_type="image/png")
        assert p.image_base64 == "abc123"
        assert p.media_type == "image/png"


class TestImageMessage:
    def test_from_url(self):
        msg = image_message("https://example.com/photo.jpg", "What is this?")
        assert msg.role == Role.USER
        assert msg.content == "What is this?"
        assert msg.content_parts is not None
        assert len(msg.content_parts) == 2
        assert msg.content_parts[0].type == "text"
        assert msg.content_parts[1].type == "image_url"
        assert msg.content_parts[1].image_url == "https://example.com/photo.jpg"

    def test_from_file(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        msg = image_message(str(img), "Describe")
        assert msg.content_parts[1].type == "image_base64"
        assert msg.content_parts[1].media_type == "image/png"
        assert msg.content_parts[1].image_base64 is not None

    def test_from_file_jpeg(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        msg = image_message(str(img))
        assert msg.content_parts[1].media_type == "image/jpeg"

    def test_default_prompt(self):
        msg = image_message("https://example.com/img.png")
        assert msg.content == "Describe this image."

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            image_message("/nonexistent/path/img.png")


class TestTextContent:
    def test_str_content(self):
        msg = Message(role=Role.USER, content="hello")
        assert text_content(msg) == "hello"

    def test_content_parts(self):
        msg = Message(
            role=Role.USER,
            content="fallback",
            content_parts=[
                ContentPart(type="text", text="first"),
                ContentPart(type="image_url", image_url="http://x"),
                ContentPart(type="text", text="second"),
            ],
        )
        assert text_content(msg) == "first second"

    def test_empty_content(self):
        msg = Message(role=Role.USER, content="")
        assert text_content(msg) == ""

    def test_none_content_parts(self):
        msg = Message(role=Role.USER, content="text only")
        assert text_content(msg) == "text only"


class TestMessageBackwardCompat:
    def test_str_content_still_works(self):
        msg = Message(role=Role.USER, content="hello")
        assert msg.content == "hello"
        assert msg.content_parts is None

    def test_to_dict_with_content_parts(self):
        msg = Message(
            role=Role.USER,
            content="text",
            content_parts=[ContentPart(type="text", text="hi")],
        )
        d = msg.to_dict()
        assert d["content"] == "text"

    def test_image_path_still_works(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)
        msg = Message(role=Role.USER, content="describe", image_path=str(img))
        assert msg.image_base64 is not None
        assert msg.content_parts is None


class TestOpenAIFormatContent:
    def _make_provider(self):
        from unittest.mock import MagicMock, patch

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            from selectools.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider._client = MagicMock()
            provider._async_client = MagicMock()
            provider.default_model = "gpt-4o"
            return provider

    def test_content_parts_formatting(self):
        provider = self._make_provider()
        msg = Message(
            role=Role.USER,
            content="",
            content_parts=[
                ContentPart(type="text", text="What is this?"),
                ContentPart(type="image_url", image_url="https://example.com/img.png"),
            ],
        )
        result = provider._format_content(msg)
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "What is this?"}
        assert result[1]["type"] == "image_url"

    def test_base64_content_parts_formatting(self):
        provider = self._make_provider()
        msg = Message(
            role=Role.USER,
            content="",
            content_parts=[
                ContentPart(type="text", text="Analyze"),
                ContentPart(
                    type="image_base64",
                    image_base64="abc123",
                    media_type="image/png",
                ),
            ],
        )
        result = provider._format_content(msg)
        assert result[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_legacy_image_still_works(self):
        provider = self._make_provider()
        msg = Message(role=Role.USER, content="describe")
        msg.image_base64 = "test123"
        result = provider._format_content(msg)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_plain_text_still_works(self):
        provider = self._make_provider()
        msg = Message(role=Role.USER, content="hello")
        result = provider._format_content(msg)
        assert result == "hello"
