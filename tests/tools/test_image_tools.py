"""
Tests for image generation tools (generate_image).

The openai library is mocked via sys.modules -- no network, no API keys.
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from selectools.toolbox.image_tools import generate_image

_KEY = "sk-fake-openai-key-do-not-leak"


class _FakeOpenAIError(Exception):
    pass


def _image_response(url: Optional[str] = None, b64_json: Optional[str] = None) -> object:
    item = types.SimpleNamespace(url=url, b64_json=b64_json)
    return types.SimpleNamespace(data=[item])


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> MagicMock:
    fake = types.ModuleType("openai")
    fake.OpenAI = MagicMock(return_value=client)  # type: ignore[attr-defined]
    fake.OpenAIError = _FakeOpenAIError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)
    return fake.OpenAI  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class TestGenerateImage:
    def test_tool_metadata(self) -> None:
        assert generate_image.name == "generate_image"
        assert "image" in generate_image.description.lower()

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "openai", None)
        result = generate_image.function("a red square", api_key=_KEY)
        assert "Error" in result
        assert "openai" in result

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_openai(monkeypatch, MagicMock())
        result = generate_image.function("a red square")
        assert "Error" in result
        assert "OPENAI_API_KEY" in result

    def test_empty_prompt_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_openai(monkeypatch, MagicMock())
        result = generate_image.function("   ", api_key=_KEY)
        assert "Error" in result

    def test_b64_saved_to_output_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        payload = b"\x89PNG fake image bytes"
        client = MagicMock()
        client.images.generate.return_value = _image_response(
            b64_json=base64.b64encode(payload).decode("ascii")
        )
        _install_fake_openai(monkeypatch, client)
        destination = tmp_path / "out" / "square.png"
        result = generate_image.function("a red square", output_path=str(destination), api_key=_KEY)
        assert "saved to" in result
        assert str(destination) in result
        assert destination.read_bytes() == payload
        kwargs = client.images.generate.call_args[1]
        assert kwargs["prompt"] == "a red square"
        assert kwargs["model"] == "gpt-image-1"
        assert kwargs["size"] == "1024x1024"
        assert kwargs["n"] == 1

    def test_b64_without_output_path_saves_temp_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = b"tempfile image"
        client = MagicMock()
        client.images.generate.return_value = _image_response(
            b64_json=base64.b64encode(payload).decode("ascii")
        )
        _install_fake_openai(monkeypatch, client)
        result = generate_image.function("a blue circle", api_key=_KEY)
        assert "saved to" in result
        saved = Path(result.rsplit("saved to ", 1)[1].strip())
        try:
            assert saved.read_bytes() == payload
        finally:
            saved.unlink(missing_ok=True)

    def test_url_response_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.images.generate.return_value = _image_response(
            url="https://images.example.com/abc.png"
        )
        _install_fake_openai(monkeypatch, client)
        result = generate_image.function("a dog", model="dall-e-3", api_key=_KEY)
        assert "https://images.example.com/abc.png" in result
        assert "dall-e-3" in result

    def test_env_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.images.generate.return_value = _image_response(url="https://img.example.com/x.png")
        openai_ctor = _install_fake_openai(monkeypatch, client)
        monkeypatch.setenv("OPENAI_API_KEY", _KEY)
        result = generate_image.function("a cat")
        assert "https://img.example.com/x.png" in result
        assert openai_ctor.call_args[1]["api_key"] == _KEY

    def test_api_error_readable_no_key_leak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.images.generate.side_effect = _FakeOpenAIError("rate limited")
        _install_fake_openai(monkeypatch, client)
        result = generate_image.function("a fox", api_key=_KEY)
        assert "Error" in result
        assert "Images API call failed" in result
        assert _KEY not in result

    def test_empty_response_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.images.generate.return_value = types.SimpleNamespace(data=[])
        _install_fake_openai(monkeypatch, client)
        result = generate_image.function("a fox", api_key=_KEY)
        assert "Error" in result

    def test_no_url_or_b64_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.images.generate.return_value = _image_response()
        _install_fake_openai(monkeypatch, client)
        result = generate_image.function("a fox", api_key=_KEY)
        assert "Error" in result
        assert "neither" in result
