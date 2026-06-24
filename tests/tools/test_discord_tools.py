"""
Tests for Discord tools (discord_send_message, discord_read_channel).

The requests library is mocked via sys.modules -- no network, no tokens.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from selectools.toolbox.discord_tools import discord_read_channel, discord_send_message

_TOKEN = "fake-discord-bot-token-do-not-leak"


class _FakeRequestException(Exception):
    pass


def _fake_response(status_code: int, payload: Any) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _install_fake_requests(
    monkeypatch: pytest.MonkeyPatch,
    post: Optional[MagicMock] = None,
    get: Optional[MagicMock] = None,
) -> types.ModuleType:
    fake = types.ModuleType("requests")
    exceptions = types.ModuleType("requests.exceptions")
    exceptions.RequestException = _FakeRequestException  # type: ignore[attr-defined]
    fake.exceptions = exceptions  # type: ignore[attr-defined]
    fake.post = post or MagicMock()  # type: ignore[attr-defined]
    fake.get = get or MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", fake)
    monkeypatch.setitem(sys.modules, "requests.exceptions", exceptions)
    return fake


@pytest.fixture(autouse=True)
def _clear_discord_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)


class TestDiscordSendMessage:
    def test_tool_metadata(self) -> None:
        assert discord_send_message.name == "discord_send_message"
        assert "Discord" in discord_send_message.description

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "requests", None)
        result = discord_send_message.function("123", "hi", token=_TOKEN)
        assert "Error" in result
        assert "selectools[toolbox]" in result

    def test_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = discord_send_message.function("123", "hi")
        assert "Error" in result
        assert "DISCORD_BOT_TOKEN" in result

    def test_send_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(return_value=_fake_response(200, {"id": "999888777"}))
        _install_fake_requests(monkeypatch, post=post)
        result = discord_send_message.function("123456", "hello there", token=_TOKEN)
        assert "Message sent to channel 123456" in result
        assert "999888777" in result
        url = post.call_args[0][0]
        assert url.endswith("/channels/123456/messages")
        headers = post.call_args[1]["headers"]
        assert headers["Authorization"] == f"Bot {_TOKEN}"

    def test_env_token_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(return_value=_fake_response(200, {"id": "1"}))
        _install_fake_requests(monkeypatch, post=post)
        monkeypatch.setenv("DISCORD_BOT_TOKEN", _TOKEN)
        result = discord_send_message.function("123", "hello")
        assert "Message sent" in result

    def test_api_error_readable_no_token_leak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(
            return_value=_fake_response(403, {"message": "Missing Access", "code": 50001})
        )
        _install_fake_requests(monkeypatch, post=post)
        result = discord_send_message.function("123", "hello", token=_TOKEN)
        assert "Error" in result
        assert "403" in result
        assert "Missing Access" in result
        assert _TOKEN not in result

    def test_connection_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post = MagicMock(side_effect=_FakeRequestException("boom"))
        _install_fake_requests(monkeypatch, post=post)
        result = discord_send_message.function("123", "hello", token=_TOKEN)
        assert "Error" in result
        assert "Could not reach the Discord API" in result
        assert _TOKEN not in result

    def test_empty_channel_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = discord_send_message.function("  ", "hello", token=_TOKEN)
        assert "Error" in result

    def test_empty_text_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = discord_send_message.function("123", "  ", token=_TOKEN)
        assert "Error" in result


class TestDiscordReadChannel:
    def test_tool_metadata(self) -> None:
        assert discord_read_channel.name == "discord_read_channel"
        assert "Discord" in discord_read_channel.description

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "requests", None)
        result = discord_read_channel.function("123", token=_TOKEN)
        assert "Error" in result
        assert "selectools[toolbox]" in result

    def test_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_requests(monkeypatch)
        result = discord_read_channel.function("123")
        assert "Error" in result
        assert "DISCORD_BOT_TOKEN" in result

    def test_read_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages = [
            {
                "author": {"username": "alice"},
                "content": "newest message",
                "timestamp": "2026-06-12T10:00:00Z",
            },
            {
                "author": {"username": "bob"},
                "content": "older message",
                "timestamp": "2026-06-12T09:00:00Z",
            },
        ]
        get = MagicMock(return_value=_fake_response(200, messages))
        _install_fake_requests(monkeypatch, get=get)
        result = discord_read_channel.function("123456", limit=2, token=_TOKEN)
        assert "newest message" in result
        assert "older message" in result
        assert "alice" in result
        assert get.call_args[1]["params"] == {"limit": 2}

    def test_limit_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        get = MagicMock(return_value=_fake_response(200, []))
        _install_fake_requests(monkeypatch, get=get)
        discord_read_channel.function("123", limit=9999, token=_TOKEN)
        assert get.call_args[1]["params"] == {"limit": 100}

    def test_empty_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        get = MagicMock(return_value=_fake_response(200, []))
        _install_fake_requests(monkeypatch, get=get)
        result = discord_read_channel.function("123", token=_TOKEN)
        assert "No messages" in result

    def test_api_error_readable_no_token_leak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        get = MagicMock(
            return_value=_fake_response(404, {"message": "Unknown Channel", "code": 10003})
        )
        _install_fake_requests(monkeypatch, get=get)
        result = discord_read_channel.function("123", token=_TOKEN)
        assert "Error" in result
        assert "Unknown Channel" in result
        assert _TOKEN not in result
