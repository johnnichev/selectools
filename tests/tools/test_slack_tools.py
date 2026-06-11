"""
Tests for Slack tools (slack_send_message, slack_read_channel, slack_search_messages).

slack-sdk is mocked via sys.modules -- no network, no tokens.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from selectools.toolbox.slack_tools import (
    slack_read_channel,
    slack_search_messages,
    slack_send_message,
)

_TOKEN = "xoxb-fake-token-do-not-leak"


class _FakeSlackApiError(Exception):
    def __init__(self, message: str, response: Dict[str, Any]) -> None:
        super().__init__(message)
        self.response = response


def _install_fake_slack_sdk(monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> None:
    sdk = types.ModuleType("slack_sdk")
    errors = types.ModuleType("slack_sdk.errors")
    errors.SlackApiError = _FakeSlackApiError  # type: ignore[attr-defined]
    sdk.WebClient = MagicMock(return_value=client)  # type: ignore[attr-defined]
    sdk.errors = errors  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slack_sdk", sdk)
    monkeypatch.setitem(sys.modules, "slack_sdk.errors", errors)


@pytest.fixture(autouse=True)
def _clear_slack_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)


class TestSlackSendMessage:
    def test_tool_metadata(self) -> None:
        assert slack_send_message.name == "slack_send_message"
        assert "Slack" in slack_send_message.description

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "slack_sdk", None)
        result = slack_send_message.function("#general", "hi", token=_TOKEN)
        assert "Error" in result
        assert "selectools[toolbox]" in result

    def test_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_slack_sdk(monkeypatch, MagicMock())
        result = slack_send_message.function("#general", "hi")
        assert "Error" in result
        assert "SLACK_BOT_TOKEN" in result

    def test_send_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.chat_postMessage.return_value = {"ok": True, "ts": "1234.5678"}
        _install_fake_slack_sdk(monkeypatch, client)
        result = slack_send_message.function("#general", "hello", token=_TOKEN)
        assert "Message sent to #general" in result
        assert "1234.5678" in result
        client.chat_postMessage.assert_called_once_with(channel="#general", text="hello")

    def test_env_token_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.chat_postMessage.return_value = {"ts": "1.2"}
        _install_fake_slack_sdk(monkeypatch, client)
        monkeypatch.setenv("SLACK_BOT_TOKEN", _TOKEN)
        result = slack_send_message.function("#general", "hello")
        assert "Message sent" in result

    def test_api_error_readable_no_token_leak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.chat_postMessage.side_effect = _FakeSlackApiError(
            "channel_not_found", {"error": "channel_not_found"}
        )
        _install_fake_slack_sdk(monkeypatch, client)
        result = slack_send_message.function("#nope", "hello", token=_TOKEN)
        assert "Error" in result
        assert "channel_not_found" in result
        assert _TOKEN not in result

    def test_empty_text_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_slack_sdk(monkeypatch, MagicMock())
        result = slack_send_message.function("#general", "  ", token=_TOKEN)
        assert "Error" in result


class TestSlackReadChannel:
    def test_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_slack_sdk(monkeypatch, MagicMock())
        result = slack_read_channel.function("C0123")
        assert "Error" in result
        assert "SLACK_BOT_TOKEN" in result

    def test_read_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.conversations_history.return_value = {
            "messages": [
                {"user": "U1", "text": "newest message", "ts": "2.0"},
                {"user": "U2", "text": "older message", "ts": "1.0"},
            ]
        }
        _install_fake_slack_sdk(monkeypatch, client)
        result = slack_read_channel.function("C0123", limit=2, token=_TOKEN)
        assert "newest message" in result
        assert "older message" in result
        assert "U1" in result

    def test_empty_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.conversations_history.return_value = {"messages": []}
        _install_fake_slack_sdk(monkeypatch, client)
        result = slack_read_channel.function("C0123", token=_TOKEN)
        assert "No messages" in result

    def test_api_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.conversations_history.side_effect = _FakeSlackApiError(
            "not_in_channel", {"error": "not_in_channel"}
        )
        _install_fake_slack_sdk(monkeypatch, client)
        result = slack_read_channel.function("C0123", token=_TOKEN)
        assert "Error" in result
        assert "not_in_channel" in result
        assert _TOKEN not in result


class TestSlackSearchMessages:
    def test_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_slack_sdk(monkeypatch, MagicMock())
        result = slack_search_messages.function("deploy")
        assert "Error" in result
        assert "SLACK_BOT_TOKEN" in result

    def test_search_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.search_messages.return_value = {
            "messages": {
                "total": 1,
                "matches": [
                    {
                        "channel": {"name": "general"},
                        "username": "alice",
                        "text": "deploy went fine",
                    }
                ],
            }
        }
        _install_fake_slack_sdk(monkeypatch, client)
        result = slack_search_messages.function("deploy", token=_TOKEN)
        assert "deploy went fine" in result
        assert "#general" in result
        assert "alice" in result

    def test_no_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.search_messages.return_value = {"messages": {"total": 0, "matches": []}}
        _install_fake_slack_sdk(monkeypatch, client)
        result = slack_search_messages.function("nothing-matches", token=_TOKEN)
        assert "No messages found" in result

    def test_api_error_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = MagicMock()
        client.search_messages.side_effect = _FakeSlackApiError(
            "not_allowed_token_type", {"error": "not_allowed_token_type"}
        )
        _install_fake_slack_sdk(monkeypatch, client)
        result = slack_search_messages.function("deploy", token=_TOKEN)
        assert "Error" in result
        assert "not_allowed_token_type" in result
        assert _TOKEN not in result

    def test_empty_query_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_slack_sdk(monkeypatch, MagicMock())
        result = slack_search_messages.function("", token=_TOKEN)
        assert "Error" in result
