"""
Tests for email tools (send_email, read_inbox).

All smtplib/imaplib calls are mocked -- no network, no credentials.
"""

from __future__ import annotations

import imaplib
from email.message import EmailMessage
from unittest.mock import MagicMock, patch

import pytest

from selectools.toolbox.email_tools import read_inbox, send_email

_SECRET = "super-secret-password"


@pytest.fixture(autouse=True)
def _clear_email_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make env-var fallbacks deterministic regardless of the host machine."""
    for var in (
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
        "SMTP_FROM",
        "IMAP_HOST",
        "IMAP_PORT",
        "IMAP_USERNAME",
        "IMAP_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)


# =============================================================================
# send_email
# =============================================================================


class TestSendEmail:
    def test_tool_metadata(self) -> None:
        assert send_email.name == "send_email"
        assert "email" in send_email.description.lower()

    def test_missing_host(self) -> None:
        result = send_email.function("a@b.com", "Hi", "Body")
        assert "Error" in result
        assert "SMTP_HOST" in result

    def test_missing_recipient(self) -> None:
        result = send_email.function("", "Hi", "Body", smtp_host="smtp.example.com")
        assert "Error" in result

    def test_missing_sender(self) -> None:
        result = send_email.function("a@b.com", "Hi", "Body", smtp_host="smtp.example.com")
        assert "Error" in result
        assert "sender" in result.lower() or "SMTP_FROM" in result

    @patch("selectools.toolbox.email_tools.smtplib.SMTP")
    def test_send_success_starttls(self, mock_smtp: MagicMock) -> None:
        server = mock_smtp.return_value.__enter__.return_value
        result = send_email.function(
            "a@b.com",
            "Hello",
            "Body text",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password=_SECRET,
        )
        assert "Email sent to a@b.com" in result
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("user@example.com", _SECRET)
        server.send_message.assert_called_once()

    @patch("selectools.toolbox.email_tools.smtplib.SMTP_SSL")
    def test_send_success_ssl_port_465(self, mock_smtp_ssl: MagicMock) -> None:
        result = send_email.function(
            "a@b.com",
            "Hello",
            "Body",
            smtp_host="smtp.example.com",
            smtp_port=465,
            username="user@example.com",
            password=_SECRET,
        )
        assert "Email sent" in result
        mock_smtp_ssl.assert_called_once()

    @patch("selectools.toolbox.email_tools.smtplib.SMTP")
    def test_env_var_fallback(self, mock_smtp: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SMTP_HOST", "smtp.env.example.com")
        monkeypatch.setenv("SMTP_USERNAME", "env-user@example.com")
        monkeypatch.setenv("SMTP_PASSWORD", _SECRET)
        result = send_email.function("a@b.com", "Hi", "Body")
        assert "Email sent" in result
        assert mock_smtp.call_args[0][0] == "smtp.env.example.com"

    @patch("selectools.toolbox.email_tools.smtplib.SMTP")
    def test_auth_failure_no_password_leak(self, mock_smtp: MagicMock) -> None:
        import smtplib as real_smtplib

        server = mock_smtp.return_value.__enter__.return_value
        server.login.side_effect = real_smtplib.SMTPAuthenticationError(535, b"bad creds")
        result = send_email.function(
            "a@b.com",
            "Hi",
            "Body",
            smtp_host="smtp.example.com",
            username="user@example.com",
            password=_SECRET,
        )
        assert "Error" in result
        assert "authentication" in result.lower()
        assert _SECRET not in result

    @patch("selectools.toolbox.email_tools.smtplib.SMTP")
    def test_connection_error_readable(self, mock_smtp: MagicMock) -> None:
        mock_smtp.side_effect = OSError("connection refused")
        result = send_email.function(
            "a@b.com", "Hi", "Body", smtp_host="smtp.example.com", from_addr="me@example.com"
        )
        assert "Error" in result
        assert "smtp.example.com" in result
        assert _SECRET not in result


# =============================================================================
# read_inbox
# =============================================================================


def _raw_email(subject: str, sender: str, body: str) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = "me@example.com"
    msg["Date"] = "Tue, 09 Jun 2026 10:00:00 +0000"
    msg.set_content(body)
    return msg.as_bytes()


def _mock_imap_client(messages: list[bytes]) -> MagicMock:
    client = MagicMock()
    client.login.return_value = ("OK", [b"Logged in"])
    client.select.return_value = ("OK", [b"3"])
    ids = " ".join(str(i + 1) for i in range(len(messages))).encode()
    client.search.return_value = ("OK", [ids])

    def fetch(msg_id: bytes, _spec: str) -> tuple:
        idx = int(msg_id) - 1
        return ("OK", [(b"%d (RFC822)" % (idx + 1), messages[idx])])

    client.fetch.side_effect = fetch
    return client


class TestReadInbox:
    def test_tool_metadata(self) -> None:
        assert read_inbox.name == "read_inbox"
        assert "imap" in read_inbox.description.lower() or "email" in read_inbox.description.lower()

    def test_missing_host(self) -> None:
        result = read_inbox.function()
        assert "Error" in result
        assert "IMAP_HOST" in result

    def test_missing_credentials(self) -> None:
        result = read_inbox.function(imap_host="imap.example.com")
        assert "Error" in result
        assert "IMAP_USERNAME" in result

    @patch("selectools.toolbox.email_tools.imaplib.IMAP4_SSL")
    def test_read_success(self, mock_imap: MagicMock) -> None:
        mock_imap.return_value = _mock_imap_client(
            [
                _raw_email("First", "alice@example.com", "Hello there, this is the first email."),
                _raw_email("Second", "bob@example.com", "Second body."),
            ]
        )
        result = read_inbox.function(
            limit=2, imap_host="imap.example.com", username="me@example.com", password=_SECRET
        )
        assert "Second" in result
        assert "alice@example.com" in result
        assert "first email" in result
        assert _SECRET not in result

    @patch("selectools.toolbox.email_tools.imaplib.IMAP4_SSL")
    def test_empty_inbox(self, mock_imap: MagicMock) -> None:
        client = MagicMock()
        client.login.return_value = ("OK", [b""])
        client.select.return_value = ("OK", [b"0"])
        client.search.return_value = ("OK", [b""])
        mock_imap.return_value = client
        result = read_inbox.function(
            imap_host="imap.example.com", username="me@example.com", password=_SECRET
        )
        assert "No messages" in result

    @patch("selectools.toolbox.email_tools.imaplib.IMAP4_SSL")
    def test_auth_failure_no_password_leak(self, mock_imap: MagicMock) -> None:
        client = MagicMock()
        client.login.side_effect = imaplib.IMAP4.error("LOGIN failed")
        mock_imap.return_value = client
        result = read_inbox.function(
            imap_host="imap.example.com", username="me@example.com", password=_SECRET
        )
        assert "Error" in result
        assert "authentication" in result.lower()
        assert _SECRET not in result

    @patch("selectools.toolbox.email_tools.imaplib.IMAP4_SSL")
    def test_connection_error_readable(self, mock_imap: MagicMock) -> None:
        mock_imap.side_effect = OSError("no route to host")
        result = read_inbox.function(
            imap_host="imap.example.com", username="me@example.com", password=_SECRET
        )
        assert "Error" in result
        assert "imap.example.com" in result
