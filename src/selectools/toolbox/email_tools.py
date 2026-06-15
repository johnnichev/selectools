"""
Email tools -- send mail via SMTP and read an inbox via IMAP.

Uses only the standard library (``smtplib``, ``imaplib``, ``email``).

Credentials are taken from explicit parameters first, then from environment
variables: ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USERNAME``, ``SMTP_PASSWORD``,
``SMTP_FROM`` for sending and ``IMAP_HOST``, ``IMAP_PORT``, ``IMAP_USERNAME``,
``IMAP_PASSWORD`` for reading. Passwords are never echoed in tool output,
error messages, or logs.
"""

from __future__ import annotations

import imaplib
import os
import smtplib
from email.header import decode_header
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from typing import Optional

from ..stability import stable
from ..tools import tool

_DEFAULT_SMTP_PORT = 587
_DEFAULT_IMAP_PORT = 993
_SNIPPET_LENGTH = 200


def _decode_mime_header(raw: str) -> str:
    """Decode a MIME-encoded header (e.g. ``=?utf-8?b?...?=``) to text."""
    parts = []
    for value, charset in decode_header(raw):
        if isinstance(value, bytes):
            parts.append(value.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(value)
    return "".join(parts)


def _extract_snippet(message: "EmailMessage") -> str:
    """Extract the first text/plain body fragment from an email message."""
    body = ""
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
                    break
    else:
        payload = message.get_payload(decode=True)
        if isinstance(payload, bytes):
            charset = message.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")

    snippet = " ".join(body.split())
    if len(snippet) > _SNIPPET_LENGTH:
        snippet = snippet[:_SNIPPET_LENGTH] + "..."
    return snippet


@stable
@tool(description="Send an email via SMTP (TLS)")
def send_email(
    to: str,
    subject: str,
    body: str,
    smtp_host: Optional[str] = None,
    smtp_port: int = 0,
    username: Optional[str] = None,
    password: Optional[str] = None,
    from_addr: Optional[str] = None,
) -> str:
    """
    Send a plain-text email through an SMTP server using STARTTLS
    (or implicit SSL when the port is 465).

    Connection settings fall back to environment variables when parameters
    are omitted: ``SMTP_HOST``, ``SMTP_PORT`` (default 587),
    ``SMTP_USERNAME``, ``SMTP_PASSWORD``, ``SMTP_FROM``.

    Args:
        to: Recipient email address (comma-separate multiple recipients).
        subject: Email subject line.
        body: Plain-text email body.
        smtp_host: SMTP server hostname (falls back to ``SMTP_HOST``).
        smtp_port: SMTP server port; 0 means ``SMTP_PORT`` env var or 587.
        username: SMTP login (falls back to ``SMTP_USERNAME``).
        password: SMTP password (falls back to ``SMTP_PASSWORD``). Never
            included in output or errors.
        from_addr: Sender address (falls back to ``SMTP_FROM``, then username).

    Returns:
        Confirmation message or a readable error string.
    """
    host = smtp_host or os.environ.get("SMTP_HOST", "")
    if not host:
        return "Error: No SMTP host provided. Pass smtp_host or set the SMTP_HOST env var."

    if not to or not to.strip():
        return "Error: No recipient provided."

    port = smtp_port or int(os.environ.get("SMTP_PORT", str(_DEFAULT_SMTP_PORT)))
    user = username or os.environ.get("SMTP_USERNAME", "")
    secret = password or os.environ.get("SMTP_PASSWORD", "")
    sender = from_addr or os.environ.get("SMTP_FROM", "") or user

    if not sender:
        return "Error: No sender address. Pass from_addr or set SMTP_FROM / SMTP_USERNAME."

    message = EmailMessage()
    message["From"] = sender
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=30) as server:
                if user and secret:
                    server.login(user, secret)
                server.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.starttls()
                if user and secret:
                    server.login(user, secret)
                server.send_message(message)
        return f"Email sent to {to} (subject: {subject})"
    except smtplib.SMTPAuthenticationError:
        return "Error: SMTP authentication failed. Check SMTP_USERNAME / SMTP_PASSWORD."
    except smtplib.SMTPRecipientsRefused:
        return f"Error: Recipient address rejected by the server: {to}"
    except smtplib.SMTPException as exc:
        return f"Error: SMTP error: {type(exc).__name__}: {exc}"
    except OSError as exc:
        return f"Error: Could not connect to SMTP server {host}:{port}: {exc}"


@stable
@tool(description="Read the latest emails from an IMAP inbox")
def read_inbox(
    limit: int = 5,
    imap_host: Optional[str] = None,
    imap_port: int = 0,
    username: Optional[str] = None,
    password: Optional[str] = None,
    folder: str = "INBOX",
) -> str:
    """
    Read the most recent messages from an IMAP mailbox (read-only).

    Returns subject, sender, date, and a short plain-text snippet for each
    of the latest ``limit`` messages. Connection settings fall back to
    environment variables: ``IMAP_HOST``, ``IMAP_PORT`` (default 993),
    ``IMAP_USERNAME``, ``IMAP_PASSWORD``.

    Args:
        limit: Maximum number of messages to return (default: 5, max: 50).
        imap_host: IMAP server hostname (falls back to ``IMAP_HOST``).
        imap_port: IMAP SSL port; 0 means ``IMAP_PORT`` env var or 993.
        username: IMAP login (falls back to ``IMAP_USERNAME``).
        password: IMAP password (falls back to ``IMAP_PASSWORD``). Never
            included in output or errors.
        folder: Mailbox folder to read (default: ``"INBOX"``).

    Returns:
        Formatted list of recent messages or a readable error string.
    """
    host = imap_host or os.environ.get("IMAP_HOST", "")
    if not host:
        return "Error: No IMAP host provided. Pass imap_host or set the IMAP_HOST env var."

    user = username or os.environ.get("IMAP_USERNAME", "")
    secret = password or os.environ.get("IMAP_PASSWORD", "")
    if not user or not secret:
        return (
            "Error: Missing IMAP credentials. Pass username/password or set "
            "IMAP_USERNAME / IMAP_PASSWORD."
        )

    port = imap_port or int(os.environ.get("IMAP_PORT", str(_DEFAULT_IMAP_PORT)))
    limit = max(1, min(limit, 50))

    try:
        client = imaplib.IMAP4_SSL(host, port)
    except OSError as exc:
        return f"Error: Could not connect to IMAP server {host}:{port}: {exc}"

    try:
        try:
            client.login(user, secret)
        except imaplib.IMAP4.error:
            return "Error: IMAP authentication failed. Check IMAP_USERNAME / IMAP_PASSWORD."

        status, _ = client.select(folder, readonly=True)
        if status != "OK":
            return f"Error: Could not open folder '{folder}'."

        status, data = client.search(None, "ALL")
        if status != "OK":
            return f"Error: Could not search folder '{folder}'."

        message_ids = data[0].split()
        if not message_ids:
            return f"No messages in {folder}."

        import email as email_lib

        lines = [f"Latest {min(limit, len(message_ids))} message(s) in {folder}:", ""]
        for i, msg_id in enumerate(reversed(message_ids[-limit:]), 1):
            status, msg_data = client.fetch(msg_id, "(RFC822)")
            if status != "OK" or not msg_data or not isinstance(msg_data[0], tuple):
                continue
            message = email_lib.message_from_bytes(msg_data[0][1])

            subject = _decode_mime_header(message.get("Subject", "(no subject)"))
            sender = _decode_mime_header(message.get("From", "(unknown sender)"))
            raw_date = message.get("Date", "")
            try:
                date_str = parsedate_to_datetime(raw_date).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError):
                date_str = raw_date or "(unknown date)"

            lines.append(f"{i}. {subject}")
            lines.append(f"   From: {sender}")
            lines.append(f"   Date: {date_str}")
            snippet = _extract_snippet(message)  # type: ignore[arg-type]
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines).strip()
    except imaplib.IMAP4.error as exc:
        return f"Error: IMAP error: {exc}"
    except OSError as exc:
        return f"Error: Connection error while reading inbox: {exc}"
    finally:
        try:
            client.logout()
        except Exception:
            pass


__stability__ = "stable"

__all__ = [
    "send_email",
    "read_inbox",
]
