"""
Slack tools -- send messages, read channel history, and search messages.

Requires the optional ``slack-sdk`` library, installed via
``pip install selectools[toolbox]`` (or ``pip install slack-sdk``).
The import is lazy: the module loads fine without the dependency.

Authentication uses a Slack token passed as a parameter or via the
``SLACK_BOT_TOKEN`` environment variable. Note that ``slack_search_messages``
requires a *user* token (``xoxp-``) with the ``search:read`` scope; bot
tokens cannot call the search API. Tokens are never echoed in tool output
or error messages.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..stability import beta
from ..tools import tool

_MISSING_DEP_ERROR = (
    "Error: 'slack-sdk' library not installed. Run: pip install selectools[toolbox]"
)
_MISSING_TOKEN_ERROR = (  # nosec B105 - user-facing error message, not a credential
    "Error: No Slack token provided. Pass token or set the SLACK_BOT_TOKEN env var."
)


def _resolve_token(token: Optional[str]) -> str:
    return token or os.environ.get("SLACK_BOT_TOKEN", "")


def _slack_error(exc: Any) -> str:
    """Format a SlackApiError into a readable string without leaking the token."""
    error_code = ""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            error_code = response.get("error", "")
        except Exception:
            error_code = ""
    if error_code:
        return f"Error: Slack API returned '{error_code}'."
    return f"Error: Slack API call failed: {type(exc).__name__}"


@beta
@tool(description="Send a message to a Slack channel")
def slack_send_message(channel: str, text: str, token: Optional[str] = None) -> str:
    """
    Send a text message to a Slack channel or user.

    Args:
        channel: Channel ID (``"C0123..."``), channel name (``"#general"``),
            or user ID for a DM.
        text: Message text (Slack mrkdwn supported).
        token: Slack bot token (falls back to ``SLACK_BOT_TOKEN``). Never
            included in output or errors.

    Returns:
        Confirmation with the message timestamp, or a readable error string.
    """
    try:
        from slack_sdk import WebClient  # type: ignore[import-untyped]
        from slack_sdk.errors import SlackApiError  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    resolved = _resolve_token(token)
    if not resolved:
        return _MISSING_TOKEN_ERROR

    if not channel or not channel.strip():
        return "Error: No channel provided."
    if not text or not text.strip():
        return "Error: No message text provided."

    try:
        client = WebClient(token=resolved)
        response = client.chat_postMessage(channel=channel, text=text)
        ts = response.get("ts", "")
        return f"Message sent to {channel} (ts: {ts})"
    except SlackApiError as exc:
        return _slack_error(exc)
    except Exception as exc:
        return f"Error sending Slack message: {type(exc).__name__}"


@beta
@tool(description="Read recent messages from a Slack channel")
def slack_read_channel(channel: str, limit: int = 10, token: Optional[str] = None) -> str:
    """
    Read the most recent messages from a Slack channel.

    The bot must be a member of the channel and the token needs the
    ``channels:history`` (and/or ``groups:history``) scope.

    Args:
        channel: Channel ID (``"C0123..."``).
        limit: Maximum number of messages to return (default: 10, max: 100).
        token: Slack bot token (falls back to ``SLACK_BOT_TOKEN``). Never
            included in output or errors.

    Returns:
        Formatted list of recent messages (newest first), or a readable
        error string.
    """
    try:
        from slack_sdk import WebClient  # type: ignore[import-untyped]
        from slack_sdk.errors import SlackApiError  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    resolved = _resolve_token(token)
    if not resolved:
        return _MISSING_TOKEN_ERROR

    if not channel or not channel.strip():
        return "Error: No channel provided."

    limit = max(1, min(limit, 100))

    try:
        client = WebClient(token=resolved)
        response = client.conversations_history(channel=channel, limit=limit)
        messages = response.get("messages", [])

        if not messages:
            return f"No messages found in channel {channel}."

        lines = [f"Last {len(messages)} message(s) in {channel} (newest first):", ""]
        for msg in messages:
            user = msg.get("user", msg.get("bot_id", "unknown"))
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            lines.append(f"[{ts}] {user}: {text}")

        return "\n".join(lines)
    except SlackApiError as exc:
        return _slack_error(exc)
    except Exception as exc:
        return f"Error reading Slack channel: {type(exc).__name__}"


@beta
@tool(description="Search Slack messages across the workspace")
def slack_search_messages(query: str, count: int = 10, token: Optional[str] = None) -> str:
    """
    Search messages across the Slack workspace.

    Requires a *user* token (``xoxp-``) with the ``search:read`` scope --
    Slack does not allow bot tokens to call the search API.

    Args:
        query: Search query (supports Slack search modifiers like
            ``in:#channel`` or ``from:@user``).
        count: Maximum number of matches to return (default: 10, max: 100).
        token: Slack user token (falls back to ``SLACK_BOT_TOKEN``). Never
            included in output or errors.

    Returns:
        Formatted list of matching messages, or a readable error string.
    """
    try:
        from slack_sdk import WebClient  # type: ignore[import-untyped]
        from slack_sdk.errors import SlackApiError  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    resolved = _resolve_token(token)
    if not resolved:
        return _MISSING_TOKEN_ERROR

    if not query or not query.strip():
        return "Error: No search query provided."

    count = max(1, min(count, 100))

    try:
        client = WebClient(token=resolved)
        response = client.search_messages(query=query, count=count)
        matches = response.get("messages", {}).get("matches", [])
        total = response.get("messages", {}).get("total", 0)

        if not matches:
            return f"No messages found for: {query}"

        lines = [f"Slack messages matching '{query}' ({total} total):", ""]
        for i, match in enumerate(matches[:count], 1):
            channel_name = match.get("channel", {}).get("name", "unknown")
            username = match.get("username", "unknown")
            text = match.get("text", "")
            lines.append(f"{i}. #{channel_name} -- {username}: {text}")

        return "\n".join(lines)
    except SlackApiError as exc:
        return _slack_error(exc)
    except Exception as exc:
        return f"Error searching Slack: {type(exc).__name__}"


__stability__ = "stable"

__all__ = [
    "slack_send_message",
    "slack_read_channel",
    "slack_search_messages",
]
