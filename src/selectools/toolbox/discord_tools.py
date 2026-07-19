"""
Discord tools -- send messages and read channel history.

Uses the Discord REST API v10 via the ``requests`` library (lazy optional,
same pattern as ``notion_tools``; install with ``pip install
selectools[toolbox]``). These are plain request/response tools -- no
gateway connection or ``discord.py`` dependency is required.

Authentication uses a bot token passed as a parameter or via the
``DISCORD_BOT_TOKEN`` environment variable. The bot must be a member of
the server and have the ``Send Messages`` / ``Read Message History``
permissions in the target channel. Tokens are never echoed in tool
output or error messages.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from ..stability import beta
from ..tools import tool
from ._http import DEFAULT_TIMEOUT, format_api_error

_API_BASE = "https://discord.com/api/v10"
_MISSING_DEP_ERROR = "Error: 'requests' library not installed. Run: pip install selectools[toolbox]"
_MISSING_TOKEN_ERROR = (  # nosec B105 - user-facing error message, not a credential
    "Error: No Discord bot token provided. Pass token or set the DISCORD_BOT_TOKEN env var."
)


def _resolve_token(token: Optional[str]) -> str:
    return token or os.environ.get("DISCORD_BOT_TOKEN", "")


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }


class _NonJSONResponse(Exception):
    """A 2xx Discord response whose body was not valid JSON (e.g. a CDN/Cloudflare
    interstitial). Carries a readable message for the tool to return."""


def _json_or_raise(response: Any) -> Any:
    """Parse a 2xx response body as JSON, or raise a clear _NonJSONResponse.

    ``requests`` does not raise on a 2xx with a non-JSON body — ``response.json()``
    raises ``ValueError``/``JSONDecodeError``, which would otherwise surface as an
    opaque ``"... JSONDecodeError"``.
    """
    try:
        return response.json()
    except ValueError as exc:
        raise _NonJSONResponse(
            f"Error: Discord API returned HTTP {response.status_code} with a non-JSON body"
        ) from exc


@beta
@tool(description="Send a message to a Discord channel")
def discord_send_message(channel_id: str, text: str, token: Optional[str] = None) -> str:
    """
    Send a text message to a Discord channel.

    Args:
        channel_id: Snowflake ID of the target channel (enable Developer
            Mode in Discord and right-click the channel to copy it).
        text: Message content (Discord Markdown supported, max 2000 chars).
        token: Discord bot token (falls back to ``DISCORD_BOT_TOKEN``).
            Never included in output or errors.

    Returns:
        Confirmation with the new message ID, or a readable error string.
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    resolved = _resolve_token(token)
    if not resolved:
        return _MISSING_TOKEN_ERROR

    if not channel_id or not channel_id.strip():
        return "Error: No channel_id provided."
    if not text or not text.strip():
        return "Error: No message text provided."

    try:
        response = requests.post(
            f"{_API_BASE}/channels/{channel_id.strip()}/messages",
            headers=_headers(resolved),
            data=json.dumps({"content": text}),
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code >= 400:
            return format_api_error("Discord", response)

        data = _json_or_raise(response)
        return f"Message sent to channel {channel_id} (id: {data.get('id', '?')})"
    except _NonJSONResponse as exc:
        return str(exc)
    except requests.exceptions.RequestException as exc:
        return f"Error: Could not reach the Discord API: {type(exc).__name__}"
    except Exception as exc:
        return f"Error sending Discord message: {type(exc).__name__}"


@beta
@tool(description="Read recent messages from a Discord channel")
def discord_read_channel(channel_id: str, limit: int = 10, token: Optional[str] = None) -> str:
    """
    Read the most recent messages from a Discord channel.

    The bot needs the ``View Channel`` and ``Read Message History``
    permissions in the target channel.

    Args:
        channel_id: Snowflake ID of the channel to read.
        limit: Maximum number of messages to return (default: 10, max: 100).
        token: Discord bot token (falls back to ``DISCORD_BOT_TOKEN``).
            Never included in output or errors.

    Returns:
        Formatted list of recent messages (newest first), or a readable
        error string.
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    resolved = _resolve_token(token)
    if not resolved:
        return _MISSING_TOKEN_ERROR

    if not channel_id or not channel_id.strip():
        return "Error: No channel_id provided."

    limit = max(1, min(limit, 100))

    try:
        response = requests.get(
            f"{_API_BASE}/channels/{channel_id.strip()}/messages",
            headers=_headers(resolved),
            params={"limit": limit},
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code >= 400:
            return format_api_error("Discord", response)

        messages = _json_or_raise(response)
        if not messages:
            return f"No messages found in channel {channel_id}."

        lines = [f"Last {len(messages)} message(s) in channel {channel_id} (newest first):", ""]
        for msg in messages:
            author = msg.get("author", {}).get("username", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            lines.append(f"[{timestamp}] {author}: {content}")

        return "\n".join(lines)
    except _NonJSONResponse as exc:
        return str(exc)
    except requests.exceptions.RequestException as exc:
        return f"Error: Could not reach the Discord API: {type(exc).__name__}"
    except Exception as exc:
        return f"Error reading Discord channel: {type(exc).__name__}"


__stability__ = "beta"

__all__ = [
    "discord_send_message",
    "discord_read_channel",
]
