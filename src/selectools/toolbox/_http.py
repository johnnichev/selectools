"""Shared HTTP helpers for toolbox integrations."""

from __future__ import annotations

from typing import Any

DEFAULT_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (compatible; selectools/1.3.0; +https://github.com/johnnichev/selectools)"


def format_api_error(service: str, response: Any) -> str:
    """Format a non-2xx API response into a readable error string."""
    try:
        payload = response.json()
        message = payload.get("message", "")
        code = payload.get("code", "")
    except Exception:
        message, code = "", ""
    detail = f" ({code}: {message})" if code or message else ""
    return f"Error: {service} API returned HTTP {response.status_code}{detail}"
