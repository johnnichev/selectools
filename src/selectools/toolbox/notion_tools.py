"""
Notion tools -- create pages, search the workspace, and update pages.

Uses the Notion REST API v1 via the ``requests`` library (lazy optional,
same pattern as ``web_tools``; install with ``pip install
selectools[toolbox]``).

Authentication uses an integration token passed as a parameter or via the
``NOTION_API_KEY`` environment variable. The integration must be shared
with the pages it operates on. Tokens are never echoed in tool output or
error messages.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from ..stability import beta
from ..tools import tool

_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"
_DEFAULT_TIMEOUT = 30
_MISSING_DEP_ERROR = "Error: 'requests' library not installed. Run: pip install selectools[toolbox]"
_MISSING_KEY_ERROR = (
    "Error: No Notion API key provided. Pass api_key or set the NOTION_API_KEY env var."
)


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _resolve_key(api_key: Optional[str]) -> str:
    return api_key or os.environ.get("NOTION_API_KEY", "")


def _api_error(response: Any) -> str:
    """Format a non-2xx Notion API response into a readable string."""
    try:
        payload = response.json()
        message = payload.get("message", "")
        code = payload.get("code", "")
    except Exception:
        message, code = "", ""
    detail = f" ({code}: {message})" if code or message else ""
    return f"Error: Notion API returned HTTP {response.status_code}{detail}"


@beta
@tool(description="Create a page in Notion")
def notion_create_page(
    parent_page_id: str,
    title: str,
    content: str = "",
    api_key: Optional[str] = None,
) -> str:
    """
    Create a new Notion page under an existing parent page.

    Args:
        parent_page_id: ID of the parent page (the integration must have
            access to it). Accepts dashed or undashed IDs.
        title: Title of the new page.
        content: Optional plain-text body, added as paragraph blocks
            (one per line).
        api_key: Notion integration token (falls back to ``NOTION_API_KEY``).
            Never included in output or errors.

    Returns:
        Confirmation with the new page ID and URL, or a readable error string.
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    key = _resolve_key(api_key)
    if not key:
        return _MISSING_KEY_ERROR

    if not parent_page_id or not parent_page_id.strip():
        return "Error: No parent_page_id provided."
    if not title or not title.strip():
        return "Error: No title provided."

    payload: Dict[str, Any] = {
        "parent": {"page_id": parent_page_id.strip()},
        "properties": {"title": {"title": [{"text": {"content": title}}]}},
    }
    if content:
        payload["children"] = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": line}}]},
            }
            for line in content.splitlines()
            if line.strip()
        ]

    try:
        response = requests.post(
            f"{_API_BASE}/pages",
            headers=_headers(key),
            data=json.dumps(payload),
            timeout=_DEFAULT_TIMEOUT,
        )
        if response.status_code >= 400:
            return _api_error(response)

        data = response.json()
        return f"Notion page created: '{title}' (id: {data.get('id', '?')}, url: {data.get('url', '?')})"
    except requests.exceptions.RequestException as exc:
        return f"Error: Could not reach the Notion API: {type(exc).__name__}"
    except Exception as exc:
        return f"Error creating Notion page: {type(exc).__name__}: {exc}"


@beta
@tool(description="Search pages and databases in a Notion workspace")
def notion_search(query: str, max_results: int = 5, api_key: Optional[str] = None) -> str:
    """
    Search pages and databases shared with the Notion integration.

    Args:
        query: Text to search page/database titles for.
        max_results: Maximum number of results to return (default: 5, max: 100).
        api_key: Notion integration token (falls back to ``NOTION_API_KEY``).
            Never included in output or errors.

    Returns:
        Formatted list of matching pages/databases with IDs and URLs, or a
        readable error string.
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    key = _resolve_key(api_key)
    if not key:
        return _MISSING_KEY_ERROR

    if not query or not query.strip():
        return "Error: No search query provided."

    max_results = max(1, min(max_results, 100))

    try:
        response = requests.post(
            f"{_API_BASE}/search",
            headers=_headers(key),
            data=json.dumps({"query": query, "page_size": max_results}),
            timeout=_DEFAULT_TIMEOUT,
        )
        if response.status_code >= 400:
            return _api_error(response)

        results = response.json().get("results", [])
        if not results:
            return f"No Notion pages found for: {query}"

        lines = [f"Notion results for '{query}':", ""]
        for i, item in enumerate(results[:max_results], 1):
            obj_type = item.get("object", "page")
            title = "(untitled)"
            if obj_type == "page":
                props = item.get("properties", {})
                for prop in props.values():
                    if prop.get("type") == "title":
                        rich = prop.get("title", [])
                        if rich:
                            title = rich[0].get("plain_text", title)
                        break
            else:
                rich = item.get("title", [])
                if rich:
                    title = rich[0].get("plain_text", title)
            lines.append(f"{i}. [{obj_type}] {title}")
            lines.append(f"   id: {item.get('id', '?')}")
            lines.append(f"   url: {item.get('url', '?')}")
            lines.append("")

        return "\n".join(lines).strip()
    except requests.exceptions.RequestException as exc:
        return f"Error: Could not reach the Notion API: {type(exc).__name__}"
    except Exception as exc:
        return f"Error searching Notion: {type(exc).__name__}: {exc}"


@beta
@tool(description="Update a Notion page's title or archive state")
def notion_update_page(
    page_id: str,
    title: Optional[str] = None,
    archived: Optional[bool] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Update an existing Notion page: rename it and/or archive/restore it.

    Args:
        page_id: ID of the page to update.
        title: New page title (omit to keep the current title).
        archived: ``True`` to archive (move to trash), ``False`` to restore
            (omit to leave unchanged).
        api_key: Notion integration token (falls back to ``NOTION_API_KEY``).
            Never included in output or errors.

    Returns:
        Confirmation of the update, or a readable error string.
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    key = _resolve_key(api_key)
    if not key:
        return _MISSING_KEY_ERROR

    if not page_id or not page_id.strip():
        return "Error: No page_id provided."
    if title is None and archived is None:
        return "Error: Nothing to update. Provide title and/or archived."

    payload: Dict[str, Any] = {}
    if title is not None:
        payload["properties"] = {"title": {"title": [{"text": {"content": title}}]}}
    if archived is not None:
        payload["archived"] = archived

    try:
        response = requests.patch(
            f"{_API_BASE}/pages/{page_id.strip()}",
            headers=_headers(key),
            data=json.dumps(payload),
            timeout=_DEFAULT_TIMEOUT,
        )
        if response.status_code >= 400:
            return _api_error(response)

        changes = []
        if title is not None:
            changes.append(f"title='{title}'")
        if archived is not None:
            changes.append("archived" if archived else "restored")
        return f"Notion page {page_id} updated ({', '.join(changes)})."
    except requests.exceptions.RequestException as exc:
        return f"Error: Could not reach the Notion API: {type(exc).__name__}"
    except Exception as exc:
        return f"Error updating Notion page: {type(exc).__name__}: {exc}"
