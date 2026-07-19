"""
Linear tools -- create, list, and update issues via the Linear GraphQL API.

Uses the ``requests`` library (lazy optional, same pattern as ``web_tools``;
install with ``pip install selectools[toolbox]``).

Authentication uses a personal API key passed as a parameter or via the
``LINEAR_API_KEY`` environment variable (sent as-is in the ``Authorization``
header, per Linear's API convention). Keys are never echoed in tool output
or error messages.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from ..stability import stable
from ..tools import tool
from ._http import DEFAULT_TIMEOUT

_API_URL = "https://api.linear.app/graphql"
_MISSING_DEP_ERROR = "Error: 'requests' library not installed. Run: pip install selectools[toolbox]"
_MISSING_KEY_ERROR = (
    "Error: No Linear API key provided. Pass api_key or set the LINEAR_API_KEY env var."
)


def _resolve_key(api_key: Optional[str]) -> str:
    return api_key or os.environ.get("LINEAR_API_KEY", "")


def _graphql(api_key: str, query: str, variables: Dict[str, Any]) -> Any:
    """POST a GraphQL request to Linear.

    Returns:
        Tuple of (HTTP status code, parsed JSON body).

    Raises:
        requests.exceptions.RequestException: on connection failures.
    """
    import requests  # type: ignore[import-untyped]

    response = requests.post(
        _API_URL,
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        data=json.dumps({"query": query, "variables": variables}),
        timeout=DEFAULT_TIMEOUT,
    )
    return response.status_code, response.json()


def _format_graphql_errors(body: Dict[str, Any]) -> str:
    errors = body.get("errors", [])
    messages = "; ".join(err.get("message", "unknown error") for err in errors) or "unknown error"
    return f"Error: Linear API error: {messages}"


@stable
@tool(description="Create an issue in Linear")
def linear_create_issue(
    team_id: str,
    title: str,
    description: str = "",
    api_key: Optional[str] = None,
) -> str:
    """
    Create a new issue in a Linear team.

    Args:
        team_id: UUID of the Linear team (find it via ``linear_list_issues``
            output or the Linear UI's team settings).
        title: Issue title.
        description: Optional issue description (Markdown supported).
        api_key: Linear API key (falls back to ``LINEAR_API_KEY``). Never
            included in output or errors.

    Returns:
        Confirmation with the issue identifier and URL, or a readable
        error string.
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    key = _resolve_key(api_key)
    if not key:
        return _MISSING_KEY_ERROR

    if not team_id or not team_id.strip():
        return "Error: No team_id provided."
    if not title or not title.strip():
        return "Error: No title provided."

    mutation = """
    mutation IssueCreate($input: IssueCreateInput!) {
      issueCreate(input: $input) {
        success
        issue { id identifier title url }
      }
    }
    """
    variables = {"input": {"teamId": team_id.strip(), "title": title, "description": description}}

    try:
        status, body = _graphql(key, mutation, variables)
        if status >= 400 or "errors" in body:
            if status == 401 or status == 403:
                return "Error: Linear API authentication failed. Check LINEAR_API_KEY."
            if "errors" in body:
                return _format_graphql_errors(body)
            return f"Error: Linear API returned HTTP {status}"

        result = body.get("data", {}).get("issueCreate", {})
        if not result.get("success"):
            return "Error: Linear reported the issue was not created."
        issue = result.get("issue", {})
        return (
            f"Linear issue created: {issue.get('identifier', '?')} '{issue.get('title', title)}' "
            f"(url: {issue.get('url', '?')})"
        )
    except requests.exceptions.RequestException as exc:
        return f"Error: Could not reach the Linear API: {type(exc).__name__}"
    except Exception as exc:
        return f"Error creating Linear issue: {type(exc).__name__}: {exc}"


@stable
@tool(description="List recent issues in Linear")
def linear_list_issues(
    team_id: Optional[str] = None,
    limit: int = 10,
    api_key: Optional[str] = None,
) -> str:
    """
    List recent Linear issues, optionally filtered by team.

    Args:
        team_id: Optional team UUID to filter by. Default: all teams.
        limit: Maximum number of issues to return (default: 10, max: 50).
        api_key: Linear API key (falls back to ``LINEAR_API_KEY``). Never
            included in output or errors.

    Returns:
        Formatted list of issues with identifier, title, state, and assignee,
        or a readable error string.
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    key = _resolve_key(api_key)
    if not key:
        return _MISSING_KEY_ERROR

    limit = max(1, min(limit, 50))

    query = """
    query Issues($first: Int!, $filter: IssueFilter) {
      issues(first: $first, filter: $filter, orderBy: updatedAt) {
        nodes {
          id identifier title url
          state { name }
          assignee { name }
        }
      }
    }
    """
    variables: Dict[str, Any] = {"first": limit}
    if team_id:
        variables["filter"] = {"team": {"id": {"eq": team_id.strip()}}}

    try:
        status, body = _graphql(key, query, variables)
        if status >= 400 or "errors" in body:
            if status == 401 or status == 403:
                return "Error: Linear API authentication failed. Check LINEAR_API_KEY."
            if "errors" in body:
                return _format_graphql_errors(body)
            return f"Error: Linear API returned HTTP {status}"

        nodes = body.get("data", {}).get("issues", {}).get("nodes", [])
        if not nodes:
            return "No Linear issues found."

        lines = [f"Linear issues ({len(nodes)}):", ""]
        for issue in nodes:
            state = issue.get("state") or {}
            assignee = issue.get("assignee") or {}
            lines.append(
                f"  {issue.get('identifier', '?')} [{state.get('name', '?')}] "
                f"{issue.get('title', '(no title)')}"
            )
            lines.append(
                f"    assignee: {assignee.get('name', 'unassigned')} | id: {issue.get('id', '?')}"
            )
            lines.append("")

        return "\n".join(lines).strip()
    except requests.exceptions.RequestException as exc:
        return f"Error: Could not reach the Linear API: {type(exc).__name__}"
    except Exception as exc:
        return f"Error listing Linear issues: {type(exc).__name__}: {exc}"


@stable
@tool(description="Update a Linear issue's title, description, or state")
def linear_update_issue(
    issue_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    state_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Update an existing Linear issue.

    Args:
        issue_id: UUID of the issue to update (the ``id`` field from
            ``linear_list_issues``, not the ``ABC-123`` identifier).
        title: New title (omit to keep current).
        description: New description in Markdown (omit to keep current).
        state_id: UUID of the new workflow state (omit to keep current).
        api_key: Linear API key (falls back to ``LINEAR_API_KEY``). Never
            included in output or errors.

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

    if not issue_id or not issue_id.strip():
        return "Error: No issue_id provided."

    update_input: Dict[str, Any] = {}
    if title is not None:
        update_input["title"] = title
    if description is not None:
        update_input["description"] = description
    if state_id is not None:
        update_input["stateId"] = state_id
    if not update_input:
        return "Error: Nothing to update. Provide title, description, and/or state_id."

    mutation = """
    mutation IssueUpdate($id: String!, $input: IssueUpdateInput!) {
      issueUpdate(id: $id, input: $input) {
        success
        issue { identifier title url state { name } }
      }
    }
    """
    variables = {"id": issue_id.strip(), "input": update_input}

    try:
        status, body = _graphql(key, mutation, variables)
        if status >= 400 or "errors" in body:
            if status == 401 or status == 403:
                return "Error: Linear API authentication failed. Check LINEAR_API_KEY."
            if "errors" in body:
                return _format_graphql_errors(body)
            return f"Error: Linear API returned HTTP {status}"

        result = body.get("data", {}).get("issueUpdate", {})
        if not result.get("success"):
            return "Error: Linear reported the issue was not updated."
        issue = result.get("issue", {})
        state = issue.get("state") or {}
        return (
            f"Linear issue {issue.get('identifier', issue_id)} updated "
            f"(title: '{issue.get('title', '?')}', state: {state.get('name', '?')})"
        )
    except requests.exceptions.RequestException as exc:
        return f"Error: Could not reach the Linear API: {type(exc).__name__}"
    except Exception as exc:
        return f"Error updating Linear issue: {type(exc).__name__}: {exc}"


__stability__ = "stable"

__all__ = [
    "linear_create_issue",
    "linear_list_issues",
    "linear_update_issue",
]
