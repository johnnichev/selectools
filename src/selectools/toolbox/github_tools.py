"""
GitHub API tools for searching repositories, reading files, and listing issues.

Uses the GitHub REST API v3 with optional authentication via the
``GITHUB_TOKEN`` environment variable. All requests use ``urllib`` from
the standard library -- no external dependencies required.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from typing import Any

from ..tools import tool

_API_BASE = "https://api.github.com"
_DEFAULT_TIMEOUT = 15
_USER_AGENT = "Mozilla/5.0 (compatible; selectools/0.21; +https://github.com/johnnichev/selectools)"


def _github_request(path: str, params: dict[str, str] | None = None) -> Any:
    """Make an authenticated GET request to the GitHub API.

    Uses ``GITHUB_TOKEN`` from the environment when available.

    Raises:
        urllib.error.HTTPError: on non-2xx responses.
        urllib.error.URLError: on connection failures.
    """
    url = f"{_API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _USER_AGENT,
    }
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


@tool(description="Search GitHub repositories")
def github_search_repos(query: str, max_results: int = 5) -> str:
    """
    Search GitHub repositories using the GitHub Search API.

    Authentication via the ``GITHUB_TOKEN`` environment variable is optional
    but recommended to avoid rate limits.

    Args:
        query: Search query (e.g. ``"machine learning language:python"``).
        max_results: Maximum number of results to return (default: 5, max: 30).

    Returns:
        Formatted list of matching repositories.
    """
    if not query or not query.strip():
        return "Error: No search query provided."

    max_results = max(1, min(max_results, 30))

    try:
        data = _github_request(
            "/search/repositories",
            params={"q": query, "per_page": str(max_results)},
        )

        items = data.get("items", []) if isinstance(data, dict) else []
        total = data.get("total_count", 0) if isinstance(data, dict) else 0

        if not items:
            return f"No repositories found for: {query}"

        lines = [f"GitHub repositories for: {query} ({total} total)", ""]
        for i, repo in enumerate(items, 1):
            name = repo.get("full_name", "unknown")
            desc = repo.get("description", "") or "(no description)"
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language", "unknown") or "unknown"
            url = repo.get("html_url", "")
            lines.append(f"{i}. {name}")
            lines.append(f"   {desc}")
            lines.append(f"   Stars: {stars} | Language: {lang}")
            lines.append(f"   URL: {url}")
            lines.append("")

        return "\n".join(lines).strip()

    except urllib.error.HTTPError as e:
        if e.code == 403:
            return "Error: GitHub API rate limit exceeded. Set GITHUB_TOKEN to increase limits."
        return f"Error: GitHub API returned HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: Could not connect to GitHub API: {e.reason}"
    except Exception as e:
        return f"Error searching GitHub: {e}"


@tool(description="Get file contents from a GitHub repository")
def github_get_file(repo: str, path: str, ref: str = "main") -> str:
    """
    Retrieve the contents of a file from a GitHub repository.

    Args:
        repo: Repository in ``owner/repo`` format (e.g. ``"johnnichev/selectools"``).
        path: File path within the repository (e.g. ``"src/selectools/__init__.py"``).
        ref: Git ref (branch, tag, or commit SHA). Default: ``"main"``.

    Returns:
        File contents or an error message.
    """
    if not repo or "/" not in repo:
        return "Error: repo must be in 'owner/repo' format (e.g. 'johnnichev/selectools')."

    if not path or not path.strip():
        return "Error: No file path provided."

    try:
        encoded_path = urllib.parse.quote(path.strip(), safe="/")
        data = _github_request(
            f"/repos/{repo}/contents/{encoded_path}",
            params={"ref": ref},
        )

        if isinstance(data, list):
            # Path is a directory -- list entries
            lines = [f"Directory listing for {repo}/{path} (ref: {ref}):", ""]
            for entry in data:
                entry_type = entry.get("type", "unknown")
                entry_name = entry.get("name", "?")
                marker = "/" if entry_type == "dir" else ""
                lines.append(f"  {entry_name}{marker}")
            return "\n".join(lines)

        if not isinstance(data, dict):
            return "Error: Unexpected response from GitHub API."

        encoding = data.get("encoding", "")
        content_b64 = data.get("content", "")
        size = data.get("size", 0)

        if encoding == "base64" and content_b64:
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            lines = [
                f"File: {repo}/{path} (ref: {ref})",
                f"Size: {size} bytes",
                "",
                content,
            ]
            return "\n".join(lines)

        # Large files return a download URL instead
        download_url = data.get("download_url", "")
        if download_url:
            return f"File too large for inline content. Download URL: {download_url}"

        return "Error: Could not decode file content."

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Error: File not found: {repo}/{path} (ref: {ref})"
        if e.code == 403:
            return "Error: GitHub API rate limit exceeded. Set GITHUB_TOKEN to increase limits."
        return f"Error: GitHub API returned HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: Could not connect to GitHub API: {e.reason}"
    except Exception as e:
        return f"Error fetching file: {e}"


@tool(description="List issues in a GitHub repository")
def github_list_issues(repo: str, state: str = "open", max_results: int = 10) -> str:
    """
    List issues in a GitHub repository.

    Args:
        repo: Repository in ``owner/repo`` format (e.g. ``"johnnichev/selectools"``).
        state: Issue state filter -- ``"open"``, ``"closed"``, or ``"all"``
            (default: ``"open"``).
        max_results: Maximum number of issues to return (default: 10, max: 100).

    Returns:
        Formatted list of issues or an error message.
    """
    if not repo or "/" not in repo:
        return "Error: repo must be in 'owner/repo' format (e.g. 'johnnichev/selectools')."

    if state not in ("open", "closed", "all"):
        return "Error: state must be 'open', 'closed', or 'all'."

    max_results = max(1, min(max_results, 100))

    try:
        data = _github_request(
            f"/repos/{repo}/issues",
            params={"state": state, "per_page": str(max_results)},
        )

        if not isinstance(data, list):
            return "Error: Unexpected response from GitHub API."

        # Filter out pull requests (the issues API includes them)
        issues = [item for item in data if "pull_request" not in item]

        if not issues:
            return f"No {state} issues found in {repo}."

        lines = [f"Issues in {repo} (state: {state}):", ""]
        for issue in issues[:max_results]:
            number = issue.get("number", "?")
            title = issue.get("title", "(no title)")
            issue_state = issue.get("state", "?")
            user = issue.get("user", {}).get("login", "unknown")
            labels = ", ".join(label.get("name", "") for label in issue.get("labels", []))
            label_str = f" [{labels}]" if labels else ""
            lines.append(f"  #{number} ({issue_state}) {title}{label_str}")
            lines.append(f"    by {user}")
            lines.append("")

        return "\n".join(lines).strip()

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"Error: Repository not found: {repo}"
        if e.code == 403:
            return "Error: GitHub API rate limit exceeded. Set GITHUB_TOKEN to increase limits."
        return f"Error: GitHub API returned HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: Could not connect to GitHub API: {e.reason}"
    except Exception as e:
        return f"Error listing issues: {e}"
