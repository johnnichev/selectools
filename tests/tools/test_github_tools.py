"""
Tests for GitHub API tools (github_search_repos, github_get_file, github_list_issues).

All urllib calls are mocked to avoid hitting the real GitHub API.
"""

from __future__ import annotations

import base64
import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from selectools.toolbox import github_tools


def _mock_github_api(data: dict | list) -> MagicMock:
    """Create a mock urlopen response returning JSON data."""
    body = json.dumps(data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# =============================================================================
# github_search_repos tests
# =============================================================================


class TestGithubSearchRepos:
    """Tests for the github_search_repos tool."""

    def test_tool_has_correct_metadata(self) -> None:
        """Tool has name, description, and function attributes."""
        assert github_tools.github_search_repos.name == "github_search_repos"
        assert "GitHub" in github_tools.github_search_repos.description

    def test_empty_query_rejected(self) -> None:
        result = github_tools.github_search_repos.function("")
        assert "Error" in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_successful_search(self, mock_urlopen: MagicMock) -> None:
        """Successful search returns formatted repos."""
        mock_urlopen.return_value = _mock_github_api(
            {
                "total_count": 1,
                "items": [
                    {
                        "full_name": "owner/repo",
                        "description": "A test repo",
                        "stargazers_count": 42,
                        "language": "Python",
                        "html_url": "https://github.com/owner/repo",
                    }
                ],
            }
        )
        result = github_tools.github_search_repos.function("test")
        assert "owner/repo" in result
        assert "42" in result
        assert "Python" in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_no_results(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_github_api({"total_count": 0, "items": []})
        result = github_tools.github_search_repos.function("xyznotaword123")
        assert "No repositories" in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_rate_limit_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="",
            code=403,
            msg="Forbidden",
            hdrs=MagicMock(),
            fp=BytesIO(b""),
        )
        result = github_tools.github_search_repos.function("test")
        assert "rate limit" in result.lower()

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_max_results_clamped(self, mock_urlopen: MagicMock) -> None:
        """max_results is clamped to [1, 30]."""
        mock_urlopen.return_value = _mock_github_api({"total_count": 0, "items": []})
        github_tools.github_search_repos.function("test", max_results=0)
        github_tools.github_search_repos.function("test", max_results=100)


# =============================================================================
# github_get_file tests
# =============================================================================


class TestGithubGetFile:
    """Tests for the github_get_file tool."""

    def test_tool_has_correct_metadata(self) -> None:
        assert github_tools.github_get_file.name == "github_get_file"
        assert "file" in github_tools.github_get_file.description.lower()

    def test_invalid_repo_format(self) -> None:
        result = github_tools.github_get_file.function("no-slash", "README.md")
        assert "Error" in result
        assert "owner/repo" in result

    def test_empty_path_rejected(self) -> None:
        result = github_tools.github_get_file.function("owner/repo", "")
        assert "Error" in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_successful_file_fetch(self, mock_urlopen: MagicMock) -> None:
        """Successful file fetch decodes base64 content."""
        content_b64 = base64.b64encode(b"# Hello World\n").decode()
        mock_urlopen.return_value = _mock_github_api(
            {
                "encoding": "base64",
                "content": content_b64,
                "size": 15,
            }
        )
        result = github_tools.github_get_file.function("owner/repo", "README.md")
        assert "Hello World" in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_directory_listing(self, mock_urlopen: MagicMock) -> None:
        """Path pointing to a directory returns a directory listing."""
        mock_urlopen.return_value = _mock_github_api(
            [
                {"name": "file1.py", "type": "file"},
                {"name": "subdir", "type": "dir"},
            ]
        )
        result = github_tools.github_get_file.function("owner/repo", "src/")
        assert "file1.py" in result
        assert "subdir/" in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_file_not_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="",
            code=404,
            msg="Not Found",
            hdrs=MagicMock(),
            fp=BytesIO(b""),
        )
        result = github_tools.github_get_file.function("owner/repo", "missing.txt")
        assert "not found" in result.lower()

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_large_file_with_download_url(self, mock_urlopen: MagicMock) -> None:
        """Large files that return download_url instead of content."""
        mock_urlopen.return_value = _mock_github_api(
            {
                "encoding": "",
                "content": "",
                "size": 5000000,
                "download_url": "https://raw.githubusercontent.com/owner/repo/main/big.bin",
            }
        )
        result = github_tools.github_get_file.function("owner/repo", "big.bin")
        assert "Download URL" in result or "download" in result.lower()


# =============================================================================
# github_list_issues tests
# =============================================================================


class TestGithubListIssues:
    """Tests for the github_list_issues tool."""

    def test_tool_has_correct_metadata(self) -> None:
        assert github_tools.github_list_issues.name == "github_list_issues"
        assert "issue" in github_tools.github_list_issues.description.lower()

    def test_invalid_repo_format(self) -> None:
        result = github_tools.github_list_issues.function("noslash")
        assert "Error" in result

    def test_invalid_state_rejected(self) -> None:
        result = github_tools.github_list_issues.function("owner/repo", state="invalid")
        assert "Error" in result
        assert "open" in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_successful_issue_listing(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_github_api(
            [
                {
                    "number": 42,
                    "title": "Bug in parser",
                    "state": "open",
                    "user": {"login": "alice"},
                    "labels": [{"name": "bug"}],
                },
            ]
        )
        result = github_tools.github_list_issues.function("owner/repo")
        assert "#42" in result
        assert "Bug in parser" in result
        assert "alice" in result
        assert "bug" in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_no_issues(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_github_api([])
        result = github_tools.github_list_issues.function("owner/repo")
        assert "No open issues" in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_pull_requests_filtered_out(self, mock_urlopen: MagicMock) -> None:
        """Pull requests returned by the issues API are excluded."""
        mock_urlopen.return_value = _mock_github_api(
            [
                {
                    "number": 1,
                    "title": "A real issue",
                    "state": "open",
                    "user": {"login": "bob"},
                    "labels": [],
                },
                {
                    "number": 2,
                    "title": "A pull request",
                    "state": "open",
                    "user": {"login": "charlie"},
                    "labels": [],
                    "pull_request": {"url": "https://api.github.com/..."},
                },
            ]
        )
        result = github_tools.github_list_issues.function("owner/repo")
        assert "A real issue" in result
        assert "A pull request" not in result

    @patch("selectools.toolbox.github_tools.urllib.request.urlopen")
    def test_repo_not_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="",
            code=404,
            msg="Not Found",
            hdrs=MagicMock(),
            fp=BytesIO(b""),
        )
        result = github_tools.github_list_issues.function("owner/nonexistent")
        assert "not found" in result.lower()
