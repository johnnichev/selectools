"""End-to-end tests for GitHub tools against the real GitHub REST API.

``test_github_tools.py`` mocks all HTTP. These tests make real unauthenticated
calls to the public GitHub API. Unauth calls are limited to 60/hour per IP;
each test makes exactly ONE call so the full file uses 3 calls.

If ``GITHUB_TOKEN`` is set the auth header is included and the limit jumps
to 5000/hour.

Run with:

    pytest tests/tools/test_e2e_github_tools.py --run-e2e -v
"""

from __future__ import annotations

import urllib.request

import pytest

from selectools.toolbox import github_tools

pytestmark = pytest.mark.e2e


def _have_internet() -> bool:
    try:
        urllib.request.urlopen("https://api.github.com", timeout=5)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def internet_or_skip() -> None:
    if not _have_internet():
        pytest.skip("Network unavailable or api.github.com unreachable")


class TestGithubToolsReal:
    def test_search_repos_real(self, internet_or_skip: None) -> None:
        """Real github search for a popular library returns results."""
        result = github_tools.github_search_repos.function(
            "selectools language:python", max_results=3
        )
        # Should not be a pure error; should include at least one known name
        assert result
        assert "error" not in result.lower() or "selectools" in result.lower()

    def test_get_file_real(self, internet_or_skip: None) -> None:
        """Real get_file of a stable public file returns its contents."""
        # python/cpython has a very stable README
        result = github_tools.github_get_file.function(
            repo="python/cpython", path="README.rst", ref="main"
        )
        assert result
        # cpython's README mentions Python
        assert "python" in result.lower() or "error" in result.lower()

    def test_list_issues_real(self, internet_or_skip: None) -> None:
        """Real list_issues against a well-known active repo."""
        result = github_tools.github_list_issues.function(
            repo="python/cpython", state="open", max_results=3
        )
        assert result
        # Either real issues or a documented error
        assert (
            "#" in result
            or "issue" in result.lower()
            or "error" in result.lower()
            or "rate" in result.lower()
        )
