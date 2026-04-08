"""End-to-end tests for web_search and scrape_url against real endpoints.

``test_search_tools.py`` mocks all HTTP. These tests hit real servers:

- ``web_search`` → DuckDuckGo HTML search (no API key)
- ``scrape_url`` → https://example.com (stable for decades)

Both are rate-limited and kept minimal (1-2 calls each) so they don't
hammer anyone. If the network is unavailable the tests skip.

Run with:

    pytest tests/tools/test_e2e_search_tools.py --run-e2e -v
"""

from __future__ import annotations

import urllib.request

import pytest

from selectools.toolbox import search_tools

pytestmark = pytest.mark.e2e


def _have_internet() -> bool:
    try:
        urllib.request.urlopen("https://example.com", timeout=5)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def internet_or_skip() -> None:
    if not _have_internet():
        pytest.skip("Network unavailable")


class TestWebSearchReal:
    def test_duckduckgo_returns_results(self, internet_or_skip: None) -> None:
        """Real DuckDuckGo HTML search returns non-empty output."""
        result = search_tools.web_search.function("python programming language")
        # Should not be an error string, and should mention something relevant
        assert result
        assert "error" not in result.lower() or "python" in result.lower()
        # Should be plaintext (not raw HTML)
        assert "<script" not in result.lower()


class TestScrapeUrlReal:
    def test_scrape_example_com(self, internet_or_skip: None) -> None:
        """Real scrape of example.com returns the canonical page text."""
        result = search_tools.scrape_url.function("https://example.com")
        assert "Example Domain" in result
        # HTML tags should be stripped
        assert "<html" not in result.lower()
        assert "<body" not in result.lower()
