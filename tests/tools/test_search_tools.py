"""
Tests for web search and scraping tools (web_search, scrape_url).

All urllib calls are mocked to avoid hitting real websites.
"""

from __future__ import annotations

import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from selectools.toolbox import search_tools


def _make_urlopen_response(body: str, content_type: str = "text/html") -> MagicMock:
    """Create a mock urllib response with the given body."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body.encode("utf-8")
    mock_resp.headers = {"Content-Type": content_type}
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# =============================================================================
# web_search tests
# =============================================================================


class TestWebSearch:
    """Tests for the web_search tool."""

    def test_tool_has_correct_metadata(self) -> None:
        """Tool has name, description, and function attributes."""
        assert search_tools.web_search.name == "web_search"
        assert "search" in search_tools.web_search.description.lower()

    def test_stability_marker_is_beta(self) -> None:
        """web_search should carry the @beta stability marker."""
        assert getattr(search_tools.web_search, "__stability__", None) == "beta"

    def test_empty_query_rejected(self) -> None:
        """Empty query returns an error."""
        result = search_tools.web_search.function("")
        assert "Error" in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_successful_search(self, mock_urlopen: MagicMock) -> None:
        """Successful search returns formatted results."""
        html_body = """
        <div class="result">
            <a class="result__a" href="https://example.com">Example Title</a>
            <a class="result__snippet">This is a snippet about Example.</a>
        </div>
        """
        mock_urlopen.return_value = _make_urlopen_response(html_body)
        result = search_tools.web_search.function("test query")
        assert "Example Title" in result
        assert "test query" in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_no_results(self, mock_urlopen: MagicMock) -> None:
        """No results found returns informative message."""
        mock_urlopen.return_value = _make_urlopen_response("<html><body></body></html>")
        result = search_tools.web_search.function("xyznotaword123")
        assert "No results" in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen: MagicMock) -> None:
        """URLError is caught and reported."""
        mock_urlopen.side_effect = urllib.error.URLError("Network unreachable")
        result = search_tools.web_search.function("test")
        assert "Error" in result
        assert "DuckDuckGo" in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_num_results_clamped(self, mock_urlopen: MagicMock) -> None:
        """num_results is clamped to [1, 20]."""
        mock_urlopen.return_value = _make_urlopen_response("<html></html>")
        # Should not error with extreme values
        search_tools.web_search.function("test", num_results=0)
        search_tools.web_search.function("test", num_results=100)

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_uddg_redirect_url_extraction(self, mock_urlopen: MagicMock) -> None:
        """DuckDuckGo redirect URLs are properly decoded."""
        html_body = """
        <a class="result__a"
           href="/l/?uddg=https%3A%2F%2Freal.example.com%2Fpage">Real Title</a>
        <a class="result__snippet">Snippet text</a>
        """
        mock_urlopen.return_value = _make_urlopen_response(html_body)
        result = search_tools.web_search.function("test")
        assert "real.example.com" in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_generic_exception_handled(self, mock_urlopen: MagicMock) -> None:
        """Unexpected exceptions are caught gracefully."""
        mock_urlopen.side_effect = RuntimeError("something broke")
        result = search_tools.web_search.function("test")
        assert "Error" in result


# =============================================================================
# scrape_url tests
# =============================================================================


class TestScrapeUrl:
    """Tests for the scrape_url tool."""

    def test_tool_has_correct_metadata(self) -> None:
        """Tool has name, description, and function attributes."""
        assert search_tools.scrape_url.name == "scrape_url"
        assert (
            "URL" in search_tools.scrape_url.description
            or "url" in search_tools.scrape_url.description.lower()
        )

    def test_stability_marker_is_beta(self) -> None:
        """scrape_url should carry the @beta stability marker."""
        assert getattr(search_tools.scrape_url, "__stability__", None) == "beta"

    def test_empty_url_rejected(self) -> None:
        """Empty URL returns an error."""
        result = search_tools.scrape_url.function("")
        assert "Error" in result

    def test_invalid_scheme_rejected(self) -> None:
        """URL without http/https is rejected."""
        result = search_tools.scrape_url.function("ftp://example.com")
        assert "Error" in result
        assert "http" in result.lower()

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_successful_scrape(self, mock_urlopen: MagicMock) -> None:
        """Successful scrape returns text content."""
        html_body = "<html><body><p>Hello World</p></body></html>"
        mock_urlopen.return_value = _make_urlopen_response(html_body)
        result = search_tools.scrape_url.function("https://example.com")
        assert "Hello World" in result
        assert "example.com" in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_html_tags_stripped(self, mock_urlopen: MagicMock) -> None:
        """HTML tags are removed from output."""
        html_body = "<div><b>Bold</b> and <i>italic</i></div>"
        mock_urlopen.return_value = _make_urlopen_response(html_body)
        result = search_tools.scrape_url.function("https://example.com")
        assert "<div>" not in result
        assert "<b>" not in result
        assert "Bold" in result
        assert "italic" in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_selector_filters_tags(self, mock_urlopen: MagicMock) -> None:
        """Selector parameter filters to matching tags only."""
        html_body = """
        <html><body>
            <p>Paragraph one</p>
            <div>Not a paragraph</div>
            <p>Paragraph two</p>
        </body></html>
        """
        mock_urlopen.return_value = _make_urlopen_response(html_body)
        result = search_tools.scrape_url.function("https://example.com", selector="p")
        assert "Paragraph one" in result
        assert "Paragraph two" in result
        assert "Not a paragraph" not in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_selector_no_matches(self, mock_urlopen: MagicMock) -> None:
        """Selector with no matches returns informative message."""
        html_body = "<html><body><div>Content</div></body></html>"
        mock_urlopen.return_value = _make_urlopen_response(html_body)
        result = search_tools.scrape_url.function("https://example.com", selector="article")
        assert "No <article>" in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_http_error_handled(self, mock_urlopen: MagicMock) -> None:
        """HTTPError is caught and reported."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://example.com",
            code=404,
            msg="Not Found",
            hdrs=MagicMock(),
            fp=BytesIO(b""),
        )
        result = search_tools.scrape_url.function("https://example.com")
        assert "404" in result

    @patch("selectools.toolbox.search_tools.urllib.request.urlopen")
    def test_output_truncation(self, mock_urlopen: MagicMock) -> None:
        """Large pages are truncated to 10 KB."""
        big_body = "<html><body>" + "A" * 20000 + "</body></html>"
        mock_urlopen.return_value = _make_urlopen_response(big_body)
        result = search_tools.scrape_url.function("https://example.com")
        assert "truncated" in result
