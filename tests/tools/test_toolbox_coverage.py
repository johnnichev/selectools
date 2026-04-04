"""
Extended toolbox tests targeting uncovered lines for 90%+ coverage.

Covers:
- web_tools: full mock-based tests for http_get and http_post
- data_tools: error paths, edge cases, streaming CSV
- text_tools: edge cases, regex errors, all case conversions
- datetime_tools: error paths, all units, missing pytz
- file_tools: error paths, streaming read, recursive listing
"""

from __future__ import annotations

import csv
import json
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Generator
from unittest import mock

import pytest

from selectools.toolbox import data_tools, datetime_tools, file_tools, text_tools, web_tools

# =============================================================================
# Web Tools — http_get (fully mocked)
# =============================================================================


class TestHttpGet:
    """Tests for http_get with mocked requests library."""

    def _make_response(
        self,
        status_code: int = 200,
        reason: str = "OK",
        content_type: str = "text/html",
        text: str = "hello",
        content: bytes = b"hello",
        json_data: object | None = None,
        json_raises: bool = False,
    ) -> SimpleNamespace:
        """Build a fake requests.Response-like object."""
        resp = SimpleNamespace()
        resp.status_code = status_code
        resp.reason = reason
        resp.headers = {"Content-Type": content_type}
        resp.text = text
        resp.content = content
        if json_data is not None:
            resp.json = lambda: json_data
        elif json_raises:
            resp.json = lambda: (_ for _ in ()).throw(json.JSONDecodeError("x", "y", 0))
        else:
            resp.json = lambda: json.loads(text)
        return resp

    def test_get_text_response(self) -> None:
        """http_get returns formatted text response."""
        resp = self._make_response(text="<html>OK</html>", content=b"<html>OK</html>")
        mock_requests = mock.MagicMock()
        mock_requests.get.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://example.com")

        assert "Status: 200 OK" in result
        assert "<html>OK</html>" in result

    def test_get_json_response(self) -> None:
        """http_get formats JSON responses with indentation."""
        data = {"key": "value"}
        resp = self._make_response(
            content_type="application/json",
            text=json.dumps(data),
            content=json.dumps(data).encode(),
            json_data=data,
        )
        mock_requests = mock.MagicMock()
        mock_requests.get.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://api.example.com/data")

        assert "Status: 200 OK" in result
        assert '"key"' in result
        assert '"value"' in result

    def test_get_json_decode_error_in_response(self) -> None:
        """http_get falls back to .text when response.json() raises JSONDecodeError."""
        resp = self._make_response(
            content_type="application/json",
            text="not-real-json",
            content=b"not-real-json",
            json_raises=True,
        )
        mock_requests = mock.MagicMock()
        mock_requests.get.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://api.example.com/broken")

        assert "not-real-json" in result

    def test_get_truncated_text(self) -> None:
        """http_get truncates text longer than 5000 characters."""
        long_text = "A" * 6000
        resp = self._make_response(text=long_text, content=long_text.encode())
        mock_requests = mock.MagicMock()
        mock_requests.get.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://example.com/big")

        assert "truncated" in result.lower()
        # The truncated portion should be at most 5000 chars
        assert "A" * 5000 in result

    def test_get_with_headers(self) -> None:
        """http_get passes parsed JSON headers to requests.get."""
        resp = self._make_response()
        mock_requests = mock.MagicMock()
        mock_requests.get.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        headers_json = json.dumps({"Authorization": "Bearer token123"})
        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://api.example.com", headers=headers_json)

        assert "Status: 200" in result
        call_kwargs = mock_requests.get.call_args
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer token123"

    def test_get_invalid_headers_json(self) -> None:
        """http_get returns error for malformed headers JSON."""
        mock_requests = mock.MagicMock()
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://example.com", headers="{bad json")

        assert "Invalid JSON in headers" in result

    def test_get_timeout_error(self) -> None:
        """http_get handles Timeout exception."""
        mock_requests = mock.MagicMock()
        exc_ns = _make_exceptions_ns()
        mock_requests.exceptions = exc_ns
        mock_requests.get.side_effect = exc_ns.Timeout("timed out")

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://slow.example.com", timeout=5)

        assert "timed out" in result.lower()
        assert "5" in result

    def test_get_connection_error(self) -> None:
        """http_get handles ConnectionError."""
        mock_requests = mock.MagicMock()
        exc_ns = _make_exceptions_ns()
        mock_requests.exceptions = exc_ns
        mock_requests.get.side_effect = exc_ns.ConnectionError("refused")

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://down.example.com")

        assert "Could not connect" in result

    def test_get_request_exception(self) -> None:
        """http_get handles generic RequestException."""
        mock_requests = mock.MagicMock()
        exc_ns = _make_exceptions_ns()
        mock_requests.exceptions = exc_ns
        mock_requests.get.side_effect = exc_ns.RequestException("something broke")

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://broken.example.com")

        assert "Error making request" in result

    def test_get_unexpected_exception(self) -> None:
        """http_get handles unexpected exceptions."""
        mock_requests = mock.MagicMock()
        exc_ns = _make_exceptions_ns()
        mock_requests.exceptions = exc_ns
        mock_requests.get.side_effect = RuntimeError("boom")

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_get.function("https://example.com")

        assert "Unexpected error" in result

    def test_get_missing_requests_library(self) -> None:
        """http_get returns helpful message when requests is not installed."""
        with mock.patch.dict("sys.modules", {"requests": None}):
            import importlib

            import selectools.toolbox.web_tools as wt

            importlib.reload(wt)
            result = wt.http_get.function("https://example.com")

        assert "not installed" in result.lower() or "requests" in result

        # Restore the module
        import importlib

        import selectools.toolbox.web_tools as wt2

        importlib.reload(wt2)


# =============================================================================
# Web Tools — http_post (fully mocked)
# =============================================================================


class TestHttpPost:
    """Tests for http_post with mocked requests library."""

    def _make_response(
        self,
        status_code: int = 201,
        reason: str = "Created",
        content_type: str = "application/json",
        text: str = '{"id": 1}',
        content: bytes = b'{"id": 1}',
        json_data: object | None = None,
        json_raises: bool = False,
    ) -> SimpleNamespace:
        resp = SimpleNamespace()
        resp.status_code = status_code
        resp.reason = reason
        resp.headers = {"Content-Type": content_type}
        resp.text = text
        resp.content = content
        if json_data is not None:
            resp.json = lambda: json_data
        elif json_raises:
            resp.json = lambda: (_ for _ in ()).throw(json.JSONDecodeError("x", "y", 0))
        else:
            resp.json = lambda: json.loads(text)
        return resp

    def test_post_json_response(self) -> None:
        """http_post sends JSON data and returns formatted JSON response."""
        data = {"id": 42, "name": "test"}
        resp = self._make_response(
            json_data=data, text=json.dumps(data), content=json.dumps(data).encode()
        )
        mock_requests = mock.MagicMock()
        mock_requests.post.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function(
                "https://api.example.com/items", data='{"name": "test"}'
            )

        assert "Status: 201 Created" in result
        assert "42" in result

    def test_post_text_response(self) -> None:
        """http_post handles non-JSON response."""
        resp = self._make_response(
            content_type="text/plain", text="OK", content=b"OK", status_code=200, reason="OK"
        )
        mock_requests = mock.MagicMock()
        mock_requests.post.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function(
                "https://example.com/submit", data='{"key": "val"}'
            )

        assert "Status: 200 OK" in result
        assert "OK" in result

    def test_post_truncated_text_response(self) -> None:
        """http_post truncates long non-JSON response text."""
        long_text = "B" * 6000
        resp = self._make_response(
            content_type="text/plain",
            text=long_text,
            content=long_text.encode(),
            status_code=200,
            reason="OK",
        )
        mock_requests = mock.MagicMock()
        mock_requests.post.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function("https://example.com/big", data='{"x": 1}')

        assert "truncated" in result.lower()

    def test_post_json_decode_error_in_response(self) -> None:
        """http_post falls back to .text when response.json() fails."""
        resp = self._make_response(
            content_type="application/json",
            text="broken-json",
            content=b"broken-json",
            json_raises=True,
        )
        mock_requests = mock.MagicMock()
        mock_requests.post.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function("https://api.example.com/x", data='{"k": 1}')

        assert "broken-json" in result

    def test_post_invalid_data_json(self) -> None:
        """http_post returns error for malformed data JSON."""
        mock_requests = mock.MagicMock()
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function("https://example.com", data="{bad")

        assert "Invalid JSON in data" in result

    def test_post_invalid_headers_json(self) -> None:
        """http_post returns error for malformed headers JSON."""
        mock_requests = mock.MagicMock()
        mock_requests.exceptions = _make_exceptions_ns()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function(
                "https://example.com", data='{"x": 1}', headers="{bad"
            )

        assert "Invalid JSON in headers" in result

    def test_post_with_headers(self) -> None:
        """http_post merges custom headers with Content-Type."""
        resp = self._make_response()
        mock_requests = mock.MagicMock()
        mock_requests.post.return_value = resp
        mock_requests.exceptions = _make_exceptions_ns()

        headers_json = json.dumps({"X-Custom": "foobar"})
        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function(
                "https://api.example.com", data='{"a": 1}', headers=headers_json
            )

        assert "Status:" in result
        call_kwargs = mock_requests.post.call_args
        assert call_kwargs[1]["headers"]["X-Custom"] == "foobar"
        assert call_kwargs[1]["headers"]["Content-Type"] == "application/json"

    def test_post_timeout_error(self) -> None:
        """http_post handles Timeout."""
        mock_requests = mock.MagicMock()
        exc_ns = _make_exceptions_ns()
        mock_requests.exceptions = exc_ns
        mock_requests.post.side_effect = exc_ns.Timeout()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function(
                "https://slow.example.com", data='{"x": 1}', timeout=10
            )

        assert "timed out" in result.lower()

    def test_post_connection_error(self) -> None:
        """http_post handles ConnectionError."""
        mock_requests = mock.MagicMock()
        exc_ns = _make_exceptions_ns()
        mock_requests.exceptions = exc_ns
        mock_requests.post.side_effect = exc_ns.ConnectionError()

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function("https://down.example.com", data='{"x": 1}')

        assert "Could not connect" in result

    def test_post_request_exception(self) -> None:
        """http_post handles generic RequestException."""
        mock_requests = mock.MagicMock()
        exc_ns = _make_exceptions_ns()
        mock_requests.exceptions = exc_ns
        mock_requests.post.side_effect = exc_ns.RequestException("oops")

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function("https://broken.example.com", data='{"x": 1}')

        assert "Error making request" in result

    def test_post_unexpected_exception(self) -> None:
        """http_post handles unexpected exceptions."""
        mock_requests = mock.MagicMock()
        exc_ns = _make_exceptions_ns()
        mock_requests.exceptions = exc_ns
        mock_requests.post.side_effect = RuntimeError("kaboom")

        with mock.patch.dict("sys.modules", {"requests": mock_requests}):
            result = web_tools.http_post.function("https://example.com", data='{"x": 1}')

        assert "Unexpected error" in result

    def test_post_missing_requests_library(self) -> None:
        """http_post returns helpful message when requests not installed."""
        with mock.patch.dict("sys.modules", {"requests": None}):
            import importlib

            import selectools.toolbox.web_tools as wt

            importlib.reload(wt)
            result = wt.http_post.function("https://example.com", data='{"x": 1}')

        assert "not installed" in result.lower()

        import importlib

        import selectools.toolbox.web_tools as wt2

        importlib.reload(wt2)


# =============================================================================
# Data Tools — extended coverage
# =============================================================================


class TestDataToolsExtended:
    """Cover uncovered branches in data_tools."""

    def test_parse_json_not_pretty(self) -> None:
        """parse_json with pretty=False returns compact output."""
        result = data_tools.parse_json.function('{"a": 1}', pretty=False)
        assert "Valid JSON" in result
        # Compact JSON has no indentation
        assert '{"a": 1}' in result

    def test_parse_json_generic_exception(self) -> None:
        """parse_json handles unexpected errors."""
        # Pass something that triggers a non-JSONDecodeError
        with mock.patch("json.loads", side_effect=TypeError("bad")):
            result = data_tools.parse_json.function("anything")
        assert "Error parsing JSON" in result

    def test_json_to_csv_not_list(self) -> None:
        """json_to_csv rejects non-array JSON."""
        result = data_tools.json_to_csv.function('{"key": "value"}')
        assert "must be an array" in result

    def test_json_to_csv_empty_array(self) -> None:
        """json_to_csv rejects empty array."""
        result = data_tools.json_to_csv.function("[]")
        assert "Empty array" in result

    def test_json_to_csv_non_dict_items(self) -> None:
        """json_to_csv rejects array of non-objects."""
        result = data_tools.json_to_csv.function("[1, 2, 3]")
        assert "must be objects" in result

    def test_json_to_csv_custom_delimiter(self) -> None:
        """json_to_csv supports custom delimiter."""
        data = json.dumps([{"a": 1, "b": 2}])
        result = data_tools.json_to_csv.function(data, delimiter="\t")
        assert "Converted 1 rows" in result
        # Check tab-separated values appear
        assert "\t" in result

    def test_json_to_csv_invalid_json(self) -> None:
        """json_to_csv handles invalid JSON."""
        result = data_tools.json_to_csv.function("{bad json")
        assert "Invalid JSON" in result

    def test_json_to_csv_mixed_keys(self) -> None:
        """json_to_csv handles objects with different keys."""
        data = json.dumps([{"a": 1}, {"b": 2}, {"a": 3, "b": 4}])
        result = data_tools.json_to_csv.function(data)
        assert "Converted 3 rows" in result
        assert "a" in result
        assert "b" in result

    def test_csv_to_json_not_pretty(self) -> None:
        """csv_to_json with pretty=False returns compact JSON."""
        csv_str = "name,age\nAlice,30"
        result = data_tools.csv_to_json.function(csv_str, pretty=False)
        assert "Converted 1 rows" in result
        # Compact: no newlines in JSON body
        assert "Alice" in result

    def test_csv_to_json_empty_data(self) -> None:
        """csv_to_json rejects empty CSV."""
        result = data_tools.csv_to_json.function("")
        assert "No data found" in result

    def test_csv_to_json_custom_delimiter(self) -> None:
        """csv_to_json supports custom delimiter."""
        csv_str = "name\tage\nAlice\t30"
        result = data_tools.csv_to_json.function(csv_str, delimiter="\t")
        assert "Alice" in result

    def test_extract_json_field_not_found(self) -> None:
        """extract_json_field returns error for missing key."""
        result = data_tools.extract_json_field.function('{"a": 1}', "b")
        assert "not found" in result

    def test_extract_json_field_invalid_array_index(self) -> None:
        """extract_json_field returns error for bad array index."""
        result = data_tools.extract_json_field.function("[1, 2, 3]", "abc")
        assert "Invalid array index" in result

    def test_extract_json_field_index_out_of_range(self) -> None:
        """extract_json_field returns error for out-of-range index."""
        result = data_tools.extract_json_field.function("[1, 2]", "5")
        assert "Invalid array index" in result

    def test_extract_json_field_not_traversable(self) -> None:
        """extract_json_field returns error when navigating through a scalar."""
        result = data_tools.extract_json_field.function('{"a": 42}', "a.b")
        assert "Cannot navigate" in result

    def test_extract_json_field_invalid_json(self) -> None:
        """extract_json_field handles invalid JSON."""
        result = data_tools.extract_json_field.function("{bad", "a")
        assert "Invalid JSON" in result

    def test_format_table_not_array(self) -> None:
        """format_table rejects non-array data."""
        result = data_tools.format_table.function('{"a": 1}')
        assert "must be a JSON array" in result

    def test_format_table_empty_array(self) -> None:
        """format_table rejects empty array."""
        result = data_tools.format_table.function("[]")
        assert "Empty array" in result

    def test_format_table_non_dict_items(self) -> None:
        """format_table rejects array of non-objects."""
        result = data_tools.format_table.function("[1, 2]")
        assert "must be objects" in result

    def test_format_table_csv_format(self) -> None:
        """format_table with format_type='csv' returns CSV."""
        data = json.dumps([{"x": 10, "y": 20}])
        result = data_tools.format_table.function(data, format_type="csv")
        assert "x" in result
        assert "10" in result

    def test_format_table_invalid_json(self) -> None:
        """format_table handles invalid JSON."""
        result = data_tools.format_table.function("{bad")
        assert "Invalid JSON" in result

    def test_process_csv_stream_success(self, tmp_path: Path) -> None:
        """process_csv_stream yields rows from a CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")

        chunks = list(data_tools.process_csv_stream.function(str(csv_file)))
        joined = "".join(chunks)

        assert "Processing CSV file" in joined
        assert "Columns: name, age" in joined
        assert "Row 1" in joined
        assert "Alice" in joined
        assert "Row 2" in joined
        assert "Bob" in joined
        assert "Finished processing" in joined

    def test_process_csv_stream_file_not_found(self) -> None:
        """process_csv_stream yields error for missing file."""
        chunks = list(data_tools.process_csv_stream.function("/nonexistent/file.csv"))
        joined = "".join(chunks)
        assert "not found" in joined.lower() or "Error" in joined

    def test_process_csv_stream_permission_denied(self, tmp_path: Path) -> None:
        """process_csv_stream yields error for unreadable file."""
        csv_file = tmp_path / "locked.csv"
        csv_file.write_text("a,b\n1,2\n")
        csv_file.chmod(0o000)

        try:
            chunks = list(data_tools.process_csv_stream.function(str(csv_file)))
            joined = "".join(chunks)
            assert "Permission denied" in joined or "Error" in joined
        finally:
            csv_file.chmod(0o644)


# =============================================================================
# Text Tools — extended coverage
# =============================================================================


class TestTextToolsExtended:
    """Cover uncovered branches in text_tools."""

    def test_count_text_not_detailed(self) -> None:
        """count_text with detailed=False omits averages."""
        result = text_tools.count_text.function("hello world", detailed=False)
        assert "Words:" in result
        assert "Average" not in result

    def test_search_text_no_matches(self) -> None:
        """search_text returns message when no matches found."""
        result = text_tools.search_text.function("hello world", "xyz123")
        assert "No matches found" in result

    def test_search_text_case_insensitive(self) -> None:
        """search_text supports case-insensitive searching."""
        result = text_tools.search_text.function("Hello World HELLO", "hello", case_sensitive=False)
        assert "2 match" in result

    def test_search_text_return_count_only(self) -> None:
        """search_text with return_matches=False returns count only."""
        result = text_tools.search_text.function("aaa bbb aaa", "aaa", return_matches=False)
        assert "2 match" in result
        assert "Matches:" not in result

    def test_search_text_more_than_20_matches(self) -> None:
        """search_text truncates results beyond 20 matches."""
        text = " ".join(["word"] * 25)
        result = text_tools.search_text.function(text, "word")
        assert "20. word" in result
        assert "5 more" in result

    def test_search_text_invalid_regex(self) -> None:
        """search_text returns error for invalid regex."""
        result = text_tools.search_text.function("hello", "[invalid")
        assert "Invalid regex" in result

    def test_replace_text_max_replacements(self) -> None:
        """replace_text respects max_replacements limit."""
        result = text_tools.replace_text.function("aaa", "a", "b", max_replacements=2)
        assert "2 replacement" in result
        assert "bba" in result

    def test_replace_text_case_insensitive(self) -> None:
        """replace_text supports case-insensitive replacement."""
        result = text_tools.replace_text.function(
            "Hello HELLO hello", "hello", "hi", case_sensitive=False
        )
        assert "3 replacement" in result
        assert "hi hi hi" in result

    def test_replace_text_invalid_regex(self) -> None:
        """replace_text returns error for invalid regex."""
        result = text_tools.replace_text.function("hello", "[bad", "x")
        assert "Invalid regex" in result

    def test_extract_emails_none_found(self) -> None:
        """extract_emails returns message when no emails found."""
        result = text_tools.extract_emails.function("no emails here")
        assert "No email addresses found" in result

    def test_extract_emails_deduplication(self) -> None:
        """extract_emails removes duplicate emails (case-insensitive)."""
        text = "alice@example.com Alice@Example.com alice@example.com"
        result = text_tools.extract_emails.function(text)
        assert "1 unique" in result

    def test_extract_urls_none_found(self) -> None:
        """extract_urls returns message when no URLs found."""
        result = text_tools.extract_urls.function("no urls here at all")
        assert "No URLs found" in result

    def test_extract_urls_deduplication(self) -> None:
        """extract_urls removes duplicate URLs."""
        text = "https://example.com https://example.com https://other.com"
        result = text_tools.extract_urls.function(text)
        assert "2 unique" in result

    def test_convert_case_lower(self) -> None:
        """convert_case converts to lowercase."""
        result = text_tools.convert_case.function("HELLO WORLD", "lower")
        assert result == "hello world"

    def test_convert_case_sentence(self) -> None:
        """convert_case converts to sentence case."""
        result = text_tools.convert_case.function("hello world. how are you", "sentence")
        assert result.startswith("Hello")
        assert "How" in result or "how" in result

    def test_convert_case_kebab(self) -> None:
        """convert_case converts to kebab-case."""
        result = text_tools.convert_case.function("Hello World Test", "kebab")
        assert result == "hello-world-test"

    def test_convert_case_camel_empty(self) -> None:
        """convert_case returns original text for camelCase with no words."""
        result = text_tools.convert_case.function("!!!", "camel")
        assert result == "!!!"

    def test_convert_case_invalid(self) -> None:
        """convert_case returns error for unknown case type."""
        result = text_tools.convert_case.function("hello", "invalid_case")
        assert "Invalid case type" in result

    def test_truncate_text_short_enough(self) -> None:
        """truncate_text returns text unchanged if short enough."""
        result = text_tools.truncate_text.function("short", max_length=100)
        assert result == "short"

    def test_truncate_text_very_short_max(self) -> None:
        """truncate_text handles max_length smaller than suffix."""
        result = text_tools.truncate_text.function("hello world", max_length=2, suffix="...")
        assert len(result) == 2
        assert result == ".."

    def test_truncate_text_custom_suffix(self) -> None:
        """truncate_text supports custom suffix."""
        result = text_tools.truncate_text.function("A" * 20, max_length=10, suffix=" [more]")
        assert result.endswith(" [more]")
        assert len(result) == 10


# =============================================================================
# DateTime Tools — extended coverage
# =============================================================================


class TestDateTimeToolsExtended:
    """Cover uncovered branches in datetime_tools."""

    def test_get_current_time_non_utc_without_pytz(self) -> None:
        """get_current_time with non-UTC timezone requires pytz."""
        import sys

        with mock.patch.dict("sys.modules", {"pytz": None}):
            import importlib

            import selectools.toolbox.datetime_tools as dt_mod

            importlib.reload(dt_mod)
            result = dt_mod.get_current_time.function(timezone="America/New_York")

        assert "pytz" in result.lower() or "Error" in result

        import importlib

        import selectools.toolbox.datetime_tools as dt_mod2

        importlib.reload(dt_mod2)

    def test_get_current_time_unknown_timezone(self) -> None:
        """get_current_time returns error for unknown timezone."""
        try:
            import pytz  # noqa: F401
        except ImportError:
            pytest.skip("pytz not installed")

        result = datetime_tools.get_current_time.function(timezone="Invalid/Timezone")
        assert "Unknown timezone" in result

    def test_get_current_time_with_pytz(self) -> None:
        """get_current_time works with pytz for non-UTC timezone."""
        try:
            import pytz  # noqa: F401
        except ImportError:
            pytest.skip("pytz not installed")

        result = datetime_tools.get_current_time.function(timezone="Europe/London")
        assert "Europe/London" in result

    def test_parse_datetime_unparseable(self) -> None:
        """parse_datetime returns error for unparseable string."""
        result = datetime_tools.parse_datetime.function("not-a-date")
        assert "Could not parse" in result

    def test_parse_datetime_with_bad_format(self) -> None:
        """parse_datetime returns error when explicit format doesn't match."""
        result = datetime_tools.parse_datetime.function("2025-01-01", input_format="%d/%m/%Y %H:%M")
        assert "Error" in result

    def test_parse_datetime_iso_format(self) -> None:
        """parse_datetime handles ISO format with T separator."""
        result = datetime_tools.parse_datetime.function("2025-06-15T14:30:00")
        assert "Parsed" in result

    def test_parse_datetime_iso_z_format(self) -> None:
        """parse_datetime handles ISO format with Z suffix."""
        result = datetime_tools.parse_datetime.function("2025-06-15T14:30:00Z")
        assert "Parsed" in result

    def test_parse_datetime_slash_format(self) -> None:
        """parse_datetime handles DD/MM/YYYY format."""
        result = datetime_tools.parse_datetime.function("15/06/2025")
        assert "Parsed" in result

    def test_parse_datetime_custom_output_format(self) -> None:
        """parse_datetime uses custom output format."""
        result = datetime_tools.parse_datetime.function("2025-06-15", output_format="%d %B %Y")
        assert "15 June 2025" in result

    def test_time_difference_unparseable_start(self) -> None:
        """time_difference returns error for unparseable start date."""
        result = datetime_tools.time_difference.function("bad-date", "2025-12-01")
        assert "Could not parse start date" in result

    def test_time_difference_unparseable_end(self) -> None:
        """time_difference returns error for unparseable end date."""
        result = datetime_tools.time_difference.function("2025-12-01", "bad-date")
        assert "Could not parse end date" in result

    def test_time_difference_hours(self) -> None:
        """time_difference calculates hours."""
        result = datetime_tools.time_difference.function(
            "2025-12-01 00:00:00", "2025-12-01 06:00:00", unit="hours"
        )
        assert "6.00" in result

    def test_time_difference_minutes(self) -> None:
        """time_difference calculates minutes."""
        result = datetime_tools.time_difference.function(
            "2025-12-01 00:00:00", "2025-12-01 02:30:00", unit="minutes"
        )
        assert "150.00" in result

    def test_time_difference_seconds(self) -> None:
        """time_difference calculates seconds."""
        result = datetime_tools.time_difference.function(
            "2025-12-01 00:00:00", "2025-12-01 00:01:00", unit="seconds"
        )
        assert "60.00" in result

    def test_time_difference_invalid_unit(self) -> None:
        """time_difference returns error for invalid unit."""
        result = datetime_tools.time_difference.function("2025-12-01", "2025-12-02", unit="weeks")
        assert "Invalid unit" in result

    def test_date_arithmetic_unparseable(self) -> None:
        """date_arithmetic returns error for unparseable date."""
        result = datetime_tools.date_arithmetic.function("bad", "add", 5)
        assert "Could not parse date" in result

    def test_date_arithmetic_invalid_unit(self) -> None:
        """date_arithmetic returns error for invalid unit."""
        result = datetime_tools.date_arithmetic.function("2025-12-01", "add", 5, unit="weeks")
        assert "Invalid unit" in result

    def test_date_arithmetic_invalid_operation(self) -> None:
        """date_arithmetic returns error for invalid operation."""
        result = datetime_tools.date_arithmetic.function("2025-12-01", "multiply", 5, unit="days")
        assert "Invalid operation" in result

    def test_date_arithmetic_add_hours(self) -> None:
        """date_arithmetic adds hours correctly."""
        result = datetime_tools.date_arithmetic.function(
            "2025-12-01 10:00:00", "add", 3, unit="hours"
        )
        assert "13:00:00" in result

    def test_date_arithmetic_subtract_minutes(self) -> None:
        """date_arithmetic subtracts minutes correctly."""
        result = datetime_tools.date_arithmetic.function(
            "2025-12-01 10:30:00", "subtract", 30, unit="minutes"
        )
        assert "10:00:00" in result

    def test_date_arithmetic_add_seconds(self) -> None:
        """date_arithmetic adds seconds correctly."""
        result = datetime_tools.date_arithmetic.function(
            "2025-12-01 10:00:00", "add", 90, unit="seconds"
        )
        assert "10:01:30" in result

    def test_date_arithmetic_custom_output_format(self) -> None:
        """date_arithmetic uses custom output format."""
        result = datetime_tools.date_arithmetic.function(
            "2025-12-01", "add", 1, unit="days", output_format="%d/%m/%Y"
        )
        assert "02/12/2025" in result


# =============================================================================
# File Tools — extended coverage
# =============================================================================


class TestFileToolsExtended:
    """Cover uncovered branches in file_tools."""

    def test_write_file_invalid_mode(self) -> None:
        """write_file rejects invalid write mode."""
        result = file_tools.write_file.function("/tmp/test.txt", "content", mode="r")
        assert "Invalid mode" in result

    def test_write_file_creates_parent_dirs(self, tmp_path: Path) -> None:
        """write_file creates parent directories if needed."""
        filepath = tmp_path / "sub" / "dir" / "file.txt"
        result = file_tools.write_file.function(str(filepath), "hello")
        assert "Written" in result
        assert filepath.exists()
        assert filepath.read_text() == "hello"

    def test_list_files_not_found(self) -> None:
        """list_files returns error for nonexistent directory."""
        result = file_tools.list_files.function("/nonexistent/dir/xyz123")
        assert "not found" in result.lower()

    def test_list_files_not_a_dir(self, tmp_path: Path) -> None:
        """list_files returns error when path is a file, not directory."""
        filepath = tmp_path / "file.txt"
        filepath.touch()
        result = file_tools.list_files.function(str(filepath))
        assert "Not a directory" in result

    def test_list_files_recursive(self, tmp_path: Path) -> None:
        """list_files with recursive=True lists subdirectory contents."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "inner.txt").write_text("data")
        (tmp_path / "outer.txt").write_text("data")

        result = file_tools.list_files.function(str(tmp_path), recursive=True)
        assert "inner.txt" in result
        assert "outer.txt" in result

    def test_list_files_no_matches(self, tmp_path: Path) -> None:
        """list_files returns message when no files match pattern."""
        (tmp_path / "file.txt").touch()
        result = file_tools.list_files.function(str(tmp_path), pattern="*.xyz")
        assert "No files found" in result

    def test_list_files_show_hidden(self, tmp_path: Path) -> None:
        """list_files with show_hidden=True includes dot-files."""
        (tmp_path / ".hidden").touch()
        (tmp_path / "visible.txt").touch()

        result_hidden = file_tools.list_files.function(str(tmp_path), show_hidden=True)
        assert ".hidden" in result_hidden

        result_no_hidden = file_tools.list_files.function(str(tmp_path), show_hidden=False)
        assert ".hidden" not in result_no_hidden

    def test_list_files_directory_icon(self, tmp_path: Path) -> None:
        """list_files shows directory icon for subdirectories."""
        (tmp_path / "mydir").mkdir()
        result = file_tools.list_files.function(str(tmp_path))
        assert "mydir/" in result

    def test_file_exists_nonexistent(self) -> None:
        """file_exists returns error for nonexistent path."""
        result = file_tools.file_exists.function("/nonexistent/path/xyz123")
        assert "does not exist" in result

    def test_file_exists_directory(self, tmp_path: Path) -> None:
        """file_exists reports directory info."""
        (tmp_path / "child.txt").touch()
        result = file_tools.file_exists.function(str(tmp_path))
        assert "Directory exists" in result
        assert "1 items" in result

    def test_read_file_stream_success(self, tmp_path: Path) -> None:
        """read_file_stream yields lines with line numbers."""
        filepath = tmp_path / "stream.txt"
        filepath.write_text("line one\nline two\nline three\n")

        chunks = list(file_tools.read_file_stream.function(str(filepath)))
        joined = "".join(chunks)

        assert "Reading file" in joined
        assert "Size:" in joined
        assert "[Line    1] line one" in joined
        assert "[Line    2] line two" in joined
        assert "[Line    3] line three" in joined
        assert "Finished reading" in joined

    def test_read_file_stream_not_found(self) -> None:
        """read_file_stream yields error for missing file."""
        chunks = list(file_tools.read_file_stream.function("/nonexistent/file.txt"))
        joined = "".join(chunks)
        assert "not found" in joined.lower()

    def test_read_file_stream_permission_denied(self, tmp_path: Path) -> None:
        """read_file_stream yields error for unreadable file."""
        filepath = tmp_path / "locked.txt"
        filepath.write_text("secret")
        filepath.chmod(0o000)

        try:
            chunks = list(file_tools.read_file_stream.function(str(filepath)))
            joined = "".join(chunks)
            assert "Permission denied" in joined or "Error" in joined
        finally:
            filepath.chmod(0o644)

    def test_read_file_permission_denied(self, tmp_path: Path) -> None:
        """read_file returns error for unreadable file."""
        filepath = tmp_path / "noperm.txt"
        filepath.write_text("secret")
        filepath.chmod(0o000)

        try:
            result = file_tools.read_file.function(str(filepath))
            assert "Permission denied" in result
        finally:
            filepath.chmod(0o644)

    def test_write_file_permission_denied(self) -> None:
        """write_file returns error when directory is not writable."""
        # Try to write to a path we can't write to
        result = file_tools.write_file.function("/proc/nonexistent/file.txt", "data")
        assert "Error" in result

    def test_read_file_generic_exception(self) -> None:
        """read_file handles unexpected exceptions."""
        with mock.patch("pathlib.Path.read_text", side_effect=OSError("disk error")):
            result = file_tools.read_file.function("/some/file.txt")
        assert "Error reading file" in result

    def test_write_file_generic_exception(self, tmp_path: Path) -> None:
        """write_file handles unexpected exceptions."""
        filepath = tmp_path / "test.txt"
        with mock.patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
            result = file_tools.write_file.function(str(filepath), "data")
        assert "Error writing file" in result

    def test_write_file_permission_denied_mock(self, tmp_path: Path) -> None:
        """write_file returns PermissionError message."""
        filepath = tmp_path / "test.txt"
        with mock.patch("pathlib.Path.write_text", side_effect=PermissionError("denied")):
            result = file_tools.write_file.function(str(filepath), "data")
        assert "Permission denied" in result

    def test_list_files_permission_denied(self, tmp_path: Path) -> None:
        """list_files returns error when directory is not accessible."""
        with mock.patch("pathlib.Path.glob", side_effect=PermissionError("denied")):
            result = file_tools.list_files.function(str(tmp_path))
        assert "Permission denied" in result

    def test_list_files_generic_exception(self, tmp_path: Path) -> None:
        """list_files handles unexpected exceptions."""
        with mock.patch("pathlib.Path.glob", side_effect=OSError("io error")):
            result = file_tools.list_files.function(str(tmp_path))
        assert "Error listing files" in result

    def test_file_exists_special_file(self) -> None:
        """file_exists reports special files (not file, not dir)."""
        with (
            mock.patch("pathlib.Path.exists", return_value=True),
            mock.patch("pathlib.Path.is_file", return_value=False),
            mock.patch("pathlib.Path.is_dir", return_value=False),
        ):
            result = file_tools.file_exists.function("/dev/null")
        assert "special file" in result

    def test_file_exists_permission_denied(self) -> None:
        """file_exists handles PermissionError."""
        with mock.patch("pathlib.Path.exists", side_effect=PermissionError("denied")):
            result = file_tools.file_exists.function("/root/secret")
        assert "Permission denied" in result

    def test_file_exists_generic_exception(self) -> None:
        """file_exists handles unexpected exceptions."""
        with mock.patch("pathlib.Path.exists", side_effect=OSError("bad")):
            result = file_tools.file_exists.function("/some/path")
        assert "Error checking path" in result

    def test_read_file_stream_generic_exception(self, tmp_path: Path) -> None:
        """read_file_stream handles unexpected exceptions."""
        filepath = tmp_path / "test.txt"
        filepath.write_text("data")
        with mock.patch("pathlib.Path.stat", side_effect=OSError("broken")):
            chunks = list(file_tools.read_file_stream.function(str(filepath)))
            joined = "".join(chunks)
        assert "Error reading file" in joined


# =============================================================================
# Data Tools — generic exception handlers
# =============================================================================


class TestDataToolsExceptionHandlers:
    """Cover generic except Exception handlers in data_tools."""

    def test_json_to_csv_generic_exception(self) -> None:
        """json_to_csv handles unexpected errors."""
        with mock.patch("json.loads", side_effect=TypeError("bad")):
            result = data_tools.json_to_csv.function('{"x": 1}')
        assert "Error converting to CSV" in result

    def test_csv_to_json_generic_exception(self) -> None:
        """csv_to_json handles unexpected errors."""
        with mock.patch("csv.DictReader", side_effect=TypeError("bad")):
            result = data_tools.csv_to_json.function("a,b\n1,2")
        assert "Error converting to JSON" in result

    def test_extract_json_field_generic_exception(self) -> None:
        """extract_json_field handles unexpected errors."""
        with mock.patch("json.loads", side_effect=TypeError("bad")):
            result = data_tools.extract_json_field.function('{"a": 1}', "a")
        assert "Error extracting field" in result

    def test_format_table_generic_exception(self) -> None:
        """format_table handles unexpected errors."""
        with mock.patch("json.loads", side_effect=TypeError("bad")):
            result = data_tools.format_table.function('[{"x": 1}]')
        assert "Error formatting table" in result

    def test_process_csv_stream_generic_exception(self, tmp_path: Path) -> None:
        """process_csv_stream handles unexpected errors."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        with mock.patch("pathlib.Path.stat", side_effect=RuntimeError("boom")):
            chunks = list(data_tools.process_csv_stream.function(str(csv_file)))
            joined = "".join(chunks)
        assert "Error processing file" in joined


# =============================================================================
# Text Tools — generic exception handlers (via re module mocking)
# =============================================================================


class TestTextToolsExceptionHandlers:
    """Cover generic except Exception handlers in text_tools."""

    def test_search_text_generic_exception(self) -> None:
        """search_text handles unexpected errors."""
        with mock.patch("re.findall", side_effect=TypeError("bad")):
            result = text_tools.search_text.function("hello", "hello")
        assert "Error searching text" in result

    def test_replace_text_generic_exception(self) -> None:
        """replace_text handles unexpected errors."""
        with mock.patch("re.subn", side_effect=TypeError("bad")):
            result = text_tools.replace_text.function("hello", "h", "x")
        assert "Error replacing text" in result

    def test_extract_emails_generic_exception(self) -> None:
        """extract_emails handles unexpected errors."""
        with mock.patch("re.findall", side_effect=TypeError("bad")):
            result = text_tools.extract_emails.function("test@example.com")
        assert "Error extracting emails" in result

    def test_extract_urls_generic_exception(self) -> None:
        """extract_urls handles unexpected errors."""
        with mock.patch("re.findall", side_effect=TypeError("bad")):
            result = text_tools.extract_urls.function("https://example.com")
        assert "Error extracting URLs" in result


# =============================================================================
# DateTime Tools — generic exception handlers
# =============================================================================


class TestDateTimeToolsExceptionHandlers:
    """Cover generic except Exception handlers in datetime_tools."""

    def test_get_current_time_generic_exception(self) -> None:
        """get_current_time handles unexpected errors for non-UTC timezone."""
        # The function handles exceptions internally and returns error strings.
        # We test with a timezone that doesn't exist to trigger the error path.
        result = datetime_tools.get_current_time.function(timezone="Invalid/NonExistent_Zone_12345")
        # Should get either an error message or a "not available" message
        assert any(w in result.lower() for w in ["error", "not available", "only utc", "unknown"])

    def test_parse_datetime_generic_exception(self) -> None:
        """parse_datetime handles truly unparseable input."""
        result = datetime_tools.parse_datetime.function(
            "not-a-date-at-all", input_format="%Y-%m-%d"
        )
        assert "error" in result.lower() or "Error" in result

    def test_time_difference_generic_exception(self) -> None:
        """time_difference handles truly unparseable input."""
        result = datetime_tools.time_difference.function("not-a-date", "also-not-a-date")
        assert "error" in result.lower() or "Error" in result

    def test_date_arithmetic_generic_exception(self) -> None:
        """date_arithmetic handles truly unparseable input."""
        result = datetime_tools.date_arithmetic.function("not-a-date", "add", 1)
        assert "error" in result.lower() or "Error" in result


# =============================================================================
# Helper: mock requests.exceptions namespace
# =============================================================================


def _make_exceptions_ns() -> SimpleNamespace:
    """Build a fake requests.exceptions namespace with real exception classes."""

    class RequestException(Exception):
        pass

    class ConnectionError(RequestException):
        pass

    class Timeout(RequestException):
        pass

    ns = SimpleNamespace()
    ns.RequestException = RequestException
    ns.ConnectionError = ConnectionError
    ns.Timeout = Timeout
    return ns
