"""Tests for CSV, JSON, HTML, and URL document loaders."""

from __future__ import annotations

import json
import textwrap
import urllib.error
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from selectools.rag.loaders import DocumentLoader
from selectools.rag.vector_store import Document

# ---------------------------------------------------------------------------
# from_csv
# ---------------------------------------------------------------------------


class TestFromCSV:
    """Tests for DocumentLoader.from_csv()."""

    def test_csv_with_text_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,content,author\n1,Hello world,Alice\n2,Goodbye,Bob\n")
        docs = DocumentLoader.from_csv(str(csv_file), text_column="content")
        assert len(docs) == 2
        assert docs[0].text == "Hello world"
        assert docs[1].text == "Goodbye"
        assert docs[0].metadata["row"] == 0
        assert docs[1].metadata["row"] == 1

    def test_csv_without_text_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")
        docs = DocumentLoader.from_csv(str(csv_file))
        assert len(docs) == 2
        assert "name: Alice" in docs[0].text
        assert "age: 30" in docs[0].text

    def test_csv_with_metadata_columns(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,text,category,lang\n1,Hello,news,en\n")
        docs = DocumentLoader.from_csv(
            str(csv_file), text_column="text", metadata_columns=["category"]
        )
        assert len(docs) == 1
        assert docs[0].metadata["category"] == "news"
        assert "lang" not in docs[0].metadata

    def test_csv_metadata_includes_all_except_text(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,text,tag\n1,Hello,foo\n")
        docs = DocumentLoader.from_csv(str(csv_file), text_column="text")
        assert "id" in docs[0].metadata
        assert "tag" in docs[0].metadata
        assert "text" not in {k for k in docs[0].metadata if k not in ("source", "row")}

    def test_csv_source_metadata(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n")
        docs = DocumentLoader.from_csv(str(csv_file))
        assert docs[0].metadata["source"] == str(csv_file)

    def test_csv_empty_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        docs = DocumentLoader.from_csv(str(csv_file))
        assert docs == []

    def test_csv_header_only(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "header.csv"
        csv_file.write_text("a,b,c\n")
        docs = DocumentLoader.from_csv(str(csv_file))
        assert docs == []

    def test_csv_missing_text_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="text_column.*not found"):
            DocumentLoader.from_csv(str(csv_file), text_column="missing")

    def test_csv_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            DocumentLoader.from_csv("/nonexistent/data.csv")

    def test_csv_custom_delimiter(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.tsv"
        csv_file.write_text("name\tval\nAlice\t42\n")
        docs = DocumentLoader.from_csv(str(csv_file), delimiter="\t")
        assert len(docs) == 1
        assert "Alice" in docs[0].text

    def test_csv_skips_empty_rows(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("text\nHello\n\nWorld\n")
        docs = DocumentLoader.from_csv(str(csv_file), text_column="text")
        assert len(docs) == 2

    def test_csv_latin1_fallback(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "latin.csv"
        csv_file.write_bytes(b"text\ncaf\xe9\n")
        docs = DocumentLoader.from_csv(str(csv_file), text_column="text")
        assert len(docs) == 1
        assert "caf" in docs[0].text


# ---------------------------------------------------------------------------
# from_json
# ---------------------------------------------------------------------------


class TestFromJSON:
    """Tests for DocumentLoader.from_json()."""

    def test_json_array(self, tmp_path: Path) -> None:
        data = [{"text": "Hello"}, {"text": "World"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        docs = DocumentLoader.from_json(str(f))
        assert len(docs) == 2
        assert docs[0].text == "Hello"
        assert docs[1].text == "World"

    def test_json_single_object(self, tmp_path: Path) -> None:
        data = {"text": "Solo"}
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        docs = DocumentLoader.from_json(str(f))
        assert len(docs) == 1
        assert docs[0].text == "Solo"

    def test_json_custom_text_field(self, tmp_path: Path) -> None:
        data = [{"body": "Content here", "title": "T1"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        docs = DocumentLoader.from_json(str(f), text_field="body")
        assert docs[0].text == "Content here"

    def test_json_metadata_fields(self, tmp_path: Path) -> None:
        data = [{"text": "Hello", "author": "A", "year": 2024, "secret": "x"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        docs = DocumentLoader.from_json(str(f), metadata_fields=["author", "year"])
        assert docs[0].metadata["author"] == "A"
        assert docs[0].metadata["year"] == 2024
        assert "secret" not in docs[0].metadata

    def test_json_all_metadata_except_text(self, tmp_path: Path) -> None:
        data = [{"text": "Hello", "k1": "v1", "k2": "v2"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        docs = DocumentLoader.from_json(str(f))
        assert docs[0].metadata["k1"] == "v1"
        assert docs[0].metadata["k2"] == "v2"

    def test_json_source_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"text": "hi"}]))
        docs = DocumentLoader.from_json(str(f))
        assert docs[0].metadata["source"] == str(f)

    def test_json_empty_array(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        f.write_text("[]")
        docs = DocumentLoader.from_json(str(f))
        assert docs == []

    def test_json_skips_empty_text(self, tmp_path: Path) -> None:
        data = [{"text": ""}, {"text": "ok"}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        docs = DocumentLoader.from_json(str(f))
        assert len(docs) == 1
        assert docs[0].text == "ok"

    def test_json_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            DocumentLoader.from_json("/nonexistent/data.json")

    def test_json_invalid_top_level_type(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text('"just a string"')
        with pytest.raises(ValueError, match="Expected JSON array or object"):
            DocumentLoader.from_json(str(f))

    def test_json_skips_non_dict_items(self, tmp_path: Path) -> None:
        data = [{"text": "ok"}, "bare string", 42]
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data))
        docs = DocumentLoader.from_json(str(f))
        assert len(docs) == 1

    def test_json_latin1_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "latin.json"
        f.write_bytes(b'[{"text": "caf\xe9"}]')
        docs = DocumentLoader.from_json(str(f))
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# from_html
# ---------------------------------------------------------------------------


class TestFromHTML:
    """Tests for DocumentLoader.from_html()."""

    def test_html_basic_strip(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Hello world</p></body></html>")
        docs = DocumentLoader.from_html(str(f))
        assert len(docs) == 1
        assert "Hello world" in docs[0].text
        assert "<p>" not in docs[0].text

    def test_html_no_strip(self, tmp_path: Path) -> None:
        html = "<div>Hello</div>"
        f = tmp_path / "page.html"
        f.write_text(html)
        docs = DocumentLoader.from_html(str(f), strip_tags=False)
        assert "<div>" in docs[0].text

    def test_html_with_selector_bs4(self, tmp_path: Path) -> None:
        html = "<html><body><article>Article text</article><footer>F</footer></body></html>"
        f = tmp_path / "page.html"
        f.write_text(html)
        docs = DocumentLoader.from_html(str(f), selector="article")
        assert len(docs) == 1
        assert "Article text" in docs[0].text
        assert "F" not in docs[0].text

    def test_html_selector_no_match(self, tmp_path: Path) -> None:
        html = "<html><body><p>Hi</p></body></html>"
        f = tmp_path / "page.html"
        f.write_text(html)
        docs = DocumentLoader.from_html(str(f), selector=".nonexistent")
        assert docs == []

    def test_html_multiple_selector_matches(self, tmp_path: Path) -> None:
        html = "<div class='item'>A</div><div class='item'>B</div>"
        f = tmp_path / "page.html"
        f.write_text(html)
        docs = DocumentLoader.from_html(str(f), selector=".item")
        assert len(docs) == 2

    def test_html_source_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text("<p>Hi</p>")
        docs = DocumentLoader.from_html(str(f))
        assert docs[0].metadata["source"] == str(f)

    def test_html_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.html"
        f.write_text("")
        docs = DocumentLoader.from_html(str(f))
        assert docs == []

    def test_html_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            DocumentLoader.from_html("/nonexistent/page.html")

    def test_html_without_bs4_strips_tags(self, tmp_path: Path) -> None:
        """When BeautifulSoup is not installed, fall back to regex stripping."""
        f = tmp_path / "page.html"
        f.write_text("<h1>Title</h1><p>Body</p>")

        import builtins

        real_import = builtins.__import__

        def _mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "bs4":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            docs = DocumentLoader.from_html(str(f))

        assert len(docs) == 1
        assert "Title" in docs[0].text
        assert "<h1>" not in docs[0].text

    def test_html_without_bs4_selector_warning(self, tmp_path: Path) -> None:
        """Selector is ignored (with warning) when bs4 is missing."""
        f = tmp_path / "page.html"
        f.write_text("<p>Hello</p>")

        import builtins

        real_import = builtins.__import__

        def _mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "bs4":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            docs = DocumentLoader.from_html(str(f), selector="p")

        assert len(docs) == 1
        assert "Hello" in docs[0].text

    def test_html_latin1_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_bytes(b"<p>caf\xe9</p>")
        docs = DocumentLoader.from_html(str(f))
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# from_url
# ---------------------------------------------------------------------------


class TestFromURL:
    """Tests for DocumentLoader.from_url()."""

    def _mock_urlopen(
        self,
        body: bytes,
        content_type: str = "text/html; charset=utf-8",
        status: int = 200,
    ) -> MagicMock:
        """Create a mock for urllib.request.urlopen context manager."""
        response = MagicMock()
        response.read.return_value = body
        response.headers = {"Content-Type": content_type}
        response.status = status
        response.__enter__ = lambda s: s
        response.__exit__ = MagicMock(return_value=False)
        return response

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_html_response(self, mock_urlopen: MagicMock) -> None:
        html = b"<html><body><p>Hello from URL</p></body></html>"
        mock_urlopen.return_value = self._mock_urlopen(html)
        docs = DocumentLoader.from_url("https://example.com")
        assert len(docs) == 1
        assert "Hello from URL" in docs[0].text
        assert docs[0].metadata["source"] == "https://example.com"
        assert "html" in docs[0].metadata["content_type"]

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_plain_text_response(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = self._mock_urlopen(
            b"Plain text content", content_type="text/plain"
        )
        docs = DocumentLoader.from_url("https://example.com/file.txt")
        assert len(docs) == 1
        assert docs[0].text == "Plain text content"
        assert docs[0].metadata["content_type"] == "text/plain"

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_with_selector(self, mock_urlopen: MagicMock) -> None:
        html = b"<div><article>Target</article><aside>Skip</aside></div>"
        mock_urlopen.return_value = self._mock_urlopen(html)
        docs = DocumentLoader.from_url("https://example.com", selector="article")
        assert len(docs) == 1
        assert "Target" in docs[0].text

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_with_headers(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = self._mock_urlopen(b"<p>ok</p>")
        DocumentLoader.from_url(
            "https://example.com",
            headers={"Authorization": "Bearer token123"},
        )
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "Bearer token123"

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://example.com",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with pytest.raises(ValueError, match="HTTP error.*404"):
            DocumentLoader.from_url("https://example.com/missing")

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_connection_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(ConnectionError, match="Could not connect"):
            DocumentLoader.from_url("https://down.example.com")

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_timeout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("timed out")
        with pytest.raises(ConnectionError):
            DocumentLoader.from_url("https://slow.example.com", timeout=1.0)

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_charset_detection(self, mock_urlopen: MagicMock) -> None:
        body = "Caf\u00e9".encode("utf-8")
        mock_urlopen.return_value = self._mock_urlopen(
            body, content_type="text/html; charset=utf-8"
        )
        docs = DocumentLoader.from_url("https://example.com")
        assert "Caf\u00e9" in docs[0].text

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_latin1_body_fallback(self, mock_urlopen: MagicMock) -> None:
        body = b"caf\xe9"
        mock_urlopen.return_value = self._mock_urlopen(body, content_type="text/plain")
        docs = DocumentLoader.from_url("https://example.com/file.txt")
        assert len(docs) == 1

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_content_type_metadata(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = self._mock_urlopen(
            b"data", content_type="application/octet-stream"
        )
        docs = DocumentLoader.from_url("https://example.com/blob")
        assert docs[0].metadata["content_type"] == "application/octet-stream"

    @patch("selectools.rag.loaders.urllib.request.urlopen")
    def test_url_passes_timeout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = self._mock_urlopen(b"<p>ok</p>")
        DocumentLoader.from_url("https://example.com", timeout=5.0)
        _, kwargs = mock_urlopen.call_args
        assert kwargs.get("timeout") == 5.0
