"""End-to-end tests for DocumentLoader with real files and URLs.

Exercises the four new v0.21.0 loaders (from_csv, from_json, from_html,
from_url) against real data on disk and (for from_url) a stable public URL.

No API keys are required. ``from_url`` hits ``https://example.com`` which
has been stable for decades and is the canonical "test I can fetch HTML"
target.

Run with:

    pytest tests/rag/test_e2e_document_loaders.py --run-e2e -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from selectools.rag import DocumentLoader

pytestmark = pytest.mark.e2e


class TestFromCSVReal:
    def test_csv_with_text_column(self, tmp_path: Path) -> None:
        """Load a real CSV file using text_column to pick the body field."""
        path = tmp_path / "articles.csv"
        path.write_text(
            "title,body,author\n"
            "First post,This is the body of the first post.,alice\n"
            "Second,Body of the second article.,bob\n",
            encoding="utf-8",
        )
        docs = DocumentLoader.from_csv(
            str(path), text_column="body", metadata_columns=["title", "author"]
        )
        assert len(docs) == 2
        assert docs[0].text == "This is the body of the first post."
        assert docs[0].metadata["title"] == "First post"
        assert docs[0].metadata["author"] == "alice"
        assert docs[1].text == "Body of the second article."

    def test_csv_all_columns_concatenated(self, tmp_path: Path) -> None:
        """When text_column is None, all columns are joined into the text."""
        path = tmp_path / "rows.csv"
        path.write_text("k1,k2\nfoo,bar\n", encoding="utf-8")
        docs = DocumentLoader.from_csv(str(path))
        assert len(docs) == 1
        # Both column values should be present somewhere in the text
        assert "foo" in docs[0].text
        assert "bar" in docs[0].text


class TestFromJSONReal:
    def test_json_array_of_objects(self, tmp_path: Path) -> None:
        """A real JSON array yields one Document per item."""
        path = tmp_path / "posts.json"
        payload = [
            {"body": "first body", "title": "A", "tag": "x"},
            {"body": "second body", "title": "B", "tag": "y"},
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")
        docs = DocumentLoader.from_json(
            str(path), text_field="body", metadata_fields=["title", "tag"]
        )
        assert len(docs) == 2
        assert docs[0].text == "first body"
        assert docs[0].metadata["title"] == "A"
        assert docs[1].metadata["tag"] == "y"

    def test_json_single_object(self, tmp_path: Path) -> None:
        """A single object produces a single Document."""
        path = tmp_path / "one.json"
        path.write_text(json.dumps({"text": "alone", "meta": "value"}), encoding="utf-8")
        docs = DocumentLoader.from_json(str(path), text_field="text")
        assert len(docs) == 1
        assert docs[0].text == "alone"


class TestFromHTMLReal:
    def test_html_full_text_extraction(self, tmp_path: Path) -> None:
        """Real HTML file -> stripped plain text."""
        path = tmp_path / "page.html"
        path.write_text(
            "<html><body>"
            "<h1>Title</h1>"
            "<p>First paragraph.</p>"
            "<p>Second paragraph.</p>"
            "</body></html>",
            encoding="utf-8",
        )
        docs = DocumentLoader.from_html(str(path))
        assert len(docs) == 1
        text = docs[0].text
        assert "Title" in text
        assert "First paragraph" in text
        assert "Second paragraph" in text
        # Tags should be stripped
        assert "<h1>" not in text
        assert "<p>" not in text


class TestFromURLReal:
    def test_fetch_example_com(self) -> None:
        """Real HTTP GET to example.com — this URL has been stable for years."""
        try:
            docs = DocumentLoader.from_url("https://example.com", timeout=15.0)
        except Exception as exc:  # pragma: no cover - network hiccup only
            pytest.skip(f"Network unavailable: {exc}")
        assert len(docs) == 1
        text = docs[0].text
        # example.com contains "Example Domain" — very stable
        assert "Example Domain" in text
        # Source metadata should be the URL
        assert docs[0].metadata.get("source") == "https://example.com"
