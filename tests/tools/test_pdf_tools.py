"""
Tests for PDF tools (extract_pdf_text, extract_pdf_tables).

pdfplumber is mocked via sys.modules -- no real PDFs or optional deps needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from selectools.toolbox.pdf_tools import extract_pdf_tables, extract_pdf_text


class _FakePage:
    def __init__(self, text: str = "", tables: Optional[List[Any]] = None) -> None:
        self._text = text
        self._tables = tables or []

    def extract_text(self) -> str:
        return self._text

    def extract_tables(self) -> List[Any]:
        return self._tables


class _FakePdf:
    def __init__(self, pages: List[_FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> "_FakePdf":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


def _install_fake_pdfplumber(monkeypatch: pytest.MonkeyPatch, pages: List[_FakePage]) -> MagicMock:
    fake = types.ModuleType("pdfplumber")
    opener = MagicMock(return_value=_FakePdf(pages))
    fake.open = opener  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pdfplumber", fake)
    return opener


@pytest.fixture()
def pdf_file(tmp_path: Path) -> str:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return str(path)


class TestExtractPdfText:
    def test_tool_metadata(self) -> None:
        assert extract_pdf_text.name == "extract_pdf_text"
        assert "PDF" in extract_pdf_text.description

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch, pdf_file: str) -> None:
        monkeypatch.setitem(sys.modules, "pdfplumber", None)
        result = extract_pdf_text.function(pdf_file)
        assert "Error" in result
        assert "selectools[toolbox]" in result

    def test_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_pdfplumber(monkeypatch, [])
        result = extract_pdf_text.function("/nonexistent/file.pdf")
        assert "Error" in result
        assert "not found" in result.lower()

    def test_extract_all_pages(self, monkeypatch: pytest.MonkeyPatch, pdf_file: str) -> None:
        _install_fake_pdfplumber(
            monkeypatch, [_FakePage("Page one text"), _FakePage("Page two text")]
        )
        result = extract_pdf_text.function(pdf_file)
        assert "Page one text" in result
        assert "Page two text" in result
        assert "Page 1" in result
        assert "Page 2" in result

    def test_page_selection(self, monkeypatch: pytest.MonkeyPatch, pdf_file: str) -> None:
        _install_fake_pdfplumber(
            monkeypatch,
            [_FakePage("Alpha"), _FakePage("Beta"), _FakePage("Gamma")],
        )
        result = extract_pdf_text.function(pdf_file, pages="1,3")
        assert "Alpha" in result
        assert "Gamma" in result
        assert "Beta" not in result

    def test_page_range(self, monkeypatch: pytest.MonkeyPatch, pdf_file: str) -> None:
        _install_fake_pdfplumber(
            monkeypatch,
            [_FakePage("Alpha"), _FakePage("Beta"), _FakePage("Gamma")],
        )
        result = extract_pdf_text.function(pdf_file, pages="2-3")
        assert "Beta" in result
        assert "Gamma" in result
        assert "Alpha" not in result

    def test_out_of_range_pages(self, monkeypatch: pytest.MonkeyPatch, pdf_file: str) -> None:
        _install_fake_pdfplumber(monkeypatch, [_FakePage("Alpha")])
        result = extract_pdf_text.function(pdf_file, pages="5")
        assert "Error" in result

    def test_malformed_page_spec(self, monkeypatch: pytest.MonkeyPatch, pdf_file: str) -> None:
        _install_fake_pdfplumber(monkeypatch, [_FakePage("Alpha")])
        result = extract_pdf_text.function(pdf_file, pages="abc")
        assert "Error" in result

    def test_corrupt_pdf_readable_error(
        self, monkeypatch: pytest.MonkeyPatch, pdf_file: str
    ) -> None:
        fake = types.ModuleType("pdfplumber")
        fake.open = MagicMock(side_effect=ValueError("not a PDF"))  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pdfplumber", fake)
        result = extract_pdf_text.function(pdf_file)
        assert "Error reading PDF" in result


class TestExtractPdfTables:
    def test_tool_metadata(self) -> None:
        assert extract_pdf_tables.name == "extract_pdf_tables"
        assert "table" in extract_pdf_tables.description.lower()

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch, pdf_file: str) -> None:
        monkeypatch.setitem(sys.modules, "pdfplumber", None)
        result = extract_pdf_tables.function(pdf_file)
        assert "Error" in result
        assert "selectools[toolbox]" in result

    def test_extract_tables(self, monkeypatch: pytest.MonkeyPatch, pdf_file: str) -> None:
        table = [["name", "age"], ["Alice", "30"], ["Bob", None]]
        _install_fake_pdfplumber(monkeypatch, [_FakePage(tables=[table])])
        result = extract_pdf_tables.function(pdf_file)
        assert "name | age" in result
        assert "Alice | 30" in result
        assert "Bob |" in result

    def test_no_tables_found(self, monkeypatch: pytest.MonkeyPatch, pdf_file: str) -> None:
        _install_fake_pdfplumber(monkeypatch, [_FakePage("just text")])
        result = extract_pdf_tables.function(pdf_file)
        assert "No tables found" in result

    def test_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_pdfplumber(monkeypatch, [])
        result = extract_pdf_tables.function("/nonexistent/file.pdf")
        assert "Error" in result
