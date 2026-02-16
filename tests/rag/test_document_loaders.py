"""
Unit tests for document loaders.

Tests the DocumentLoader class with various sources:
- from_text()
- from_file()
- from_directory()
- from_pdf()
"""

from __future__ import annotations

import os
import tempfile
import types
from pathlib import Path
from typing import Generator

import pytest

from selectools.rag import Document
from selectools.rag.loaders import DocumentLoader

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Get path to fixtures directory."""
    return Path(__file__).parents[1] / "fixtures"


@pytest.fixture
def temp_docs_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        (temp_path / "test1.txt").write_text("This is test file 1.")
        (temp_path / "test2.txt").write_text("This is test file 2.")
        (temp_path / "test.md").write_text("# Markdown\n\nThis is markdown.")

        # Create subdirectory
        sub_dir = temp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "test3.txt").write_text("This is test file 3 in subdir.")

        yield temp_path


# ============================================================================
# DocumentLoader.from_text() Tests
# ============================================================================


class TestDocumentLoaderFromText:
    """Test loading documents from raw text."""

    def test_simple_text(self) -> None:
        """Test loading simple text."""
        text = "Hello, world!"
        docs = DocumentLoader.from_text(text)

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].text == text
        assert docs[0].metadata == {}

    def test_text_with_metadata(self) -> None:
        """Test loading text with metadata."""
        text = "Test content"
        metadata = {"source": "test", "author": "tester"}

        docs = DocumentLoader.from_text(text, metadata=metadata)

        assert len(docs) == 1
        assert docs[0].text == text
        assert docs[0].metadata == metadata

    def test_empty_text(self) -> None:
        """Test loading empty text."""
        docs = DocumentLoader.from_text("")

        assert len(docs) == 1
        assert docs[0].text == ""

    def test_multiline_text(self) -> None:
        """Test loading multiline text."""
        text = "Line 1\nLine 2\nLine 3"
        docs = DocumentLoader.from_text(text)

        assert len(docs) == 1
        assert docs[0].text == text
        assert "\n" in docs[0].text


# ============================================================================
# DocumentLoader.from_file() Tests
# ============================================================================


class TestDocumentLoaderFromFile:
    """Test loading documents from files."""

    def test_load_txt_file(self, fixtures_dir: Path) -> None:
        """Test loading a .txt file."""
        file_path = fixtures_dir / "sample.txt"
        docs = DocumentLoader.from_file(str(file_path))

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert len(docs[0].text) > 0
        assert docs[0].metadata["source"] == str(file_path)

    def test_load_md_file(self, fixtures_dir: Path) -> None:
        """Test loading a .md file."""
        file_path = fixtures_dir / "sample.md"
        docs = DocumentLoader.from_file(str(file_path))

        assert len(docs) == 1
        assert len(docs[0].text) > 0
        assert docs[0].metadata["source"] == str(file_path)

    def test_load_with_custom_metadata(self, fixtures_dir: Path) -> None:
        """Test loading file with additional metadata."""
        file_path = fixtures_dir / "sample.txt"
        metadata = {"category": "test", "version": "1.0"}

        docs = DocumentLoader.from_file(str(file_path), metadata=metadata)

        assert len(docs) == 1
        assert docs[0].metadata["category"] == "test"
        assert docs[0].metadata["version"] == "1.0"
        assert docs[0].metadata["source"] == str(file_path)

    def test_load_nonexistent_file(self) -> None:
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            DocumentLoader.from_file("nonexistent.txt")

    def test_unicode_content(self) -> None:
        """Test loading file with Unicode content."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello 世界 🌍")
            temp_path = f.name

        try:
            docs = DocumentLoader.from_file(temp_path)
            assert "世界" in docs[0].text
            assert "🌍" in docs[0].text
        finally:
            os.remove(temp_path)


# ============================================================================
# DocumentLoader.from_directory() Tests
# ============================================================================


class TestDocumentLoaderFromDirectory:
    """Test loading documents from directories."""

    def test_load_all_txt_files(self, temp_docs_dir: Path) -> None:
        """Test loading all .txt files from directory."""
        docs = DocumentLoader.from_directory(
            str(temp_docs_dir), glob_pattern="*.txt", recursive=False
        )

        # Should find test1.txt and test2.txt (not in subdir)
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert all("test file" in d.text for d in docs)

    def test_load_recursive(self, temp_docs_dir: Path) -> None:
        """Test loading files recursively."""
        docs = DocumentLoader.from_directory(str(temp_docs_dir), glob_pattern="**/*.txt")

        # Should find test1.txt, test2.txt, and subdir/test3.txt
        assert len(docs) == 3

    def test_load_specific_pattern(self, temp_docs_dir: Path) -> None:
        """Test loading with specific glob pattern."""
        docs = DocumentLoader.from_directory(str(temp_docs_dir), glob_pattern="test1.txt")

        assert len(docs) == 1
        assert "test file 1" in docs[0].text

    def test_load_multiple_extensions(self, temp_docs_dir: Path) -> None:
        """Test loading files with multiple extensions."""
        # Load both .txt and .md files
        txt_docs = DocumentLoader.from_directory(
            str(temp_docs_dir), glob_pattern="*.txt", recursive=False
        )
        md_docs = DocumentLoader.from_directory(
            str(temp_docs_dir), glob_pattern="*.md", recursive=False
        )

        assert len(txt_docs) == 2
        assert len(md_docs) == 1

    def test_load_empty_directory(self) -> None:
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            docs = DocumentLoader.from_directory(temp_dir, glob_pattern="*.txt")
            assert len(docs) == 0

    def test_load_with_metadata(self, temp_docs_dir: Path) -> None:
        """Test loading directory with custom metadata."""
        metadata = {"project": "test", "version": "1.0"}
        docs = DocumentLoader.from_directory(
            str(temp_docs_dir), glob_pattern="*.txt", metadata=metadata, recursive=False
        )

        assert len(docs) == 2
        assert all(d.metadata["project"] == "test" for d in docs)
        assert all(d.metadata["version"] == "1.0" for d in docs)
        assert all("source" in d.metadata for d in docs)

    def test_metadata_source_preserved(self, temp_docs_dir: Path) -> None:
        """Test that source paths are correctly preserved in metadata."""
        docs = DocumentLoader.from_directory(str(temp_docs_dir), glob_pattern="**/*.txt")

        # Check that each document has a unique source path
        sources = [d.metadata["source"] for d in docs]
        assert len(sources) == len(set(sources))  # All unique
        assert all(isinstance(s, str) for s in sources)


# ============================================================================
# DocumentLoader.from_pdf() Tests
# ============================================================================


class TestDocumentLoaderFromPDF:
    """Test loading documents from PDF files."""

    def test_load_pdf_file(self, fixtures_dir: Path) -> None:
        """Test loading a PDF file."""
        pdf_path = fixtures_dir / "sample.pdf"

        # Skip test if pypdf not installed
        try:
            docs = DocumentLoader.from_pdf(str(pdf_path))
        except ImportError:
            pytest.skip("pypdf not installed")

        assert len(docs) > 0  # At least one page
        assert all(isinstance(d, Document) for d in docs)
        assert all(len(d.text) > 0 for d in docs)
        assert all(d.metadata["source"] == str(pdf_path) for d in docs)

    def test_pdf_page_metadata(self, fixtures_dir: Path) -> None:
        """Test that page numbers are included in metadata."""
        pdf_path = fixtures_dir / "sample.pdf"

        try:
            docs = DocumentLoader.from_pdf(str(pdf_path))
        except ImportError:
            pytest.skip("pypdf not installed")

        # Check page numbers
        page_numbers = [d.metadata.get("page") for d in docs]
        assert all(isinstance(p, int) for p in page_numbers)
        assert page_numbers == list(range(1, len(docs) + 1))

    def test_pdf_with_custom_metadata(self, fixtures_dir: Path) -> None:
        """Test loading PDF with additional metadata."""
        pdf_path = fixtures_dir / "sample.pdf"
        metadata = {"category": "documentation", "author": "test"}

        try:
            docs = DocumentLoader.from_pdf(str(pdf_path), metadata=metadata)
        except ImportError:
            pytest.skip("pypdf not installed")

        assert all(d.metadata["category"] == "documentation" for d in docs)
        assert all(d.metadata["author"] == "test" for d in docs)
        assert all("page" in d.metadata for d in docs)

    def test_pdf_nonexistent_file(self) -> None:
        """Test error handling for missing PDF."""
        try:
            from pypdf import PdfReader  # Check if pypdf is available
        except ImportError:
            pytest.skip("pypdf not installed")

        with pytest.raises(FileNotFoundError):
            DocumentLoader.from_pdf("nonexistent.pdf")

    def test_missing_pypdf_library(self) -> None:
        """Test error when pypdf is not installed."""
        # Temporarily hide pypdf
        import sys

        pypdf_module = sys.modules.get("pypdf")

        if pypdf_module:
            # Only test if we can temporarily remove it
            sys.modules["pypdf"] = None  # type: ignore[assignment]

            try:
                with pytest.raises(ImportError, match="pypdf.*required"):
                    DocumentLoader.from_pdf("dummy.pdf")
            finally:
                # Restore module
                sys.modules["pypdf"] = pypdf_module


# ============================================================================
# Integration Tests
# ============================================================================


class TestDocumentLoaderIntegration:
    """Integration tests for document loader."""

    def test_load_mixed_content(self, temp_docs_dir: Path) -> None:
        """Test loading different file types from same directory."""
        # Load all files
        txt_docs = DocumentLoader.from_directory(
            str(temp_docs_dir), glob_pattern="*.txt", recursive=False
        )
        md_docs = DocumentLoader.from_directory(
            str(temp_docs_dir), glob_pattern="*.md", recursive=False
        )

        all_docs = txt_docs + md_docs

        assert len(all_docs) == 3  # 2 txt + 1 md
        assert all(isinstance(d, Document) for d in all_docs)

    def test_document_metadata_integrity(self) -> None:
        """Test that metadata is preserved correctly."""
        text = "Test content"
        metadata = {
            "key1": "value1",
            "key2": 123,
            "key3": ["list", "items"],
            "key4": {"nested": "dict"},
        }

        docs = DocumentLoader.from_text(text, metadata=metadata)

        assert docs[0].metadata == metadata
        assert docs[0].metadata["key1"] == "value1"
        assert docs[0].metadata["key2"] == 123
        assert docs[0].metadata["key3"] == ["list", "items"]
        assert docs[0].metadata["key4"] == {"nested": "dict"}

    def test_static_methods_callable(self) -> None:
        """Test that all methods are static and callable."""
        assert callable(DocumentLoader.from_text)
        assert callable(DocumentLoader.from_file)
        assert callable(DocumentLoader.from_directory)
        assert callable(DocumentLoader.from_pdf)
