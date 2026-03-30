"""Document loaders for various file formats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from .vector_store import Document


class DocumentLoader:
    """
    Load documents from various sources.

    Supports loading from:
    - Raw text
    - Single files (.txt, .md)
    - Directories (with glob patterns)
    - PDF files

    Example:
        >>> from selectools.rag import DocumentLoader
        >>>
        >>> # Load from text
        >>> docs = DocumentLoader.from_text("Hello world")
        >>>
        >>> # Load from file
        >>> docs = DocumentLoader.from_file("document.txt")
        >>>
        >>> # Load from directory
        >>> docs = DocumentLoader.from_directory("./docs", glob_pattern="**/*.md")
        >>>
        >>> # Load from PDF
        >>> docs = DocumentLoader.from_pdf("manual.pdf")
    """

    @staticmethod
    def from_text(text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        Load a document from raw text.

        Args:
            text: Text content
            metadata: Optional metadata dict

        Returns:
            List containing a single Document
        """
        return [Document(text=text, metadata=metadata or {})]

    @staticmethod
    def from_file(path: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        Load a document from a single file.

        Supports .txt, .md, and other text files.

        Args:
            path: Path to the file
            metadata: Optional metadata dict (will be merged with auto-detected metadata)

        Returns:
            List containing a single Document
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Read file content
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with different encoding
            text = file_path.read_text(encoding="latin-1")

        # Build metadata
        meta = metadata.copy() if metadata else {}
        meta.setdefault("source", str(file_path))
        meta.setdefault("filename", file_path.name)

        return [Document(text=text, metadata=meta)]

    @staticmethod
    def from_directory(
        directory: str,
        glob_pattern: str = "**/*.txt",
        metadata: Optional[Dict] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Load documents from all files in a directory.

        Args:
            directory: Path to the directory
            glob_pattern: Glob pattern to match files (default: **/*.txt)
            metadata: Optional metadata dict to apply to all documents
            recursive: Whether to search recursively (default: True)

        Returns:
            List of Documents

        Example:
            >>> # Load all markdown files
            >>> docs = DocumentLoader.from_directory("./docs", glob_pattern="**/*.md")
            >>>
            >>> # Load all text files in current dir only
            >>> docs = DocumentLoader.from_directory("./", glob_pattern="*.txt", recursive=False)
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Reject patterns that could escape the directory via path traversal.
        # Path components that are purely ".." or contain directory separators
        # outside of the glob wildcard syntax are not legitimate glob patterns.
        if ".." in Path(glob_pattern).parts:
            raise ValueError(
                f"glob_pattern must not contain '..' components to prevent path traversal: "
                f"{glob_pattern!r}"
            )

        # Find matching files
        if recursive and "**" not in glob_pattern:
            pattern = f"**/{glob_pattern}"
        elif not recursive and "**" in glob_pattern:
            # Strip recursive wildcard so recursive=False is honoured even when
            # the caller passes a pattern that contains **
            pattern = glob_pattern.replace("**/", "").replace("**", "")
        else:
            pattern = glob_pattern

        # Guard against degenerate patterns (e.g. '**' with recursive=False becomes '')
        # that would raise ValueError inside Path.glob().
        if not pattern:
            return []

        file_paths = list(dir_path.glob(pattern))

        if not file_paths:
            return []

        # Load each file
        documents = []
        for file_path in file_paths:
            if file_path.is_file():
                try:
                    docs = DocumentLoader.from_file(str(file_path), metadata=metadata)
                    documents.extend(docs)
                except Exception as e:
                    logger.warning("Could not load %s: %s", file_path, e)
                    continue

        return documents

    @staticmethod
    def from_pdf(path: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        Load documents from a PDF file (one document per page).

        Requires pypdf: pip install pypdf

        Args:
            path: Path to the PDF file
            metadata: Optional metadata dict to apply to all pages

        Returns:
            List of Documents (one per page)

        Example:
            >>> docs = DocumentLoader.from_pdf("manual.pdf")
            >>> print(f"Loaded {len(docs)} pages")
        """
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "pypdf package required for PDF loading. " "Install with: pip install pypdf"
            ) from e

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        # Read PDF — raises PdfReadError for encrypted/corrupt files
        try:
            from pypdf.errors import PdfReadError
        except ImportError:
            PdfReadError = Exception  # type: ignore[misc,assignment]

        try:
            reader = PdfReader(path)
        except PdfReadError as e:
            raise ValueError(
                f"Could not read PDF {path!r} — it may be encrypted or corrupt: {e}"
            ) from e

        documents = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()  # returns None for image-only / encrypted pages

            if not (text or "").strip():
                # Skip empty pages
                continue

            # Build metadata
            meta = metadata.copy() if metadata else {}
            meta.setdefault("source", str(file_path))
            meta.setdefault("filename", file_path.name)
            meta["page"] = page_num + 1
            meta["total_pages"] = len(reader.pages)

            documents.append(Document(text=text, metadata=meta))

        return documents


__all__ = ["DocumentLoader"]
