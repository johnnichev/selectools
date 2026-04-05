"""Document loaders for various file formats."""

from __future__ import annotations

import csv
import io
import ipaddress
import json
import logging
import re
import socket
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from .vector_store import Document

# Private IP networks that must be blocked to prevent SSRF
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _validate_url(url: str) -> None:
    """Validate a URL to prevent SSRF attacks.

    Raises ValueError if the URL uses a non-HTTP scheme, targets localhost,
    or resolves to a private/reserved IP range.
    """
    parsed = urlparse(url)

    # Only allow http and https schemes
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme {parsed.scheme!r} is not allowed. Only http:// and https:// are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname.")

    # Block well-known loopback/internal hostnames
    lower_host = hostname.lower()
    if lower_host in ("localhost", "0.0.0.0"):
        raise ValueError(f"Requests to {hostname!r} are blocked (loopback/internal address).")

    # Resolve hostname and check against blocked networks
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname {hostname!r}: {e}") from e

    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                raise ValueError(
                    f"URL resolves to private/reserved address {ip} "
                    f"(network {network}). Requests to internal networks are blocked."
                )


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

    @staticmethod
    def from_csv(
        path: str,
        text_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        delimiter: str = ",",
    ) -> List[Document]:
        """
        Load documents from a CSV file. One document per row.

        If ``text_column`` is provided, that column's value becomes the document
        text.  Otherwise all columns are concatenated as ``"key: value"`` pairs.

        Args:
            path: Path to the CSV file
            text_column: Column name to use as document text (None = all columns)
            metadata_columns: Column names to include in metadata (None = all except text)
            delimiter: CSV delimiter (default: comma)

        Returns:
            List of Documents (one per row)

        Example:
            >>> docs = DocumentLoader.from_csv("data.csv", text_column="content")
        """
        if len(delimiter) != 1:
            raise ValueError("delimiter must be a single character")

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        fieldnames: List[str] = list(reader.fieldnames or [])

        if text_column is not None and text_column not in fieldnames:
            raise ValueError(f"text_column {text_column!r} not found in CSV columns: {fieldnames}")

        documents: List[Document] = []
        for row_idx, row in enumerate(reader):
            # Build text
            if text_column is not None:
                text = row.get(text_column, "") or ""
            else:
                text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)

            if not text.strip():
                continue

            # Build metadata
            meta: Dict[str, Any] = {"source": str(file_path), "row": row_idx}
            if metadata_columns is not None:
                for col in metadata_columns:
                    if col in row:
                        meta[col] = row[col]
            else:
                # Include all columns except the text column
                for col in fieldnames:
                    if col != text_column:
                        meta[col] = row.get(col, "")

            documents.append(Document(text=text, metadata=meta))

        return documents

    @staticmethod
    def from_json(
        path: str,
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load documents from a JSON file.

        Handles both JSON arrays (each item becomes a Document) and single
        objects (one Document).

        Args:
            path: Path to the JSON file
            text_field: Key whose value becomes the document text (default: "text")
            metadata_fields: Keys to include in metadata (None = all except text_field)

        Returns:
            List of Documents

        Example:
            >>> docs = DocumentLoader.from_json("articles.json", text_field="body")
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        data = json.loads(content)

        items: List[Dict[str, Any]]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            raise ValueError(f"Expected JSON array or object, got {type(data).__name__}")

        documents: List[Document] = []
        for item in items:
            if not isinstance(item, dict):
                logger.warning("Skipping non-object JSON item: %s", type(item).__name__)
                continue

            raw_text = item.get(text_field, "")
            text = str(raw_text) if raw_text is not None else ""
            if not text.strip():
                continue

            meta: Dict[str, Any] = {"source": str(file_path)}
            if metadata_fields is not None:
                for field in metadata_fields:
                    if field in item:
                        meta[field] = item[field]
            else:
                for key, value in item.items():
                    if key != text_field:
                        meta[key] = value

            documents.append(Document(text=text, metadata=meta))

        return documents

    @staticmethod
    def from_html(
        path: str,
        selector: Optional[str] = None,
        strip_tags: bool = True,
    ) -> List[Document]:
        """
        Load documents from an HTML file.

        Uses BeautifulSoup if available for CSS selector support and clean text
        extraction.  Falls back to regex-based tag stripping when BeautifulSoup
        is not installed.

        Args:
            path: Path to the HTML file
            selector: CSS selector to narrow content (requires BeautifulSoup)
            strip_tags: Whether to strip HTML tags from text (default: True)

        Returns:
            List containing one or more Documents

        Example:
            >>> docs = DocumentLoader.from_html("page.html", selector="article")
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {path}")

        try:
            raw_html = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_html = file_path.read_text(encoding="latin-1")

        return DocumentLoader._parse_html(
            raw_html, source=str(file_path), selector=selector, strip_tags=strip_tags
        )

    @staticmethod
    def from_url(
        url: str,
        selector: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> List[Document]:
        """
        Fetch a URL and load as document.

        Delegates to :meth:`from_html` for HTML content, or
        :meth:`from_text` for plain text.

        Args:
            url: URL to fetch
            selector: CSS selector for HTML content (requires BeautifulSoup)
            headers: Optional HTTP headers dict
            timeout: Request timeout in seconds (default: 30)

        Returns:
            List of Documents

        Example:
            >>> docs = DocumentLoader.from_url("https://example.com/article")
        """
        _validate_url(url)

        req = urllib.request.Request(url, headers=headers or {})

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content_type: str = response.headers.get("Content-Type", "")
                raw_bytes: bytes = response.read()
        except urllib.error.HTTPError as e:
            raise ValueError(f"HTTP error fetching {url}: {e.code} {e.reason}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Could not connect to {url}: {e.reason}") from e

        # Decode response body
        encoding = "utf-8"
        if "charset=" in content_type:
            encoding = (
                content_type.split("charset=")[-1].split(";")[0].strip().strip('"').strip("'")
            )
        try:
            text = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            text = raw_bytes.decode("latin-1")

        base_meta: Dict[str, Any] = {"source": url, "content_type": content_type}

        if "html" in content_type.lower():
            docs = DocumentLoader._parse_html(text, source=url, selector=selector, strip_tags=True)
            for doc in docs:
                doc.metadata.update(base_meta)
            return docs

        # Plain text (or other non-HTML)
        return [Document(text=text, metadata=base_meta)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_html(
        raw_html: str,
        source: str,
        selector: Optional[str] = None,
        strip_tags: bool = True,
    ) -> List[Document]:
        """Parse HTML into Documents, using BeautifulSoup when available."""
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]

            soup = BeautifulSoup(raw_html, "html.parser")

            if selector:
                elements = soup.select(selector)
                if not elements:
                    return []
                documents = []
                for elem in elements:
                    text = elem.get_text(separator="\n", strip=True) if strip_tags else str(elem)
                    if text.strip():
                        documents.append(Document(text=text, metadata={"source": source}))
                return documents

            text = soup.get_text(separator="\n", strip=True) if strip_tags else raw_html
            if not text.strip():
                return []
            return [Document(text=text, metadata={"source": source})]

        except ImportError:
            if selector:
                logger.warning(
                    "BeautifulSoup not installed — CSS selector %r will be ignored. "
                    "Install with: pip install beautifulsoup4",
                    selector,
                )
            if strip_tags:
                # Remove script and style tag content before generic tag strip
                text = re.sub(
                    r"<(script|style)[^>]*>.*?</\1>",
                    "",
                    raw_html,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text = re.sub(r"<[^>]+>", "", text)
                # Collapse excessive whitespace
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
            else:
                text = raw_html

            if not text.strip():
                return []
            return [Document(text=text, metadata={"source": source})]


__all__ = ["DocumentLoader"]
