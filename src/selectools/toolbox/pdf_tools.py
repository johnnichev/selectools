"""
PDF tools -- extract text and tables from PDF files.

Requires the optional ``pdfplumber`` library, installed via
``pip install selectools[toolbox]`` (or ``pip install pdfplumber``).
The import is lazy: the module loads fine without the dependency and
each tool returns a readable error string when it is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..stability import stable
from ..tools import tool

_MAX_OUTPUT_CHARS = 20_000
_MISSING_DEP_ERROR = (
    "Error: 'pdfplumber' library not installed. Run: pip install selectools[toolbox]"
)


def _parse_page_spec(pages: Optional[str], page_count: int) -> List[int]:
    """Parse a 1-indexed page spec like ``"1-3,5"`` into 0-indexed page numbers.

    Raises:
        ValueError: when the spec is malformed or out of range.
    """
    if pages is None or not pages.strip():
        return list(range(page_count))

    selected: List[int] = []
    for chunk in pages.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, _, end_str = chunk.partition("-")
            start, end = int(start_str), int(end_str)
        else:
            start = end = int(chunk)
        if start < 1 or end > page_count or start > end:
            raise ValueError(
                f"page range '{chunk}' is out of bounds (document has {page_count} pages)"
            )
        selected.extend(range(start - 1, end))

    return sorted(set(selected))


@stable
@tool(description="Extract text content from a PDF file")
def extract_pdf_text(path: str, pages: Optional[str] = None) -> str:
    """
    Extract the text content of a PDF file.

    Args:
        path: Path to the PDF file.
        pages: Optional 1-indexed page selection like ``"1-3,5"``.
            Default: all pages.

    Returns:
        Extracted text with per-page headers, or a readable error string.
    """
    try:
        import pdfplumber  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    pdf_path = Path(path)
    if not pdf_path.is_file():
        return f"Error: File not found: {path}"

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            try:
                indices = _parse_page_spec(pages, page_count)
            except ValueError as exc:
                return f"Error: Invalid pages spec: {exc}"

            lines = [f"PDF: {path} ({page_count} pages, extracting {len(indices)})", ""]
            total = 0
            for idx in indices:
                text = pdf.pages[idx].extract_text() or "(no extractable text)"
                lines.append(f"--- Page {idx + 1} ---")
                lines.append(text)
                lines.append("")
                total += len(text)
                if total > _MAX_OUTPUT_CHARS:
                    lines.append("... (output truncated)")
                    break

            return "\n".join(lines).strip()
    except Exception as exc:
        return f"Error reading PDF: {type(exc).__name__}: {exc}"


@stable
@tool(description="Extract tables from a PDF file")
def extract_pdf_tables(path: str, pages: Optional[str] = None) -> str:
    """
    Extract tables from a PDF file and render them as pipe-delimited rows.

    Args:
        path: Path to the PDF file.
        pages: Optional 1-indexed page selection like ``"1-3,5"``.
            Default: all pages.

    Returns:
        Tables grouped by page in pipe-delimited form, or a readable error
        string. Pages without detected tables are skipped.
    """
    try:
        import pdfplumber  # type: ignore[import-untyped]
    except ImportError:
        return _MISSING_DEP_ERROR

    pdf_path = Path(path)
    if not pdf_path.is_file():
        return f"Error: File not found: {path}"

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            try:
                indices = _parse_page_spec(pages, page_count)
            except ValueError as exc:
                return f"Error: Invalid pages spec: {exc}"

            lines: List[str] = []
            table_count = 0
            total = 0
            for idx in indices:
                tables = pdf.pages[idx].extract_tables() or []
                for t_num, table in enumerate(tables, 1):
                    table_count += 1
                    lines.append(f"--- Page {idx + 1}, Table {t_num} ---")
                    for row in table:
                        rendered = " | ".join("" if cell is None else str(cell) for cell in row)
                        lines.append(rendered)
                        total += len(rendered)
                    lines.append("")
                if total > _MAX_OUTPUT_CHARS:
                    lines.append("... (output truncated)")
                    break

            if table_count == 0:
                return f"No tables found in {path}."

            header = f"PDF: {path} -- {table_count} table(s) extracted"
            return "\n".join([header, ""] + lines).strip()
    except Exception as exc:
        return f"Error reading PDF: {type(exc).__name__}: {exc}"


__stability__ = "stable"

__all__ = [
    "extract_pdf_text",
    "extract_pdf_tables",
]
