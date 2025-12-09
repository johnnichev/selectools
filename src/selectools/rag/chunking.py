"""Text chunking utilities for splitting documents into smaller pieces."""

from __future__ import annotations

from typing import Callable, List, Optional

from .vector_store import Document


class TextSplitter:
    """
    Split text into chunks with fixed size and overlap.

    Example:
        >>> from selectools.rag import TextSplitter, Document
        >>>
        >>> splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        >>> text = "Long text..." * 1000
        >>> chunks = splitter.split_text(text)
        >>> print(f"Split into {len(chunks)} chunks")
        >>>
        >>> # Split documents
        >>> docs = [Document(text="Long text...", metadata={"source": "test"})]
        >>> chunked_docs = splitter.split_documents(docs)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        separator: str = "\n\n",
    ):
        """
        Initialize text splitter.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            length_function: Function to measure text length (default: len)
            separator: Separator to try to split on (default: double newline)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = self.length_function(text)

        while start < text_length:
            # Determine end position
            end = start + self.chunk_size

            if end >= text_length:
                # Last chunk
                chunks.append(text[start:])
                break

            # Try to find a natural break point (separator)
            chunk = text[start:end]
            separator_pos = chunk.rfind(self.separator)

            if separator_pos != -1 and separator_pos > self.chunk_size // 2:
                # Found a good break point
                end = start + separator_pos + len(self.separator)
                chunk = text[start:end]

            chunks.append(chunk)

            # Move start position forward, accounting for overlap
            start = end - self.chunk_overlap

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunked documents.

        Args:
            documents: List of documents to split

        Returns:
            List of chunked documents with preserved metadata
        """
        chunked_docs = []

        for doc in documents:
            chunks = self.split_text(doc.text)

            for i, chunk in enumerate(chunks):
                # Preserve metadata and add chunk index
                metadata = doc.metadata.copy()
                metadata["chunk"] = i
                metadata["total_chunks"] = len(chunks)

                chunked_docs.append(Document(text=chunk, metadata=metadata))

        return chunked_docs


class RecursiveTextSplitter(TextSplitter):
    r"""
    Split text recursively using multiple separators.

    Tries to split on separators in order: double newline, newline, sentence,
    space, then character-by-character as a last resort.

    Example:
        >>> from selectools.rag import RecursiveTextSplitter
        >>>
        >>> splitter = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=200)
        >>> text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3..."
        >>> chunks = splitter.split_text(text)
        >>> # Splits on natural boundaries (paragraphs, sentences, etc.)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
    ):
        r"""
        Initialize recursive text splitter.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to try (default: ["\\n\\n", "\\n", ". ", " ", ""])
            length_function: Function to measure text length (default: len)
        """
        super().__init__(chunk_size, chunk_overlap, length_function)

        if separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]
        else:
            self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """
        Split text recursively on separators.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the list of separators.

        Args:
            text: Text to split
            separators: Remaining separators to try

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Base case: no more separators or text is small enough
        if not separators or self.length_function(text) <= self.chunk_size:
            return [text] if text else []

        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on current separator
        if separator:
            splits = text.split(separator)
        else:
            # Last resort: split character by character
            splits = list(text)

        # Recombine splits into chunks
        chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for i, split in enumerate(splits):
            split_length = self.length_function(split)

            # Add separator length (except for first split and empty separator)
            sep_length = self.length_function(separator) if separator and i > 0 else 0

            # Check if adding this split would exceed chunk size
            if current_length + split_length + sep_length > self.chunk_size and current_chunk:
                # Current chunk is full, save it
                chunk_text = separator.join(current_chunk) if separator else "".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Keep last part for overlap
                    overlap_text = (
                        separator.join(current_chunk) if separator else "".join(current_chunk)
                    )
                    overlap_length = self.length_function(overlap_text)

                    if overlap_length > self.chunk_overlap:
                        # Trim overlap to fit
                        overlap_text = overlap_text[-self.chunk_overlap :]

                    current_chunk = (
                        [overlap_text, split] if separator else list(overlap_text) + [split]
                    )
                    current_length = self.length_function(
                        separator.join(current_chunk) if separator else "".join(current_chunk)
                    )
                else:
                    current_chunk = [split]
                    current_length = split_length
            else:
                # Add split to current chunk
                current_chunk.append(split)
                current_length += split_length + sep_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk) if separator else "".join(current_chunk)
            chunks.append(chunk_text)

        # If any chunks are still too large, recurse with next separator
        final_chunks = []
        for chunk in chunks:
            if self.length_function(chunk) > self.chunk_size and remaining_separators:
                # Chunk is still too large, try next separator
                sub_chunks = self._split_text_recursive(chunk, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks


__all__ = ["TextSplitter", "RecursiveTextSplitter"]
