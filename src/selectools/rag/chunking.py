"""Text chunking utilities for splitting documents into smaller pieces."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .vector_store import Document

if TYPE_CHECKING:
    from ..embeddings.provider import EmbeddingProvider
    from ..providers.base import Provider


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


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex heuristic.

    Handles common abbreviations and decimal numbers to avoid false splits.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(\[])')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors without numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker:
    """
    Split documents at semantic boundaries using embedding similarity.

    Instead of fixed-size windows, SemanticChunker groups consecutive sentences
    whose embeddings are similar, and splits when the similarity drops below a
    threshold.  This produces chunks that respect topic boundaries.

    Requires an ``EmbeddingProvider`` to compute sentence embeddings.

    Example:
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag import SemanticChunker, Document
        >>>
        >>> embedder = OpenAIEmbeddingProvider()
        >>> chunker = SemanticChunker(embedder, similarity_threshold=0.75)
        >>> docs = [Document(text="Long multi-topic text...")]
        >>> chunks = chunker.split_documents(docs)
    """

    def __init__(
        self,
        embedder: "EmbeddingProvider",
        similarity_threshold: float = 0.75,
        min_chunk_sentences: int = 1,
        max_chunk_sentences: int = 50,
    ) -> None:
        """Initialize the semantic chunker.

        Args:
            embedder: Embedding provider for computing sentence vectors.
            similarity_threshold: Cosine similarity below which a new chunk starts
                (range 0.0-1.0, default 0.75).
            min_chunk_sentences: Minimum sentences per chunk (default 1).
            max_chunk_sentences: Maximum sentences before forcing a split (default 50).
        """
        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if min_chunk_sentences < 1:
            raise ValueError("min_chunk_sentences must be at least 1")
        if max_chunk_sentences < min_chunk_sentences:
            raise ValueError("max_chunk_sentences must be >= min_chunk_sentences")

        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences

    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Input text to split.

        Returns:
            List of text chunks split at topic boundaries.
        """
        if not text or not text.strip():
            return []

        sentences = _split_into_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        embeddings = self.embedder.embed_texts(sentences)

        chunks: List[str] = []
        current_group: List[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            at_max = len(current_group) >= self.max_chunk_sentences
            below_threshold = (
                sim < self.similarity_threshold and len(current_group) >= self.min_chunk_sentences
            )

            if at_max or below_threshold:
                chunks.append(" ".join(current_group))
                current_group = [sentences[i]]
            else:
                current_group.append(sentences[i])

        if current_group:
            chunks.append(" ".join(current_group))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantically coherent chunked documents.

        Args:
            documents: List of documents to split.

        Returns:
            List of chunked documents with preserved and enriched metadata.
        """
        chunked: List[Document] = []
        for doc in documents:
            text_chunks = self.split_text(doc.text)
            for i, chunk in enumerate(text_chunks):
                metadata = doc.metadata.copy()
                metadata["chunk"] = i
                metadata["total_chunks"] = len(text_chunks)
                metadata["chunker"] = "semantic"
                chunked.append(Document(text=chunk, metadata=metadata))
        return chunked


class ContextualChunker:
    """
    Wrap any chunker and prepend LLM-generated context to each chunk.

    Inspired by Anthropic's *Contextual Retrieval* technique: for every chunk,
    an LLM generates a short situating description using the **full** document as
    context.  The description is prepended to the chunk text so that the
    embedding (and later retrieval) captures the chunk's role within the
    document.

    The underlying splitting can be any object with a ``split_documents``
    method (``TextSplitter``, ``RecursiveTextSplitter``, ``SemanticChunker``).

    Example:
        >>> from selectools.rag import ContextualChunker, RecursiveTextSplitter
        >>> from selectools import OpenAIProvider
        >>>
        >>> base = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=200)
        >>> provider = OpenAIProvider()
        >>> chunker = ContextualChunker(base_chunker=base, provider=provider)
        >>> docs = [Document(text="Full document text...")]
        >>> enriched = chunker.split_documents(docs)
    """

    _DEFAULT_PROMPT = (
        "Document:\n<document>\n{document}\n</document>\n\n"
        "Chunk:\n<chunk>\n{chunk}\n</chunk>\n\n"
        "Give a short (1-2 sentence) description that situates this chunk "
        "within the overall document for search and retrieval purposes. "
        "Respond ONLY with the description, nothing else."
    )

    def __init__(
        self,
        base_chunker: Any,
        provider: "Provider",
        model: str = "gpt-4o-mini",
        prompt_template: Optional[str] = None,
        max_document_chars: int = 50_000,
        context_prefix: str = "[Context] ",
    ) -> None:
        """Initialize the contextual chunker.

        Args:
            base_chunker: Underlying chunker with a ``split_documents`` method
                (e.g. ``RecursiveTextSplitter``, ``SemanticChunker``).
            provider: LLM provider for generating chunk context descriptions.
            model: Model to use for context generation (default ``"gpt-4o-mini"``).
            prompt_template: Custom prompt with ``{document}`` and ``{chunk}``
                placeholders.  Falls back to a sensible default.
            max_document_chars: Truncate the full document to this many chars
                before inserting into the prompt (default 50 000).
            context_prefix: Prefix prepended before the generated context line
                (default ``"[Context] "``).
        """
        if not hasattr(base_chunker, "split_documents"):
            raise TypeError("base_chunker must have a split_documents(documents) method")

        self.base_chunker = base_chunker
        self.provider = provider
        self.model = model
        self.prompt_template = prompt_template or self._DEFAULT_PROMPT
        self.max_document_chars = max_document_chars
        self.context_prefix = context_prefix

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents and enrich each chunk with LLM-generated context.

        Args:
            documents: List of documents to split and enrich.

        Returns:
            List of contextually enriched chunked documents.
        """
        from ..types import Message, Role

        doc_text_map: Dict[int, str] = {}
        for idx, doc in enumerate(documents):
            doc_text_map[idx] = doc.text

        all_base_chunks: List[Document] = []
        chunk_origins: List[int] = []
        for idx, doc in enumerate(documents):
            chunks = self.base_chunker.split_documents([doc])
            all_base_chunks.extend(chunks)
            chunk_origins.extend([idx] * len(chunks))

        enriched: List[Document] = []
        for chunk_doc, origin_idx in zip(all_base_chunks, chunk_origins):
            full_text = doc_text_map[origin_idx]
            truncated_doc = full_text[: self.max_document_chars]

            prompt = self.prompt_template.format(
                document=truncated_doc,
                chunk=chunk_doc.text,
            )

            response_msg, _ = self.provider.complete(
                model=self.model,
                system_prompt="You are a concise technical writer.",
                messages=[Message(role=Role.USER, content=prompt)],
                tools=[],
                temperature=0.0,
            )

            context_line = response_msg.content.strip()
            enriched_text = f"{self.context_prefix}{context_line}\n\n{chunk_doc.text}"

            metadata = chunk_doc.metadata.copy()
            metadata["context"] = context_line
            metadata["chunker"] = "contextual"

            enriched.append(Document(text=enriched_text, metadata=metadata))

        return enriched

    def split_text(self, text: str) -> List[str]:
        """
        Convenience: wrap text in a Document, split, return text chunks.

        Args:
            text: Raw text to split and enrich.

        Returns:
            List of contextually enriched text chunks.
        """
        docs = self.split_documents([Document(text=text)])
        return [d.text for d in docs]


__all__ = [
    "TextSplitter",
    "RecursiveTextSplitter",
    "SemanticChunker",
    "ContextualChunker",
]
