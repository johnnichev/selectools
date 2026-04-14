"""
Vector store interface and data models for RAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..embeddings.provider import EmbeddingProvider


@dataclass
class Document:
    """
    A document with text and metadata.

    Attributes:
        text: The document text content
        metadata: Optional metadata dict (e.g., source, page number, etc.)
        embedding: Optional pre-computed embedding vector
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Initialize metadata if None."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """
    A search result containing a document and similarity score.

    Attributes:
        document: The matched document
        score: Similarity score (typically cosine similarity, higher is better)
    """

    document: Document
    score: float


def _validate_filter(filter: Optional[Dict[str, Any]]) -> None:
    """Raise ``NotImplementedError`` if ``filter`` uses operator-dict syntax.

    BUG-25 / LlamaIndex #20246: in-memory and BM25 filter matchers compare
    metadata values with ``!=``. If a user passes an operator dict like
    ``{"user_id": {"$in": [1, 2]}}`` expecting Mongo-style operator semantics,
    the equality check fails for every document and returns zero results with
    no indication of user error. We detect operator intent (a dict value with
    one or more ``$``-prefixed keys) and raise a clear error instead of
    silently returning the wrong result.

    Literal dict metadata values (no ``$``-prefixed keys) still pass through
    for backward compatibility with nested-metadata matching.
    """
    if not filter:
        return
    for _key, value in filter.items():
        if isinstance(value, dict) and any(
            isinstance(k, str) and k.startswith("$") for k in value.keys()
        ):
            bad = next(k for k in value.keys() if isinstance(k, str) and k.startswith("$"))
            raise NotImplementedError(
                f"In-memory filter does not support operator syntax {bad!r}. "
                f"Use a vector store backend that supports operators "
                f"(Chroma, Pinecone, Qdrant, pgvector) or use equality-only filters."
            )


def _dedup_search_results(results: List["SearchResult"]) -> List["SearchResult"]:
    """Post-filter search results so each unique ``(text, source)`` pair appears once.

    Keeps the first occurrence (highest-scoring when the input is already
    sorted by descending similarity). Used by vector store ``search()`` methods
    when called with ``dedup=True``.

    Dedup key is ``(document.text, document.metadata.get("source"))`` so that
    two documents with identical text but different source metadata — a
    common case when the same snippet is ingested from multiple files — are
    preserved as distinct citations. When no ``source`` key is present the
    fallback is text-only (BUG-24 / LlamaIndex #21033).

    Args:
        results: Ordered list of SearchResult objects.

    Returns:
        New list with duplicate ``(text, source)`` results removed.
    """
    seen: set = set()
    out: List["SearchResult"] = []
    for r in results:
        source = r.document.metadata.get("source") if r.document.metadata else None
        key = (r.document.text, source)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    All vector stores must implement this interface to ensure consistency
    across different backends (in-memory, SQLite, Chroma, Pinecone).
    """

    embedder: Optional["EmbeddingProvider"] = None

    @abstractmethod
    def add_documents(
        self, documents: List[Document], embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings. If None, will be computed
                       using the store's embedding provider.

        Returns:
            List of document IDs (strings) assigned to the added documents
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        dedup: bool = False,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter (e.g., {"source": "manual.pdf"})
            dedup: If True, post-filter results so each unique document text
                appears at most once (keeps the first — highest-scoring —
                occurrence). Default False for backward compatibility.

        Returns:
            List of SearchResult objects, sorted by similarity (highest first)
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        """
        pass

    @staticmethod
    def create(backend: str, embedder: "EmbeddingProvider", **kwargs: Any) -> "VectorStore":  # noqa: F821
        """
        Factory method to create a vector store.

        Args:
            backend: Backend type ("memory", "sqlite", "chroma", "pinecone",
                     "faiss", "qdrant", "pgvector")
            embedder: Embedding provider to use for computing embeddings
            **kwargs: Additional backend-specific arguments

        Returns:
            VectorStore instance

        Example:
            >>> from selectools.embeddings import OpenAIEmbeddingProvider
            >>> from selectools.rag import VectorStore
            >>>
            >>> embedder = OpenAIEmbeddingProvider()
            >>> store = VectorStore.create("memory", embedder=embedder)
        """
        if backend == "memory":
            from .stores.memory import InMemoryVectorStore

            return InMemoryVectorStore(embedder, **kwargs)
        elif backend == "sqlite":
            from .stores.sqlite import SQLiteVectorStore

            return SQLiteVectorStore(embedder, **kwargs)
        elif backend == "chroma":
            from .stores.chroma import ChromaVectorStore

            return ChromaVectorStore(embedder, **kwargs)
        elif backend == "pinecone":
            from .stores.pinecone import PineconeVectorStore

            return PineconeVectorStore(embedder, **kwargs)
        elif backend == "faiss":
            from .stores.faiss import FAISSVectorStore

            return FAISSVectorStore(embedder, **kwargs)
        elif backend == "qdrant":
            from .stores.qdrant import QdrantVectorStore

            return QdrantVectorStore(embedder, **kwargs)
        elif backend == "pgvector":
            from .stores.pgvector import PgVectorStore

            return PgVectorStore(embedder, **kwargs)
        else:
            raise ValueError(
                f"Unknown vector store backend: {backend}. "
                f"Supported: memory, sqlite, chroma, pinecone, faiss, qdrant, pgvector"
            )


__all__ = [
    "Document",
    "SearchResult",
    "VectorStore",
    "_dedup_search_results",
    "_validate_filter",
]
