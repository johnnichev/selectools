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
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter (e.g., {"source": "manual.pdf"})

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
    def create(
        backend: str, embedder: "EmbeddingProvider", **kwargs: Any
    ) -> "VectorStore":  # noqa: F821
        """
        Factory method to create a vector store.

        Args:
            backend: Backend type ("memory", "sqlite", "chroma", "pinecone")
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
        else:
            raise ValueError(
                f"Unknown vector store backend: {backend}. "
                f"Supported: memory, sqlite, chroma, pinecone"
            )


__all__ = ["Document", "SearchResult", "VectorStore"]
