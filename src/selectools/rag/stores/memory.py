"""In-memory vector store implementation using NumPy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...embeddings.provider import EmbeddingProvider

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "numpy required for in-memory vector store. Install with: pip install numpy"
    ) from e

from ..vector_store import Document, SearchResult, VectorStore


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store using NumPy for similarity search.

    Fast, zero external dependencies (except NumPy), great for prototyping.
    All data is stored in memory and will be lost when the process ends.

    Example:
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag.stores import InMemoryVectorStore
        >>>
        >>> embedder = OpenAIEmbeddingProvider()
        >>> store = InMemoryVectorStore(embedder)
        >>>
        >>> # Add documents
        >>> docs = [Document(text="Hello world", metadata={"source": "test"})]
        >>> ids = store.add_documents(docs)
        >>>
        >>> # Search
        >>> query_emb = embedder.embed_query("hi")
        >>> results = store.search(query_emb, top_k=1)
    """

    embedder: "EmbeddingProvider"

    def __init__(self, embedder: "EmbeddingProvider"):  # noqa: F821
        """
        Initialize in-memory vector store.

        Args:
            embedder: Embedding provider to use for computing embeddings
        """
        self.embedder = embedder
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self._id_counter = 0

    def add_documents(
        self, documents: List[Document], embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the store.

        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # Compute embeddings if not provided
        if embeddings is None:
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        # Generate IDs
        new_ids = [f"doc_{self._id_counter + i}" for i in range(len(documents))]
        self._id_counter += len(documents)

        # Add to storage
        self.documents.extend(documents)
        self.ids.extend(new_ids)

        # Update embeddings matrix
        new_embeddings = np.array(embeddings, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        return new_ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Compute cosine similarity
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)

        # Avoid division by zero
        if query_norm == 0:
            return []

        # Cosine similarity = dot product / (norm1 * norm2)
        similarities = np.dot(self.embeddings, query_vec) / (doc_norms * query_norm + 1e-8)

        # Get top-k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

        # Build results with optional filtering
        results = []
        for idx in top_indices:
            doc = self.documents[idx]

            # Apply metadata filter if provided
            if filter and not self._matches_filter(doc, filter):
                continue

            results.append(SearchResult(document=doc, score=float(similarities[idx])))

            # Stop if we have enough results after filtering
            if len(results) >= top_k:
                break

        return results[:top_k]

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        # Find indices to remove
        indices_to_remove = []
        for doc_id in ids:
            if doc_id in self.ids:
                indices_to_remove.append(self.ids.index(doc_id))

        # Remove in reverse order to avoid index shifting
        for idx in sorted(indices_to_remove, reverse=True):
            del self.documents[idx]
            del self.ids[idx]
            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, idx, axis=0)

    def clear(self) -> None:
        """Clear all documents from the store."""
        self.documents = []
        self.embeddings = None
        self.ids = []
        self._id_counter = 0

    def _matches_filter(self, doc: Document, filter: Dict[str, Any]) -> bool:
        """
        Check if document metadata matches the filter.

        Args:
            doc: Document to check
            filter: Metadata filter dict

        Returns:
            True if document matches all filter criteria
        """
        for key, value in filter.items():
            if doc.metadata.get(key) != value:
                return False
        return True


__all__ = ["InMemoryVectorStore"]
