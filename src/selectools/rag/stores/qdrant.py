"""Qdrant vector store implementation."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from ...embeddings.provider import EmbeddingProvider

from ...stability import beta
from ..vector_store import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)


def _import_qdrant() -> Any:
    """Lazy-import qdrant-client, raising a helpful error on failure."""
    try:
        import qdrant_client  # noqa: F811
    except ImportError as e:
        raise ImportError(
            "qdrant-client package required for Qdrant vector store. "
            "Install with: pip install qdrant-client"
        ) from e
    return qdrant_client


def _import_qdrant_models() -> Any:
    """Lazy-import qdrant_client.models."""
    qdrant_client = _import_qdrant()
    return qdrant_client.models


@beta
class QdrantVectorStore(VectorStore):
    """
    Qdrant-based vector store for production similarity search.

    Wraps the Qdrant client for high-performance vector storage with advanced
    features like metadata filtering, gRPC transport, automatic collection
    management, and cosine similarity search.

    The collection is auto-created on the first ``add_documents()`` call if it
    does not already exist.  The embedding dimension is detected from the
    configured ``embedder``.

    Example:
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag.stores import QdrantVectorStore
        >>>
        >>> embedder = OpenAIEmbeddingProvider()
        >>> store = QdrantVectorStore(
        ...     embedder,
        ...     collection_name="my_docs",
        ...     url="http://localhost:6333",
        ... )
        >>>
        >>> # Add documents (auto-creates collection)
        >>> docs = [Document(text="Hello world", metadata={"source": "test"})]
        >>> ids = store.add_documents(docs)
        >>>
        >>> # Search with metadata filtering
        >>> query_emb = embedder.embed_query("hi")
        >>> results = store.search(
        ...     query_emb,
        ...     top_k=3,
        ...     filter={"source": "test"},
        ... )
    """

    embedder: "EmbeddingProvider"

    def __init__(
        self,
        embedder: "EmbeddingProvider",  # noqa: F821
        collection_name: str = "selectools",
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        prefer_grpc: bool = True,
        **qdrant_kwargs: Any,
    ) -> None:
        """
        Initialize Qdrant vector store.

        Args:
            embedder: Embedding provider to use for computing embeddings.
            collection_name: Name of the Qdrant collection.
            url: URL of the Qdrant server (default ``http://localhost:6333``).
            api_key: Optional API key for Qdrant Cloud or authenticated servers.
            prefer_grpc: If ``True``, use gRPC transport when available
                (default ``True``).
            **qdrant_kwargs: Additional arguments forwarded to
                ``QdrantClient``.
        """
        qdrant_client = _import_qdrant()

        self.embedder = embedder
        self.collection_name = collection_name
        self.url = url
        self._collection_exists: bool = False

        self.client = qdrant_client.QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            **qdrant_kwargs,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self, dimension: int) -> None:
        """
        Create the collection if it does not exist yet.

        Uses cosine similarity as the default distance metric.  The check is
        cached in ``_collection_exists`` to avoid repeated round-trips after
        the first call.

        Args:
            dimension: Embedding vector dimension (auto-detected from embedder).
        """
        if self._collection_exists:
            return

        models = _import_qdrant_models()

        # Check whether the collection already exists on the server
        collections = self.client.get_collections().collections
        existing_names = {c.name for c in collections}

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection %r (dim=%d, distance=cosine)",
                self.collection_name,
                dimension,
            )

        self._collection_exists = True

    def _detect_dimension(self, embeddings: List[List[float]]) -> int:
        """
        Detect embedding dimension from the first vector.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Dimension of the embedding vectors.

        Raises:
            ValueError: If the embeddings list is empty.
        """
        if not embeddings:
            raise ValueError("Cannot detect dimension from empty embeddings list")
        return len(embeddings[0])

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to the Qdrant collection.

        If the collection does not exist, it is created automatically with
        the detected embedding dimension and cosine distance.

        Args:
            documents: List of documents to add.
            embeddings: Optional pre-computed embeddings.  If ``None``, the
                store's ``embedder`` computes them.

        Returns:
            List of document IDs (UUID-based strings).
        """
        if not documents:
            return []

        models = _import_qdrant_models()

        # Compute embeddings if not provided
        if embeddings is None:
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        # Auto-create collection on first use
        dimension = self._detect_dimension(embeddings)
        self._ensure_collection(dimension)

        # Generate deterministic IDs and build Qdrant points
        ids: List[str] = []
        points: List[Any] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{hashlib.sha256(doc.text.encode()).hexdigest()[:16]}_{i}"
            ids.append(doc_id)

            # Store text alongside user metadata so we can reconstruct
            # Document objects on retrieval.
            payload: Dict[str, Any] = doc.metadata.copy()
            payload["__selectools_text__"] = doc.text

            points.append(
                models.PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert in a single batch
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Maximum number of results to return.
            filter: Optional metadata filter.  Supports two formats:

                * **Simple dict** — equality conditions, e.g.
                  ``{"source": "manual.pdf"}``.  Each key/value pair becomes a
                  ``FieldCondition`` with ``MatchValue``.
                * **Qdrant native** — a pre-built ``models.Filter`` object for
                  complex queries (range, geo, nested, etc.).

        Returns:
            List of :class:`SearchResult` objects sorted by descending
            similarity score.
        """
        models = _import_qdrant_models()

        # Build Qdrant filter from simple dict or pass-through native filter
        qdrant_filter = self._build_filter(filter)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        search_results: List[SearchResult] = []
        for scored_point in results:
            payload = scored_point.payload or {}

            # Extract document text from payload
            text = payload.get("__selectools_text__", "")
            metadata = {k: v for k, v in payload.items() if k != "__selectools_text__"}

            doc = Document(text=text, metadata=metadata)
            search_results.append(SearchResult(document=doc, score=scored_point.score))

        return search_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        models = _import_qdrant_models()

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
        )

    def clear(self) -> None:
        """
        Clear all documents by deleting and recreating the collection.

        The collection is recreated lazily on the next ``add_documents()``
        call.
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            # Collection may not exist yet — that's fine
            logger.debug("Collection %r did not exist during clear()", self.collection_name)

        self._collection_exists = False

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def _build_filter(self, filter: Optional[Dict[str, Any]]) -> Optional[Any]:
        """
        Convert a simple metadata dict into a Qdrant ``Filter``.

        If *filter* is already a ``models.Filter`` instance it is returned
        as-is.  If it is a plain dict, each key/value pair is converted to
        a ``FieldCondition`` with ``MatchValue`` (equality match).

        Args:
            filter: User-supplied filter specification.

        Returns:
            A ``models.Filter`` or ``None``.
        """
        if filter is None:
            return None

        models = _import_qdrant_models()

        # Pass-through for native Qdrant Filter objects (not a plain dict)
        if not isinstance(filter, dict):
            return filter

        # Simple equality-based filter
        conditions: List[Any] = []
        for key, value in filter.items():
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )

        return models.Filter(must=conditions)

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Return the number of points in the collection.

        Returns:
            Number of stored points, or 0 if the collection does not exist.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"QdrantVectorStore(collection={self.collection_name!r}, " f"url={self.url!r})"


__all__ = ["QdrantVectorStore"]
