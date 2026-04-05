"""PostgreSQL + pgvector vector store implementation."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...embeddings.provider import EmbeddingProvider

from ...stability import beta
from ..vector_store import Document, SearchResult, VectorStore

logger = logging.getLogger(__name__)


def _import_psycopg2() -> Any:
    """Lazy import psycopg2 with a helpful error message."""
    try:
        import psycopg2
        import psycopg2.extras  # noqa: F401 — needed for Json adapter

        return psycopg2
    except ImportError as e:
        raise ImportError(
            "psycopg2 required for PgVectorStore. "
            "Install with: pip install 'selectools[postgres]' "
            "or: pip install psycopg2-binary"
        ) from e


@beta
class PgVectorStore(VectorStore):
    """
    PostgreSQL + pgvector based vector store.

    Uses the ``pgvector`` extension for native vector similarity search with
    cosine distance (``<=>`` operator). Requires a PostgreSQL database with
    the ``vector`` extension installed.

    Features:
        - Automatic table and index creation on first use
        - HNSW index for fast approximate nearest-neighbour search
        - JSONB metadata with parameterized queries throughout (no SQL injection)
        - Thread-safe connection management
        - Cosine distance search with score conversion (1 - distance)

    Example:
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag.stores.pgvector import PgVectorStore
        >>>
        >>> embedder = OpenAIEmbeddingProvider()
        >>> store = PgVectorStore(
        ...     embedder=embedder,
        ...     connection_string="postgresql://user:pass@localhost:5432/mydb",
        ... )
        >>>
        >>> # Add documents
        >>> from selectools.rag import Document
        >>> docs = [Document(text="Hello world", metadata={"source": "test"})]
        >>> ids = store.add_documents(docs)
        >>>
        >>> # Search
        >>> query_emb = embedder.embed_query("hi")
        >>> results = store.search(query_emb, top_k=3)

    Note:
        Requires ``psycopg2`` (or ``psycopg2-binary``) and a PostgreSQL server
        with the ``vector`` extension. Install the optional dependency with::

            pip install 'selectools[postgres]'
    """

    embedder: "EmbeddingProvider"

    def __init__(
        self,
        embedder: "EmbeddingProvider",
        connection_string: str,
        table_name: str = "selectools_documents",
        dimensions: Optional[int] = None,
    ) -> None:
        """
        Initialize PgVector store.

        Args:
            embedder: Embedding provider to use for computing embeddings.
            connection_string: PostgreSQL connection string
                (e.g. ``"postgresql://user:pass@localhost:5432/mydb"``).
            table_name: Name of the table to store documents in.
                Must be a valid SQL identifier (letters, digits, underscores).
            dimensions: Embedding vector dimensions. If ``None``, auto-detected
                by calling ``embedder.embed_query("test")`` on first use.
        """
        self.embedder = embedder
        self.connection_string = connection_string
        self.table_name = self._validate_table_name(table_name)
        self._dimensions = dimensions
        self._lock = threading.Lock()
        self._initialized = False
        self._psycopg2 = _import_psycopg2()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_table_name(name: str) -> str:
        """
        Validate that the table name is a safe SQL identifier.

        Only allows alphanumeric characters and underscores to prevent
        SQL injection through table names.

        Args:
            name: Proposed table name.

        Returns:
            The validated table name.

        Raises:
            ValueError: If the table name contains invalid characters.
        """
        if not name:
            raise ValueError("Table name must not be empty")
        if not all(c.isalnum() or c == "_" for c in name):
            raise ValueError(
                f"Invalid table name: {name!r}. "
                "Only alphanumeric characters and underscores are allowed."
            )
        if name[0].isdigit():
            raise ValueError(
                f"Invalid table name: {name!r}. " "Table name must not start with a digit."
            )
        return name

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> Any:
        """
        Create a new database connection.

        Returns a ``psycopg2`` connection object. Callers are responsible for
        closing the connection.

        Returns:
            A psycopg2 connection instance.
        """
        return self._psycopg2.connect(self.connection_string)

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _detect_dimensions(self) -> int:
        """
        Auto-detect embedding dimensions by embedding a probe string.

        Returns:
            The number of dimensions produced by the embedder.
        """
        probe = self.embedder.embed_query("dimension probe")
        return len(probe)

    def _ensure_initialized(self) -> None:
        """
        Ensure the pgvector extension, table, and index exist.

        This is idempotent — it only runs once per instance. All DDL
        statements use ``IF NOT EXISTS`` so they are safe to re-run.
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if self._dimensions is None:
                self._dimensions = self._detect_dimensions()

            conn = self._get_connection()
            try:
                conn.autocommit = True
                cursor = conn.cursor()

                # Enable the vector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create the documents table
                # Table name is validated in __init__ — safe to interpolate.
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        embedding vector({self._dimensions}) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """  # nosec — table_name is validated, dimensions is int
                )

                # Create HNSW index for cosine distance
                index_name = f"idx_{self.table_name}_embedding_hnsw"
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.table_name}
                    USING hnsw (embedding vector_cosine_ops)
                    """  # nosec — names are validated
                )

                self._initialized = True
                logger.info(
                    "PgVectorStore initialized: table=%s, dimensions=%d",
                    self.table_name,
                    self._dimensions,
                )
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to the store.

        If ``embeddings`` is ``None``, embeddings are computed automatically
        using the configured embedding provider.

        Args:
            documents: List of documents to add.
            embeddings: Optional pre-computed embeddings (one per document).

        Returns:
            List of generated document IDs.

        Raises:
            ValueError: If the number of embeddings does not match the
                number of documents.
        """
        if not documents:
            return []

        self._ensure_initialized()

        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) does not match "
                f"number of documents ({len(documents)})"
            )

        # Compute embeddings outside the lock (potentially slow network call)
        if embeddings is None:
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        ids = [f"doc_{uuid.uuid4().hex}" for _ in documents]

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            for doc_id, doc, embedding in zip(ids, documents, embeddings):
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
                metadata_json = json.dumps(doc.metadata or {})

                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, text, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,  # nosec — table_name validated
                    (doc_id, doc.text, metadata_json, embedding_str),
                )

            conn.commit()
            logger.debug("Added %d documents to %s", len(ids), self.table_name)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using cosine distance.

        Uses the pgvector ``<=>`` cosine distance operator. Results are sorted
        by similarity (highest first). The similarity score is ``1 - distance``.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filter: Optional metadata filter. Each key-value pair is matched
                against the JSONB ``metadata`` column using the ``@>``
                containment operator.

        Returns:
            List of :class:`SearchResult` objects sorted by similarity.
        """
        self._ensure_initialized()

        embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        # Build query with optional metadata filter
        params: List[Any] = [embedding_str]

        filter_clause = ""
        if filter:
            filter_json = json.dumps(filter)
            filter_clause = "WHERE metadata @> %s::jsonb"
            params.append(filter_json)

        params.extend([embedding_str, top_k])

        # cosine distance: 1 - (a <=> b) gives cosine similarity
        query = f"""
            SELECT id, text, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {self.table_name}
            {filter_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """  # nosec — table_name validated

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        finally:
            conn.close()

        results: List[SearchResult] = []
        for row in rows:
            doc_id, text, metadata_raw, similarity = row

            # Parse metadata — may be dict (psycopg2 auto-parses JSONB)
            # or a JSON string depending on cursor configuration.
            if isinstance(metadata_raw, str):
                metadata = json.loads(metadata_raw)
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw
            else:
                metadata = {}

            results.append(
                SearchResult(
                    document=Document(
                        text=text,
                        metadata=metadata,
                    ),
                    score=float(similarity),
                )
            )

        return results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        self._ensure_initialized()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # Use parameterized IN clause — one placeholder per ID
            placeholders = ",".join(["%s"] * len(ids))
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",  # nosec
                ids,
            )
            conn.commit()
            logger.debug("Deleted %d documents from %s", len(ids), self.table_name)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def clear(self) -> None:
        """Clear all documents from the store."""
        self._ensure_initialized()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.table_name}")  # nosec — validated
            conn.commit()
            logger.debug("Cleared all documents from %s", self.table_name)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Return the number of documents in the store.

        Returns:
            Number of documents.
        """
        self._ensure_initialized()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")  # nosec
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            conn.close()

    def get(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a single document by ID.

        Args:
            doc_id: The document ID.

        Returns:
            The document if found, otherwise ``None``.
        """
        self._ensure_initialized()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, text, metadata FROM {self.table_name} WHERE id = %s",  # nosec
                (doc_id,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        _, text, metadata_raw = row
        if isinstance(metadata_raw, str):
            metadata = json.loads(metadata_raw)
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            metadata = {}

        return Document(text=text, metadata=metadata)

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"PgVectorStore("
            f"table={self.table_name!r}, "
            f"dimensions={self._dimensions}, "
            f"initialized={self._initialized})"
        )


__all__ = ["PgVectorStore"]
