"""SQLite vector store implementation for persistent local storage."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...embeddings.provider import EmbeddingProvider

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "numpy required for SQLite vector store. Install with: pip install numpy"
    ) from e

from ..vector_store import Document, SearchResult, VectorStore, _dedup_search_results


class SQLiteVectorStore(VectorStore):
    """
    SQLite-based vector store for persistent local storage.

    Stores documents and their embeddings in an SQLite database. Embeddings are
    stored as JSON arrays. For production use with large datasets, consider using
    sqlite-vss extension for faster similarity search.

    Example:
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag.stores import SQLiteVectorStore
        >>>
        >>> embedder = OpenAIEmbeddingProvider()
        >>> store = SQLiteVectorStore(embedder, db_path="my_vectors.db")
        >>>
        >>> # Add documents (persisted to disk)
        >>> docs = [Document(text="Hello world", metadata={"source": "test"})]
        >>> ids = store.add_documents(docs)
        >>>
        >>> # Search (loads from disk)
        >>> query_emb = embedder.embed_query("hi")
        >>> results = store.search(query_emb, top_k=1)
    """

    embedder: "EmbeddingProvider"

    def __init__(self, embedder: "EmbeddingProvider", db_path: str = "vector_store.db") -> None:  # noqa: F821
        """
        Initialize SQLite vector store.

        Args:
            embedder: Embedding provider to use for computing embeddings
            db_path: Path to SQLite database file (will be created if doesn't exist)
        """
        if db_path == ":memory:":
            raise ValueError(
                "SQLiteVectorStore does not support ':memory:' databases. "
                "Each sqlite3.connect(':memory:') call creates a new empty database, "
                "so documents written in one operation are invisible to the next. "
                "Use a file path (e.g. db_path='vectors.db') or InMemoryVectorStore instead."
            )
        self.embedder = embedder
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()

                # Create documents table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        metadata TEXT,
                        embedding TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Create index on metadata for faster filtering
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_metadata ON documents(metadata)
                """
                )

                conn.commit()
            finally:
                conn.close()

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

        # Compute embeddings outside the lock (potentially slow network call)
        if embeddings is None:
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        # UUID-based IDs: unique per insertion regardless of content or batch order,
        # matching InMemoryVectorStore's counter-based scheme.
        ids = [f"doc_{uuid.uuid4().hex}" for _ in documents]

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                for doc_id, doc, embedding in zip(ids, documents, embeddings):
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO documents (id, text, metadata, embedding)
                        VALUES (?, ?, ?, ?)
                    """,
                        (doc_id, doc.text, json.dumps(doc.metadata), json.dumps(embedding)),
                    )
                conn.commit()
            finally:
                conn.close()

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        dedup: bool = False,
    ) -> List[SearchResult]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter
            dedup: If True, drop duplicate-text results (keeps highest-scoring).

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id, text, metadata, embedding FROM documents")
                rows = cursor.fetchall()
            finally:
                conn.close()

        if not rows:
            return []

        # Apply metadata filter in Python (flexible matching).
        # Guard against NULL metadata stored by external writers:
        #   - SQL NULL arrives as Python None → json.loads(None) raises TypeError.
        #   - JSON string "null" → json.loads("null") returns Python None (not a dict).
        # Both cases are handled by the helper below.
        def _safe_meta(raw: object) -> dict:
            if raw is None:
                return {}
            parsed = json.loads(raw)  # type: ignore[arg-type]
            return parsed if isinstance(parsed, dict) else {}

        if filter:
            rows = [
                row
                for row in rows
                if all(_safe_meta(row[2]).get(k) == v for k, v in filter.items())
            ]

        if not rows:
            return []

        # Compute similarities
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        results = []
        for row in rows:
            doc_id, text, metadata_json, embedding_json = row
            metadata = _safe_meta(metadata_json)

            # Guard against NULL or JSON-null embedding stored by external writers.
            # json.loads('null') returns None and np.array(None, ...) raises TypeError.
            if embedding_json is None:
                continue
            parsed_embedding = json.loads(embedding_json)
            if not isinstance(parsed_embedding, list) or len(parsed_embedding) == 0:
                continue
            embedding = np.array(parsed_embedding, dtype=np.float32)

            doc_norm = np.linalg.norm(embedding)
            if doc_norm == 0:
                continue

            similarity = np.dot(embedding, query_vec) / (doc_norm * query_norm)
            results.append(
                SearchResult(
                    document=Document(text=text, metadata=metadata, embedding=embedding.tolist()),
                    score=float(similarity),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        if dedup:
            results = _dedup_search_results(results)
        return results[:top_k]

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            return

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                placeholders = ",".join("?" for _ in ids)
                cursor.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", ids)  # nosec
                conn.commit()
            finally:
                conn.close()

    def clear(self) -> None:
        """Clear all documents from the store."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents")
                conn.commit()
            finally:
                conn.close()


__all__ = ["SQLiteVectorStore"]
