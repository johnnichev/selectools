"""SQLite vector store implementation for persistent local storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...embeddings.provider import EmbeddingProvider

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "numpy required for SQLite vector store. Install with: pip install numpy"
    ) from e

from ..vector_store import Document, SearchResult, VectorStore


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

    def __init__(
        self, embedder: "EmbeddingProvider", db_path: str = "vector_store.db"
    ) -> None:  # noqa: F821
        """
        Initialize SQLite vector store.

        Args:
            embedder: Embedding provider to use for computing embeddings
            db_path: Path to SQLite database file (will be created if doesn't exist)
        """
        self.embedder = embedder
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
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

        # Compute embeddings if not provided
        if embeddings is None:
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate IDs and insert documents
        ids = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{hash(doc.text)}_{i}"
            ids.append(doc_id)

            # Serialize data
            metadata_json = json.dumps(doc.metadata)
            embedding_json = json.dumps(embedding)

            # Insert into database
            cursor.execute(
                """
                INSERT OR REPLACE INTO documents (id, text, metadata, embedding)
                VALUES (?, ?, ?, ?)
            """,
                (doc_id, doc.text, metadata_json, embedding_json),
            )

        conn.commit()
        conn.close()

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
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Fetch all documents (with optional metadata filtering)
        if filter:
            # Simple metadata filtering (exact match on JSON string)
            cursor.execute("SELECT id, text, metadata, embedding FROM documents")
            rows = cursor.fetchall()
            # Filter in Python for more flexible matching
            filtered_rows = []
            for row in rows:
                metadata = json.loads(row[2])
                if all(metadata.get(k) == v for k, v in filter.items()):
                    filtered_rows.append(row)
            rows = filtered_rows
        else:
            cursor.execute("SELECT id, text, metadata, embedding FROM documents")
            rows = cursor.fetchall()

        conn.close()

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
            metadata = json.loads(metadata_json)
            embedding = np.array(json.loads(embedding_json), dtype=np.float32)

            # Cosine similarity
            doc_norm = np.linalg.norm(embedding)
            if doc_norm == 0:
                continue

            similarity = np.dot(embedding, query_vec) / (doc_norm * query_norm)

            doc = Document(text=text, metadata=metadata, embedding=embedding.tolist())
            results.append(SearchResult(document=doc, score=float(similarity)))

        # Sort by similarity (highest first) and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete documents
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", ids)  # nosec

        conn.commit()
        conn.close()

    def clear(self) -> None:
        """Clear all documents from the store."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM documents")

        conn.commit()
        conn.close()


__all__ = ["SQLiteVectorStore"]
