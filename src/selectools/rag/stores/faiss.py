"""FAISS vector store implementation for fast local similarity search."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ...embeddings.provider import EmbeddingProvider

from ...stability import beta
from ..vector_store import Document, SearchResult, VectorStore


def _import_faiss() -> Any:
    """Lazy import of faiss-cpu with a helpful error message."""
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss-cpu package required for FAISS vector store. "
            "Install with: pip install faiss-cpu"
        ) from e
    return faiss


def _import_numpy() -> Any:
    """Lazy import of numpy with a helpful error message."""
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy package required for FAISS vector store. " "Install with: pip install numpy"
        ) from e
    return np


@beta
class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store for fast local similarity search.

    Uses FAISS IndexFlatIP (inner product) with L2-normalized vectors to
    perform exact cosine similarity search. Documents are stored in a parallel
    list since FAISS only manages raw vectors.

    Thread-safe: all write operations are protected by a threading.Lock.
    Read operations (search) take a snapshot under the lock, then compute
    outside it.

    Persistence is supported via ``save()`` / ``load()`` which use FAISS
    native ``write_index`` / ``read_index`` plus a JSON sidecar for documents.

    Example:
        >>> from selectools.embeddings import OpenAIEmbeddingProvider
        >>> from selectools.rag.stores import FAISSVectorStore
        >>>
        >>> embedder = OpenAIEmbeddingProvider()
        >>> store = FAISSVectorStore(embedder, dimension=1536)
        >>>
        >>> # Add documents
        >>> docs = [Document(text="Hello world", metadata={"source": "test"})]
        >>> ids = store.add_documents(docs)
        >>>
        >>> # Search
        >>> query_emb = embedder.embed_query("hi")
        >>> results = store.search(query_emb, top_k=1)
        >>>
        >>> # Persist and reload
        >>> store.save("/tmp/my_index")
        >>> loaded = FAISSVectorStore.load("/tmp/my_index", embedder)
    """

    embedder: Optional["EmbeddingProvider"]

    def __init__(
        self,
        embedder: Optional["EmbeddingProvider"] = None,
        dimension: Optional[int] = None,
    ) -> None:
        """
        Initialize FAISS vector store.

        The FAISS index is created lazily on the first ``add_documents`` call
        if *dimension* is not provided up front. When *dimension* is given, the
        index is allocated immediately, which lets ``search`` return an empty
        list instead of raising on an uninitialised store.

        Args:
            embedder: Embedding provider for computing embeddings.  May be
                ``None`` when loading a persisted store that already contains
                pre-computed vectors.
            dimension: Embedding vector dimension.  If ``None``, inferred from
                the first batch of added documents.
        """
        self.embedder = embedder
        self._dimension = dimension
        self._index: Any = None  # faiss.IndexFlatIP — created lazily or on load
        self._documents: List[Document] = []
        self._ids: List[str] = []
        self._id_counter: int = 0
        self._lock = threading.Lock()

        if dimension is not None:
            self._init_index(dimension)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_index(self, dimension: int) -> None:
        """Create a fresh FAISS IndexFlatIP with the given dimension."""
        faiss = _import_faiss()
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)

    @staticmethod
    def _normalize(vectors: Any) -> Any:
        """L2-normalize vectors so inner product == cosine similarity.

        Args:
            vectors: numpy array of shape ``(n, d)``.

        Returns:
            A new array with each row L2-normalised.  Rows with zero norm
            are left as-is (all zeros) to avoid division by zero.
        """
        np = _import_numpy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Replace zero norms with 1 to avoid NaN
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _generate_id(self, doc: Document, index: int) -> str:
        """Generate a deterministic document ID."""
        content_hash = hashlib.sha256(doc.text.encode()).hexdigest()[:16]
        return f"faiss_{content_hash}_{self._id_counter + index}"

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to the FAISS index.

        Vectors are L2-normalized before insertion so that inner-product
        scores equal cosine similarity.

        Args:
            documents: Documents to add.
            embeddings: Pre-computed embeddings.  If ``None``, computed via
                the store's embedding provider.

        Returns:
            List of string IDs assigned to the documents.

        Raises:
            ValueError: If no embeddings are provided and no embedder is set.
            ValueError: If embedding dimension does not match the index.
        """
        if not documents:
            return []

        np = _import_numpy()

        # Compute embeddings outside the lock (potentially slow I/O)
        if embeddings is None:
            if self.embedder is None:
                raise ValueError(
                    "Cannot compute embeddings: no embedding provider set. "
                    "Either pass embeddings explicitly or provide an embedder."
                )
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        vectors = self._normalize(vectors)

        with self._lock:
            # Lazy index creation on first add
            if self._index is None:
                self._init_index(vectors.shape[1])

            # Dimension validation
            if vectors.shape[1] != self._dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: got {vectors.shape[1]}, "
                    f"expected {self._dimension}"
                )

            # Generate IDs
            new_ids = [self._generate_id(doc, i) for i, doc in enumerate(documents)]
            self._id_counter += len(documents)

            # Add to FAISS index
            self._index.add(vectors)

            # Store documents and IDs in parallel lists
            self._documents.extend(documents)
            self._ids.extend(new_ids)

        return new_ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using cosine similarity.

        The query vector is L2-normalized so that the FAISS inner-product
        score equals cosine similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filter: Optional metadata filter dict.  All key-value pairs must
                match for a document to be included.

        Returns:
            List of ``SearchResult`` objects sorted by descending similarity.
        """
        np = _import_numpy()

        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            # Snapshot under lock
            index_snapshot = self._index
            docs_snapshot = list(self._documents)
            ids_snapshot = list(self._ids)
            n_total = self._index.ntotal

        # Prepare query vector
        query_vec = np.array([query_embedding], dtype=np.float32)
        query_vec = self._normalize(query_vec)

        # Over-fetch when filtering to compensate for filtered-out docs
        fetch_k = min(top_k * 4, n_total) if filter else min(top_k, n_total)

        # FAISS search returns (distances, indices) arrays of shape (1, fetch_k)
        scores, indices = index_snapshot.search(query_vec, fetch_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS returns -1 for empty slots
            if idx < 0:
                continue

            doc = docs_snapshot[idx]

            # Apply metadata filter
            if filter and not self._matches_filter(doc, filter):
                continue

            results.append(SearchResult(document=doc, score=float(score)))

            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Because FAISS ``IndexFlatIP`` does not support in-place deletion, the
        entire index is rebuilt from the remaining vectors.  This is O(n) but
        keeps the implementation simple and correct.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        np = _import_numpy()
        faiss = _import_faiss()

        with self._lock:
            if self._index is None:
                return

            ids_to_remove = set(ids)
            keep_mask = [i for i, doc_id in enumerate(self._ids) if doc_id not in ids_to_remove]

            if len(keep_mask) == len(self._ids):
                return  # nothing to delete

            # Reconstruct vectors for the kept documents
            if keep_mask:
                n_total = self._index.ntotal
                all_vectors = np.zeros((n_total, self._dimension), dtype=np.float32)
                self._index.reconstruct_n(0, n_total, all_vectors)

                kept_vectors = all_vectors[keep_mask]
                self._documents = [self._documents[i] for i in keep_mask]
                self._ids = [self._ids[i] for i in keep_mask]

                # Rebuild the index
                new_index = faiss.IndexFlatIP(self._dimension)
                new_index.add(kept_vectors)
                self._index = new_index
            else:
                # All documents deleted
                self._documents = []
                self._ids = []
                self._index = faiss.IndexFlatIP(self._dimension)

    def clear(self) -> None:
        """
        Remove all documents and reset the index.

        The index is recreated with the same dimension so subsequent
        ``add_documents`` calls succeed without re-inferring the dimension.
        """
        faiss = _import_faiss()

        with self._lock:
            self._documents = []
            self._ids = []
            self._id_counter = 0
            if self._dimension is not None:
                self._index = faiss.IndexFlatIP(self._dimension)
            else:
                self._index = None

    # ------------------------------------------------------------------
    # Metadata filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_filter(doc: Document, filter: Dict[str, Any]) -> bool:
        """Check whether a document's metadata matches all filter criteria.

        Args:
            doc: The document to check.
            filter: Key-value pairs that must all appear in ``doc.metadata``.

        Returns:
            ``True`` if every filter key is present and equal in the document's
            metadata.
        """
        for key, value in filter.items():
            if doc.metadata.get(key) != value:
                return False
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the FAISS index and document metadata to disk.

        Creates two files:
        - ``<path>.index`` — FAISS binary index (via ``faiss.write_index``)
        - ``<path>.meta.json`` — JSON with documents, IDs, dimension, counter

        Args:
            path: Base path (without extension).  Parent directories are
                created automatically.

        Raises:
            RuntimeError: If the index has not been initialised yet.
        """
        faiss = _import_faiss()

        with self._lock:
            if self._index is None:
                raise RuntimeError(
                    "Cannot save: FAISS index has not been initialised. "
                    "Add documents first or specify a dimension."
                )

            # Ensure parent directory exists
            parent = Path(path).parent
            parent.mkdir(parents=True, exist_ok=True)

            # Write FAISS index
            index_path = f"{path}.index"
            faiss.write_index(self._index, index_path)

            # Write metadata sidecar
            meta = {
                "dimension": self._dimension,
                "id_counter": self._id_counter,
                "ids": self._ids,
                "documents": [
                    {"text": doc.text, "metadata": doc.metadata} for doc in self._documents
                ],
            }
            meta_path = f"{path}.meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        path: str,
        embedder: Optional["EmbeddingProvider"] = None,
    ) -> "FAISSVectorStore":
        """
        Load a persisted FAISS index and document metadata from disk.

        Args:
            path: Base path used in a previous ``save()`` call.
            embedder: Optional embedding provider to attach to the loaded store.

        Returns:
            A new ``FAISSVectorStore`` instance with the restored index and
            documents.

        Raises:
            FileNotFoundError: If the index or metadata files are missing.
        """
        faiss = _import_faiss()

        index_path = f"{path}.index"
        meta_path = f"{path}.meta.json"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        # Read FAISS index
        index = faiss.read_index(index_path)

        # Read metadata sidecar
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Reconstruct store
        store = cls(embedder=embedder, dimension=meta["dimension"])
        store._index = index
        store._id_counter = meta["id_counter"]
        store._ids = meta["ids"]
        store._documents = [
            Document(text=d["text"], metadata=d.get("metadata", {})) for d in meta["documents"]
        ]

        return store

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Return the number of documents stored in the index."""
        with self._lock:
            return len(self._documents)

    @property
    def dimension(self) -> Optional[int]:
        """Return the vector dimension, or ``None`` if not yet initialised."""
        return self._dimension

    def __len__(self) -> int:
        """Return number of stored documents."""
        return self.count

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FAISSVectorStore(dimension={self._dimension}, " f"count={len(self._documents)})"


__all__ = ["FAISSVectorStore"]
