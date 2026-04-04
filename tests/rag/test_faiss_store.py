"""Tests for FAISS vector store implementation."""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Build a mock faiss module so tests run without faiss-cpu installed.
# ---------------------------------------------------------------------------


class _MockIndexFlatIP:
    """Mimics faiss.IndexFlatIP using numpy for inner-product search."""

    def __init__(self, d: int) -> None:
        self.d = d
        self.ntotal = 0
        self._vectors: np.ndarray = np.empty((0, d), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._vectors = np.vstack([self._vectors, vectors]) if self.ntotal > 0 else vectors
        self.ntotal = self._vectors.shape[0]

    def search(self, query: np.ndarray, k: int) -> tuple:
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        k = min(k, self.ntotal)
        if self.ntotal == 0:
            return np.array([[-1.0] * k]), np.array([[-1] * k])
        scores = query @ self._vectors.T  # (1, ntotal)
        top_k = min(k, scores.shape[1])
        indices = np.argsort(-scores[0])[:top_k]
        return scores[:, indices].reshape(1, -1), indices.reshape(1, -1).astype(np.int64)

    def reconstruct_n(self, start: int, n: int, out: np.ndarray) -> None:
        out[:n] = self._vectors[start : start + n]

    def reset(self) -> None:
        self._vectors = np.empty((0, self.d), dtype=np.float32)
        self.ntotal = 0


def _mock_write_index(index: _MockIndexFlatIP, path: str) -> None:
    """Persist mock index vectors to disk as raw numpy bytes."""
    with open(path, "wb") as f:
        np.save(f, index._vectors)


def _mock_read_index(path: str) -> _MockIndexFlatIP:
    """Load mock index vectors from disk."""
    with open(path, "rb") as f:
        vectors = np.load(f)
    idx = _MockIndexFlatIP(vectors.shape[1])
    if vectors.shape[0] > 0:
        idx.add(vectors)
    return idx


@pytest.fixture(autouse=True)
def _patch_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a mock faiss module into sys.modules before each test."""
    mock_faiss = types.ModuleType("faiss")
    mock_faiss.IndexFlatIP = _MockIndexFlatIP  # type: ignore[attr-defined]
    mock_faiss.write_index = _mock_write_index  # type: ignore[attr-defined]
    mock_faiss.read_index = _mock_read_index  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faiss", mock_faiss)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def embedder() -> MagicMock:
    """Return a mock embedding provider that produces deterministic vectors."""

    mock = MagicMock()

    def _embed_texts(texts: List[str]) -> List[List[float]]:
        # Produce a simple deterministic embedding: hash-based
        result = []
        for text in texts:
            h = hash(text) % 10000
            vec = [float(h % (i + 2)) / 10.0 for i in range(4)]
            result.append(vec)
        return result

    mock.embed_texts.side_effect = _embed_texts
    mock.embed_query.side_effect = lambda t: _embed_texts([t])[0]
    return mock


@pytest.fixture()
def store(embedder: MagicMock) -> Any:
    from selectools.rag.stores.faiss import FAISSVectorStore

    return FAISSVectorStore(embedder=embedder, dimension=4)


@pytest.fixture()
def store_no_dim(embedder: MagicMock) -> Any:
    from selectools.rag.stores.faiss import FAISSVectorStore

    return FAISSVectorStore(embedder=embedder)


# ---------------------------------------------------------------------------
# Helper: build some test documents
# ---------------------------------------------------------------------------


def _make_docs(n: int = 3) -> list:
    from selectools.rag.vector_store import Document

    return [Document(text=f"doc_{i}", metadata={"idx": i, "even": i % 2 == 0}) for i in range(n)]


def _make_embeddings(n: int = 3, dim: int = 4) -> List[List[float]]:
    np.random.seed(42)
    return np.random.randn(n, dim).astype(np.float32).tolist()


# ===========================================================================
# add_documents
# ===========================================================================


class TestAddDocuments:
    def test_add_returns_ids(self, store: Any) -> None:
        docs = _make_docs(3)
        ids = store.add_documents(docs, embeddings=_make_embeddings(3))
        assert len(ids) == 3
        assert all(id_.startswith("faiss_") for id_ in ids)

    def test_add_empty_list(self, store: Any) -> None:
        assert store.add_documents([]) == []

    def test_add_computes_embeddings_from_provider(self, store: Any, embedder: MagicMock) -> None:
        docs = _make_docs(2)
        ids = store.add_documents(docs)
        assert len(ids) == 2
        embedder.embed_texts.assert_called_once()

    def test_add_without_embedder_raises(self) -> None:
        from selectools.rag.stores.faiss import FAISSVectorStore

        store = FAISSVectorStore(embedder=None, dimension=4)
        docs = _make_docs(1)
        with pytest.raises(ValueError, match="no embedding provider"):
            store.add_documents(docs)

    def test_add_dimension_mismatch_raises(self, store: Any) -> None:
        from selectools.rag.vector_store import Document

        docs = [Document(text="x")]
        wrong_dim = [[1.0, 2.0]]  # dim=2 but store expects 4
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add_documents(docs, embeddings=wrong_dim)

    def test_add_lazy_index_creation(self, store_no_dim: Any) -> None:
        """Index is created on first add when dimension was not given."""
        assert store_no_dim._index is None
        docs = _make_docs(1)
        store_no_dim.add_documents(docs, embeddings=_make_embeddings(1))
        assert store_no_dim._index is not None
        assert store_no_dim.dimension == 4

    def test_add_increments_counter(self, store: Any) -> None:
        docs1 = _make_docs(2)
        ids1 = store.add_documents(docs1, embeddings=_make_embeddings(2))
        docs2 = _make_docs(3)
        ids2 = store.add_documents(docs2, embeddings=_make_embeddings(3))
        assert len(set(ids1 + ids2)) == 5  # all unique

    def test_add_updates_count(self, store: Any) -> None:
        assert store.count == 0
        store.add_documents(_make_docs(3), embeddings=_make_embeddings(3))
        assert store.count == 3


# ===========================================================================
# search
# ===========================================================================


class TestSearch:
    def test_search_returns_results(self, store: Any) -> None:
        docs = _make_docs(5)
        embs = _make_embeddings(5)
        store.add_documents(docs, embeddings=embs)
        results = store.search(embs[0], top_k=3)
        assert len(results) <= 3
        assert all(hasattr(r, "score") for r in results)

    def test_search_empty_store(self, store: Any) -> None:
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_search_uninitialised_store(self, store_no_dim: Any) -> None:
        results = store_no_dim.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_search_top_k_clamp(self, store: Any) -> None:
        """Requesting more results than documents returns all documents."""
        docs = _make_docs(2)
        embs = _make_embeddings(2)
        store.add_documents(docs, embeddings=embs)
        results = store.search(embs[0], top_k=100)
        assert len(results) == 2

    def test_search_scores_are_floats(self, store: Any) -> None:
        docs = _make_docs(3)
        embs = _make_embeddings(3)
        store.add_documents(docs, embeddings=embs)
        results = store.search(embs[0], top_k=3)
        for r in results:
            assert isinstance(r.score, float)

    def test_search_with_filter(self, store: Any) -> None:
        docs = _make_docs(4)
        embs = _make_embeddings(4)
        store.add_documents(docs, embeddings=embs)
        results = store.search(embs[0], top_k=10, filter={"even": True})
        for r in results:
            assert r.document.metadata["even"] is True

    def test_search_filter_no_match(self, store: Any) -> None:
        docs = _make_docs(3)
        embs = _make_embeddings(3)
        store.add_documents(docs, embeddings=embs)
        results = store.search(embs[0], top_k=10, filter={"nonexistent": True})
        assert results == []

    def test_search_self_similarity_highest(self, store: Any) -> None:
        """The exact same vector should be the top result."""
        from selectools.rag.vector_store import Document

        docs = [Document(text=f"t{i}") for i in range(3)]
        # Use orthogonal-ish vectors for a cleaner test
        embs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        store.add_documents(docs, embeddings=embs)
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0].document.text == "t0"


# ===========================================================================
# delete
# ===========================================================================


class TestDelete:
    def test_delete_reduces_count(self, store: Any) -> None:
        docs = _make_docs(3)
        embs = _make_embeddings(3)
        ids = store.add_documents(docs, embeddings=embs)
        store.delete([ids[0]])
        assert store.count == 2

    def test_delete_removes_correct_document(self, store: Any) -> None:
        from selectools.rag.vector_store import Document

        docs = [Document(text=f"unique_{i}") for i in range(3)]
        embs = _make_embeddings(3)
        ids = store.add_documents(docs, embeddings=embs)
        store.delete([ids[1]])
        remaining_texts = [d.text for d in store._documents]
        assert "unique_1" not in remaining_texts
        assert "unique_0" in remaining_texts
        assert "unique_2" in remaining_texts

    def test_delete_empty_list(self, store: Any) -> None:
        docs = _make_docs(2)
        store.add_documents(docs, embeddings=_make_embeddings(2))
        store.delete([])
        assert store.count == 2

    def test_delete_nonexistent_id(self, store: Any) -> None:
        docs = _make_docs(2)
        store.add_documents(docs, embeddings=_make_embeddings(2))
        store.delete(["nonexistent_id"])
        assert store.count == 2

    def test_delete_all(self, store: Any) -> None:
        ids = store.add_documents(_make_docs(3), embeddings=_make_embeddings(3))
        store.delete(ids)
        assert store.count == 0

    def test_delete_on_uninitialised_store(self, store_no_dim: Any) -> None:
        store_no_dim.delete(["anything"])  # should not raise


# ===========================================================================
# clear
# ===========================================================================


class TestClear:
    def test_clear_empties_store(self, store: Any) -> None:
        store.add_documents(_make_docs(5), embeddings=_make_embeddings(5))
        store.clear()
        assert store.count == 0
        assert len(store._ids) == 0
        assert len(store._documents) == 0

    def test_clear_resets_counter(self, store: Any) -> None:
        store.add_documents(_make_docs(3), embeddings=_make_embeddings(3))
        store.clear()
        assert store._id_counter == 0

    def test_clear_allows_reuse(self, store: Any) -> None:
        store.add_documents(_make_docs(2), embeddings=_make_embeddings(2))
        store.clear()
        ids = store.add_documents(_make_docs(3), embeddings=_make_embeddings(3))
        assert store.count == 3
        assert len(ids) == 3

    def test_clear_uninitialised_store(self, store_no_dim: Any) -> None:
        store_no_dim.clear()
        assert store_no_dim._index is None


# ===========================================================================
# save / load round-trip
# ===========================================================================


class TestPersistence:
    def test_save_and_load(self, store: Any, embedder: MagicMock, tmp_path: Any) -> None:
        docs = _make_docs(3)
        embs = _make_embeddings(3)
        ids = store.add_documents(docs, embeddings=embs)

        base = str(tmp_path / "test_index")
        store.save(base)

        from selectools.rag.stores.faiss import FAISSVectorStore

        loaded = FAISSVectorStore.load(base, embedder=embedder)

        assert loaded.count == 3
        assert loaded._ids == ids
        assert loaded.dimension == 4
        assert [d.text for d in loaded._documents] == [d.text for d in docs]

    def test_save_creates_parent_dirs(self, store: Any, tmp_path: Any) -> None:
        store.add_documents(_make_docs(1), embeddings=_make_embeddings(1))
        deep_path = str(tmp_path / "a" / "b" / "c" / "idx")
        store.save(deep_path)
        assert os.path.exists(f"{deep_path}.index")
        assert os.path.exists(f"{deep_path}.meta.json")

    def test_save_uninitialised_raises(self, store_no_dim: Any) -> None:
        with pytest.raises(RuntimeError, match="not been initialised"):
            store_no_dim.save("/tmp/nope")

    def test_load_missing_index_raises(self, tmp_path: Any) -> None:
        from selectools.rag.stores.faiss import FAISSVectorStore

        with pytest.raises(FileNotFoundError, match="index file"):
            FAISSVectorStore.load(str(tmp_path / "missing"))

    def test_load_missing_meta_raises(self, store: Any, tmp_path: Any) -> None:
        from selectools.rag.stores.faiss import FAISSVectorStore

        # Create only the .index file
        store.add_documents(_make_docs(1), embeddings=_make_embeddings(1))
        base = str(tmp_path / "partial")
        store.save(base)
        os.remove(f"{base}.meta.json")

        with pytest.raises(FileNotFoundError, match="Metadata file"):
            FAISSVectorStore.load(base)

    def test_load_preserves_metadata(self, store: Any, tmp_path: Any) -> None:
        from selectools.rag.vector_store import Document

        docs = [Document(text="hello", metadata={"key": "value", "num": 42})]
        store.add_documents(docs, embeddings=_make_embeddings(1))

        base = str(tmp_path / "meta_test")
        store.save(base)

        from selectools.rag.stores.faiss import FAISSVectorStore

        loaded = FAISSVectorStore.load(base)
        assert loaded._documents[0].metadata == {"key": "value", "num": 42}

    def test_loaded_store_is_searchable(self, store: Any, tmp_path: Any) -> None:
        from selectools.rag.vector_store import Document

        docs = [Document(text=f"t{i}") for i in range(3)]
        embs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        store.add_documents(docs, embeddings=embs)

        base = str(tmp_path / "search_test")
        store.save(base)

        from selectools.rag.stores.faiss import FAISSVectorStore

        loaded = FAISSVectorStore.load(base)
        results = loaded.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].document.text == "t0"


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_adds(self, embedder: MagicMock) -> None:
        from selectools.rag.stores.faiss import FAISSVectorStore

        store = FAISSVectorStore(embedder=embedder, dimension=4)
        errors: List[Exception] = []

        def _add_batch(batch_id: int) -> None:
            try:
                docs = _make_docs(10)
                embs = _make_embeddings(10)
                store.add_documents(docs, embeddings=embs)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_add_batch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in concurrent adds: {errors}"
        assert store.count == 50

    def test_concurrent_reads_during_write(self, embedder: MagicMock) -> None:
        from selectools.rag.stores.faiss import FAISSVectorStore

        store = FAISSVectorStore(embedder=embedder, dimension=4)
        store.add_documents(_make_docs(10), embeddings=_make_embeddings(10))
        errors: List[Exception] = []

        def _search() -> None:
            try:
                for _ in range(20):
                    store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
            except Exception as exc:
                errors.append(exc)

        def _add() -> None:
            try:
                for _ in range(10):
                    store.add_documents(_make_docs(1), embeddings=_make_embeddings(1))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_search) for _ in range(3)]
        threads.append(threading.Thread(target=_add))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in concurrent read/write: {errors}"

    def test_concurrent_delete_and_search(self, embedder: MagicMock) -> None:
        from selectools.rag.stores.faiss import FAISSVectorStore

        store = FAISSVectorStore(embedder=embedder, dimension=4)
        ids = store.add_documents(_make_docs(20), embeddings=_make_embeddings(20))
        errors: List[Exception] = []

        def _delete() -> None:
            try:
                for doc_id in ids[:10]:
                    store.delete([doc_id])
            except Exception as exc:
                errors.append(exc)

        def _search() -> None:
            try:
                for _ in range(20):
                    store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_delete), threading.Thread(target=_search)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in concurrent delete/search: {errors}"


# ===========================================================================
# Introspection / misc
# ===========================================================================


class TestIntrospection:
    def test_len(self, store: Any) -> None:
        assert len(store) == 0
        store.add_documents(_make_docs(2), embeddings=_make_embeddings(2))
        assert len(store) == 2

    def test_repr(self, store: Any) -> None:
        r = repr(store)
        assert "FAISSVectorStore" in r
        assert "dimension=4" in r
        assert "count=0" in r

    def test_stability_marker(self) -> None:
        from selectools.rag.stores.faiss import FAISSVectorStore

        assert getattr(FAISSVectorStore, "__stability__", None) == "beta"

    def test_inherits_vector_store(self) -> None:
        from selectools.rag.stores.faiss import FAISSVectorStore
        from selectools.rag.vector_store import VectorStore

        assert issubclass(FAISSVectorStore, VectorStore)
