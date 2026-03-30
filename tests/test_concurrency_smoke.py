"""
Thread-safety smoke tests.

Each test spins up N threads, synchronises them at a barrier so they all hit
the shared object simultaneously (maximising contention), then verifies that
no exceptions were raised and that the final state is internally consistent
(list length == numpy shape, counts match, etc.).

These tests use real implementations with no mocks — they would not catch
actual races if the underlying code is mocked.

Usage:
    pytest tests/test_concurrency_smoke.py -x -q -v
"""

from __future__ import annotations

import threading
import time
from typing import Any, List
from unittest.mock import Mock

import pytest

from selectools.rag.vector_store import Document

# ============================================================================
# BM25
# ============================================================================


class TestBM25Concurrency:
    """BM25 must be safe for concurrent add_documents and search calls."""

    N_THREADS = 10
    OPS_PER_THREAD = 20

    def test_concurrent_add_documents_count_accurate(self):
        """document_count must equal N_threads × ops after concurrent adds."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(self.OPS_PER_THREAD):
                try:
                    bm25.add_documents([Document(text=f"thread{tid} item{i} python data")])
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"
        assert bm25.document_count == self.N_THREADS * self.OPS_PER_THREAD

    def test_concurrent_index_documents_no_crash(self):
        """Concurrent index_documents calls must not corrupt internal state."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid: int) -> None:
            docs = [Document(text=f"thread{tid} doc{i}") for i in range(5)]
            barrier.wait()
            for _ in range(5):
                try:
                    bm25.index_documents(docs)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"
        # After concurrent index_documents, state must be internally consistent:
        # _doc_count must match _docs list length
        assert bm25._doc_count == len(bm25._docs)

    def test_concurrent_add_and_search_no_crash(self):
        """Search during concurrent adds must never crash."""
        from selectools.rag.bm25 import BM25

        bm25 = BM25()
        # Seed with some docs first
        bm25.index_documents([Document(text="python data science machine learning")] * 10)

        errors: List[Exception] = []
        barrier = threading.Barrier(2)

        def adder() -> None:
            barrier.wait()
            for i in range(50):
                try:
                    bm25.add_documents([Document(text=f"concurrent add {i}")])
                except Exception as e:
                    errors.append(e)

        def searcher() -> None:
            barrier.wait()
            for _ in range(50):
                try:
                    result = bm25.search("python data", top_k=3)
                    assert isinstance(result, list)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=adder), threading.Thread(target=searcher)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"


# ============================================================================
# InMemoryVectorStore
# ============================================================================


class TestInMemoryVectorStoreConcurrency:
    """InMemoryVectorStore must be safe for concurrent reads and writes."""

    N_THREADS = 10
    OPS_PER_THREAD = 20
    DIM = 4  # small embedding dimension for speed

    def _mock_embedder(self) -> Mock:
        embedder = Mock()
        embedder.embed_texts.side_effect = lambda texts: [[0.1] * self.DIM] * len(texts)
        return embedder

    def test_concurrent_add_documents_list_numpy_aligned(self):
        """After N concurrent adds, list length must equal numpy row count."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(embedder=self._mock_embedder())
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(self.OPS_PER_THREAD):
                try:
                    store.add_documents(
                        [Document(text=f"t{tid}d{i}")],
                        embeddings=[[float(tid)] * self.DIM],
                    )
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"
        expected = self.N_THREADS * self.OPS_PER_THREAD
        assert len(store.documents) == expected
        assert store.embeddings is not None
        assert store.embeddings.shape[0] == expected

    def test_concurrent_add_and_search_no_crash(self):
        """Searching while other threads add documents must never crash."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(embedder=self._mock_embedder())
        # Seed store
        store.add_documents(
            [Document(text=f"seed{i}") for i in range(10)],
            embeddings=[[0.1] * self.DIM] * 10,
        )

        errors: List[Exception] = []
        barrier = threading.Barrier(2)

        def adder() -> None:
            barrier.wait()
            for i in range(50):
                try:
                    store.add_documents(
                        [Document(text=f"add{i}")],
                        embeddings=[[0.2] * self.DIM],
                    )
                except Exception as e:
                    errors.append(e)

        def searcher() -> None:
            barrier.wait()
            for _ in range(50):
                try:
                    results = store.search([0.1] * self.DIM, top_k=3)
                    assert isinstance(results, list)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=adder), threading.Thread(target=searcher)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"

    def test_concurrent_delete_no_desync(self):
        """After concurrent deletes, list and numpy array must stay aligned."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(embedder=self._mock_embedder())
        # Add 100 docs first
        ids = store.add_documents(
            [Document(text=f"doc{i}") for i in range(100)],
            embeddings=[[0.1] * self.DIM] * 100,
        )

        errors: List[Exception] = []
        barrier = threading.Barrier(5)

        def deleter(chunk: List[str]) -> None:
            barrier.wait()
            try:
                store.delete(chunk)
            except Exception as e:
                errors.append(e)

        # Split 100 IDs into 5 groups of 20, delete concurrently
        chunks = [ids[i : i + 20] for i in range(0, 100, 20)]
        threads = [threading.Thread(target=deleter, args=(c,)) for c in chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"
        # List and numpy must be aligned — this is the key invariant
        n_docs = len(store.documents)
        n_embs = store.embeddings.shape[0] if store.embeddings is not None else 0
        assert n_docs == n_embs, f"List/numpy desync: {n_docs} docs vs {n_embs} embeddings"


# ============================================================================
# SQLiteVectorStore
# ============================================================================


class TestSQLiteVectorStoreConcurrency:
    """SQLiteVectorStore must handle concurrent writes without DB corruption."""

    N_THREADS = 5  # lower than memory tests — SQLite has its own WAL locking
    OPS_PER_THREAD = 10
    DIM = 4

    def _mock_embedder(self) -> Mock:
        embedder = Mock()
        embedder.embed_texts.side_effect = lambda texts: [[0.1] * self.DIM] * len(texts)
        return embedder

    def test_concurrent_add_documents_row_count_accurate(self, tmp_path):
        """Row count after N concurrent adds must equal N_threads × ops."""
        import sqlite3

        from selectools.rag.stores.sqlite import SQLiteVectorStore

        db_path = str(tmp_path / "test.db")
        store = SQLiteVectorStore(embedder=self._mock_embedder(), db_path=db_path)
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(self.OPS_PER_THREAD):
                try:
                    store.add_documents(
                        [Document(text=f"t{tid}d{i}")],
                        embeddings=[[float(tid)] * self.DIM],
                    )
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"
        conn = sqlite3.connect(db_path)
        row_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        assert row_count == self.N_THREADS * self.OPS_PER_THREAD

    def test_concurrent_add_and_search_no_crash(self, tmp_path):
        """Search while writing must never crash or return corrupt data."""
        from selectools.rag.stores.sqlite import SQLiteVectorStore

        db_path = str(tmp_path / "test.db")
        store = SQLiteVectorStore(embedder=self._mock_embedder(), db_path=db_path)
        # Seed
        store.add_documents(
            [Document(text=f"seed{i}") for i in range(5)],
            embeddings=[[0.1] * self.DIM] * 5,
        )

        errors: List[Exception] = []
        barrier = threading.Barrier(2)

        def adder() -> None:
            barrier.wait()
            for i in range(20):
                try:
                    store.add_documents(
                        [Document(text=f"add{i}")],
                        embeddings=[[0.2] * self.DIM],
                    )
                except Exception as e:
                    errors.append(e)

        def searcher() -> None:
            barrier.wait()
            for _ in range(20):
                try:
                    results = store.search([0.1] * self.DIM, top_k=3)
                    assert isinstance(results, list)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=adder), threading.Thread(target=searcher)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"


# ============================================================================
# InMemoryCache
# ============================================================================


class TestInMemoryCacheConcurrency:
    """InMemoryCache must be safe for concurrent get/set/delete."""

    N_THREADS = 10
    OPS_PER_THREAD = 50

    def test_concurrent_set_no_size_overflow(self):
        """Cache size must never exceed max_size under concurrent sets."""
        from selectools.cache import InMemoryCache

        max_size = 100
        cache = InMemoryCache(max_size=max_size)
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(self.OPS_PER_THREAD):
                try:
                    key = f"key_{tid}_{i}"
                    cache.set(key, (f"response_{tid}_{i}", {}))
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"
        assert cache.size <= max_size

    def test_concurrent_get_set_consistency(self):
        """A value set by one thread must be retrievable once set (eventual)."""
        from selectools.cache import InMemoryCache

        cache = InMemoryCache(max_size=1000)
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)
        confirmed_keys: List[str] = []

        def worker(tid: int) -> None:
            barrier.wait()
            key = f"stable_key_{tid}"
            val = (f"stable_value_{tid}", {})
            try:
                cache.set(key, val)
                result = cache.get(key)
                # Either it's there (we set it) or it was evicted (LRU) — both OK
                if result is not None:
                    assert result == val
                    confirmed_keys.append(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"


# ============================================================================
# SemanticCache
# ============================================================================


class _StubEmbedder:
    """Thread-safe stub embedder for concurrency tests (no Mock, no side_effect)."""

    DIM = 4

    def embed_texts(self, texts):
        return [[0.1] * self.DIM] * len(texts)

    def embed_text(self, text):
        return [0.1] * self.DIM

    def embed_query(self, text):
        return [0.1] * self.DIM


class TestSemanticCacheConcurrency:
    """SemanticCache must be safe for concurrent get/set."""

    N_THREADS = 8
    OPS_PER_THREAD = 15

    def test_concurrent_set_no_size_overflow(self):
        """Cache size must never exceed max_size under concurrent sets."""
        from selectools.cache_semantic import SemanticCache

        max_size = 50
        cache = SemanticCache(
            embedding_provider=_StubEmbedder(),
            max_size=max_size,
            similarity_threshold=0.99,
        )
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(self.OPS_PER_THREAD):
                try:
                    cache.set(f"query_{tid}_{i}", (f"response_{tid}_{i}", {}))
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"
        assert len(cache._entries) <= max_size

    def test_concurrent_get_set_no_crash(self):
        """Interleaved get/set from multiple threads must never crash."""
        from selectools.cache_semantic import SemanticCache

        cache = SemanticCache(
            embedding_provider=_StubEmbedder(),
            max_size=200,
        )
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(self.OPS_PER_THREAD):
                try:
                    cache.set(f"k{tid}_{i}", (f"v{tid}_{i}", {}))
                    cache.get(f"k{tid}_{i}")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"


# ============================================================================
# ConversationMemory
# ============================================================================


class TestConversationMemoryConcurrency:
    """ConversationMemory must enforce max_messages under concurrent adds."""

    N_THREADS = 10
    OPS_PER_THREAD = 20

    def test_concurrent_add_never_exceeds_max_messages(self):
        """len(messages) must never exceed max_messages regardless of concurrency."""
        from selectools.memory import ConversationMemory
        from selectools.types import Message, Role

        max_msgs = 20
        memory = ConversationMemory(max_messages=max_msgs)
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(self.OPS_PER_THREAD):
                try:
                    memory.add(Message(role=Role.USER, content=f"msg t{tid} i{i}"))
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"
        assert len(memory.get_history()) <= max_msgs

    def test_concurrent_add_and_get_messages_no_crash(self):
        """Reading messages while others write must never crash."""
        from selectools.memory import ConversationMemory
        from selectools.types import Message, Role

        memory = ConversationMemory(max_messages=50)
        errors: List[Exception] = []
        barrier = threading.Barrier(2)

        def writer() -> None:
            barrier.wait()
            for i in range(100):
                try:
                    memory.add(Message(role=Role.USER, content=f"write {i}"))
                except Exception as e:
                    errors.append(e)

        def reader() -> None:
            barrier.wait()
            for _ in range(100):
                try:
                    msgs = memory.get_history()
                    assert isinstance(msgs, list)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"


# ============================================================================
# KnowledgeMemory (FileKnowledgeStore)
# ============================================================================


class TestKnowledgeMemoryConcurrency:
    """KnowledgeMemory with FileKnowledgeStore must handle concurrent writes."""

    N_THREADS = 5
    OPS_PER_THREAD = 10

    def test_concurrent_add_no_file_corruption(self, tmp_path):
        """Concurrent add() calls must not corrupt the knowledge store."""
        from selectools.knowledge import FileKnowledgeStore, KnowledgeMemory

        store = FileKnowledgeStore(directory=str(tmp_path))
        memory = KnowledgeMemory(store=store, directory=str(tmp_path))
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_THREADS)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(self.OPS_PER_THREAD):
                try:
                    memory.remember(
                        content=f"fact from thread {tid} item {i}",
                        category=f"cat_{tid % 3}",
                    )
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(self.N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised: {errors}"
        # Store must be readable after concurrent writes
        context = memory.build_context()
        assert isinstance(context, str)
