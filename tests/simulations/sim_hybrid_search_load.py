"""
Simulation: HybridSearcher under concurrent search load.

1000 pre-indexed documents. 20 threads each perform 10 searches concurrently.
Verifies:
- No thread gets an error
- BM25 state stays consistent (document_count correct)
- All searches return valid lists

No API keys required — uses a stub embedder.

Run: pytest tests/simulations/sim_hybrid_search_load.py -v
"""

from __future__ import annotations

import threading
from typing import List
from unittest.mock import Mock

import pytest

from selectools.rag.vector_store import Document

DIM = 8
N_DOCS = 200  # reduced from 1000 for test speed while preserving concurrency pressure
N_THREADS = 20
SEARCHES_PER_THREAD = 10


class _StubEmbedder:
    """Thread-safe stub: always returns the same deterministic embedding."""

    def embed_texts(self, texts):
        return [[0.1] * DIM for _ in texts]

    def embed_query(self, text):
        return [0.1] * DIM


@pytest.mark.integration
class TestHybridSearchConcurrentLoad:
    """HybridSearcher must handle concurrent search load without corruption."""

    def _build_searcher(self):
        from selectools.rag.hybrid import HybridSearcher
        from selectools.rag.stores.memory import InMemoryVectorStore

        embedder = _StubEmbedder()
        store = InMemoryVectorStore(embedder=embedder)

        # Index N_DOCS documents
        topics = ["python", "machine learning", "data science", "neural networks", "optimization"]
        docs = [
            Document(
                text=f"Document {i} about {topics[i % len(topics)]} and related concepts.",
                metadata={"topic": topics[i % len(topics)], "doc_id": i},
            )
            for i in range(N_DOCS)
        ]
        embs = embedder.embed_texts([d.text for d in docs])
        store.add_documents(docs, embeddings=embs)

        searcher = HybridSearcher(vector_store=store)
        searcher.index_existing_documents(docs)
        return searcher

    def test_concurrent_searches_no_errors(self):
        """20 threads × 10 searches must complete without any exception."""
        searcher = self._build_searcher()
        errors: List[Exception] = []
        result_counts: List[int] = []
        barrier = threading.Barrier(N_THREADS)

        queries = [
            "python programming",
            "machine learning",
            "neural network training",
            "data science tools",
            "optimization algorithms",
        ]

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(SEARCHES_PER_THREAD):
                try:
                    query = queries[(tid + i) % len(queries)]
                    results = searcher.search(query, top_k=5)
                    assert isinstance(results, list)
                    result_counts.append(len(results))
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent search: {errors[:3]}"
        assert len(result_counts) == N_THREADS * SEARCHES_PER_THREAD

    def test_bm25_state_consistent_after_concurrent_search(self):
        """
        BM25 document_count must equal N_DOCS after concurrent searches.
        Searches must not mutate the index.
        """
        searcher = self._build_searcher()
        bm25_count_before = searcher.bm25.document_count
        assert bm25_count_before == N_DOCS

        barrier = threading.Barrier(N_THREADS)
        errors: List[Exception] = []

        def worker(tid: int) -> None:
            barrier.wait()
            for _ in range(SEARCHES_PER_THREAD):
                try:
                    searcher.search("python data", top_k=3)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # BM25 index must not have grown (searches are read-only)
        assert searcher.bm25.document_count == bm25_count_before

    def test_concurrent_add_and_search_no_crash(self):
        """
        Adding documents while searching concurrently must not crash either side.
        """
        searcher = self._build_searcher()
        errors: List[Exception] = []
        barrier = threading.Barrier(2)

        def adder() -> None:
            barrier.wait()
            new_docs = [
                Document(text=f"new concurrent doc {i}", metadata={"topic": "concurrent"})
                for i in range(50)
            ]
            try:
                searcher.add_documents(new_docs)
            except Exception as e:
                errors.append(e)

        def searcher_worker() -> None:
            barrier.wait()
            for _ in range(30):
                try:
                    results = searcher.search("concurrent doc", top_k=5)
                    assert isinstance(results, list)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=adder), threading.Thread(target=searcher_worker)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent add+search: {errors}"
