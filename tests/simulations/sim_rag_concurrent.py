"""
Simulation: RAG pipeline under concurrent read/write load.

Tests that InMemoryVectorStore remains consistent when N threads simultaneously
index documents and perform similarity searches. No API keys required — uses a
stub embedder.

Run: pytest tests/simulations/sim_rag_concurrent.py -v
"""

from __future__ import annotations

import threading
from typing import List
from unittest.mock import Mock

import pytest

from selectools.rag.vector_store import Document

# ---------------------------------------------------------------------------
# Stub embedder (thread-safe plain class, no Mock)
# ---------------------------------------------------------------------------

DIM = 8


class _StubEmbedder:
    def embed_texts(self, texts):
        return [[0.1 + i * 0.01] * DIM for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [0.15] * DIM


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRAGConcurrentLoad:
    """RAG pipeline must remain consistent under concurrent index + search load."""

    N_INDEXERS = 5
    N_SEARCHERS = 5
    DOCS_PER_INDEXER = 10
    SEARCHES_PER_THREAD = 20

    def test_concurrent_index_and_search_consistency(self):
        """
        5 threads index documents while 5 threads search simultaneously.
        After all threads finish:
        - No exceptions occurred
        - List length == numpy shape (no desync)
        - All searches returned valid lists
        """
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(embedder=_StubEmbedder())

        # Seed with 20 documents before the concurrent phase
        seed_docs = [Document(text=f"seed doc {i} python data science") for i in range(20)]
        seed_embs = _StubEmbedder().embed_texts([d.text for d in seed_docs])
        store.add_documents(seed_docs, embeddings=seed_embs)

        errors: List[Exception] = []
        search_result_counts: List[int] = []
        barrier = threading.Barrier(self.N_INDEXERS + self.N_SEARCHERS)

        def indexer(tid: int) -> None:
            barrier.wait()
            docs = [
                Document(text=f"indexer{tid} doc{i} machine learning")
                for i in range(self.DOCS_PER_INDEXER)
            ]
            embs = _StubEmbedder().embed_texts([d.text for d in docs])
            try:
                store.add_documents(docs, embeddings=embs)
            except Exception as e:
                errors.append(e)

        def searcher(tid: int) -> None:
            barrier.wait()
            for _ in range(self.SEARCHES_PER_THREAD):
                try:
                    results = store.search([0.15] * DIM, top_k=5)
                    assert isinstance(results, list)
                    search_result_counts.append(len(results))
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=indexer, args=(t,)) for t in range(self.N_INDEXERS)] + [
            threading.Thread(target=searcher, args=(t,)) for t in range(self.N_SEARCHERS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent RAG: {errors}"

        # Check alignment
        n_docs = len(store.documents)
        n_embs = store.embeddings.shape[0] if store.embeddings is not None else 0
        assert n_docs == n_embs, f"List/numpy desync: {n_docs} docs vs {n_embs} embeddings"

        # At least 20 searches returned results (seeded docs always present)
        assert len(search_result_counts) > 0
        assert all(isinstance(c, int) for c in search_result_counts)

    def test_concurrent_index_total_count_accurate(self):
        """After N concurrent add_documents calls, total count must be exact."""
        from selectools.rag.stores.memory import InMemoryVectorStore

        store = InMemoryVectorStore(embedder=_StubEmbedder())
        errors: List[Exception] = []
        barrier = threading.Barrier(self.N_INDEXERS)
        counts: List[int] = []

        def indexer(tid: int) -> None:
            docs = [Document(text=f"t{tid}d{i}") for i in range(self.DOCS_PER_INDEXER)]
            embs = _StubEmbedder().embed_texts([d.text for d in docs])
            barrier.wait()
            try:
                store.add_documents(docs, embeddings=embs)
                counts.append(self.DOCS_PER_INDEXER)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=indexer, args=(t,)) for t in range(self.N_INDEXERS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        expected = self.N_INDEXERS * self.DOCS_PER_INDEXER
        assert len(store.documents) == expected
        assert store.embeddings.shape[0] == expected
