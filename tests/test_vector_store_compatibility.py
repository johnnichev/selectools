"""
Vector store compatibility tests.

Ensures all vector store implementations behave consistently with the same data.
"""

import os
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest

from selectools.rag import Document, VectorStore
from selectools.rag.stores.memory import InMemoryVectorStore
from selectools.rag.stores.sqlite import SQLiteVectorStore

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_embedder():
    """Create a consistent mock embedding provider."""
    embedder = Mock()
    embedder.model = "mock-embedding-model"
    embedder.dimension = 128

    def mock_embed_text(text: str):
        # Consistent embeddings based on text
        hash_val = abs(hash(text)) % 1000
        return [float(hash_val + i) / 1000.0 for i in range(128)]

    def mock_embed_texts(texts: list):
        return [mock_embed_text(text) for text in texts]

    def mock_embed_query(query: str):
        return mock_embed_text(query)

    embedder.embed_text.side_effect = mock_embed_text
    embedder.embed_texts.side_effect = mock_embed_texts
    embedder.embed_query.side_effect = mock_embed_query

    return embedder


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            text="Python is a versatile programming language used for web development.",
            metadata={"category": "programming", "difficulty": "beginner", "lang": "python"},
        ),
        Document(
            text="JavaScript is essential for modern web frontend development.",
            metadata={"category": "programming", "difficulty": "beginner", "lang": "javascript"},
        ),
        Document(
            text="Machine learning algorithms can learn patterns from data automatically.",
            metadata={"category": "ai", "difficulty": "advanced", "topic": "ml"},
        ),
        Document(
            text="Docker containers provide isolated environments for running applications.",
            metadata={"category": "devops", "difficulty": "intermediate", "tool": "docker"},
        ),
        Document(
            text="Kubernetes orchestrates container deployments across clusters.",
            metadata={"category": "devops", "difficulty": "advanced", "tool": "kubernetes"},
        ),
    ]


@pytest.fixture
def all_vector_stores(mock_embedder):
    """Create instances of all vector store types."""
    stores = {}

    # In-memory store
    stores["memory"] = InMemoryVectorStore(embedder=mock_embedder)

    # SQLite store
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    stores["sqlite"] = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)
    stores["sqlite_db_path"] = db_path

    yield stores

    # Cleanup
    if os.path.exists(stores["sqlite_db_path"]):
        try:
            os.remove(stores["sqlite_db_path"])
        except Exception:
            pass


# ============================================================================
# Consistency Tests
# ============================================================================


class TestVectorStoreConsistency:
    """Test that all vector stores behave consistently."""

    def test_add_documents_returns_same_count(self, all_vector_stores, sample_documents):
        """Test that all stores return same number of IDs when adding documents."""
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue

            doc_ids = store.add_documents(sample_documents)

            assert len(doc_ids) == len(sample_documents), f"{store_name} failed"
            assert all(isinstance(id, str) for id in doc_ids), f"{store_name} failed"

    def test_search_returns_similar_results(
        self, all_vector_stores, mock_embedder, sample_documents
    ):
        """Test that all stores return similar search results for same query."""
        # Add documents to all stores
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            store.add_documents(sample_documents)

        # Search with same query
        query_embedding = mock_embedder.embed_query("programming language")

        results_by_store = {}
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            results = store.search(query_embedding, top_k=3)
            results_by_store[store_name] = results

        # All stores should return same number of results
        result_counts = {name: len(results) for name, results in results_by_store.items()}
        assert len(set(result_counts.values())) == 1, f"Inconsistent result counts: {result_counts}"

    def test_top_k_consistency(self, all_vector_stores, mock_embedder, sample_documents):
        """Test that top_k parameter works consistently."""
        # Add documents
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            store.add_documents(sample_documents)

        # Test different top_k values
        query_embedding = mock_embedder.embed_query("test")

        for k in [1, 3, 5]:
            for store_name, store in all_vector_stores.items():
                if store_name.endswith("_db_path"):
                    continue
                results = store.search(query_embedding, top_k=k)
                assert len(results) == k, f"{store_name} returned {len(results)} instead of {k}"

    def test_score_ordering_consistency(self, all_vector_stores, mock_embedder, sample_documents):
        """Test that all stores return results in descending score order."""
        # Add documents
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            store.add_documents(sample_documents)

        query_embedding = mock_embedder.embed_query("programming")

        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            results = store.search(query_embedding, top_k=5)

            scores = [r.score for r in results]
            assert scores == sorted(
                scores, reverse=True
            ), f"{store_name} scores not in descending order"


# ============================================================================
# Metadata Filtering Tests
# ============================================================================


class TestMetadataFilteringConsistency:
    """Test that metadata filtering works consistently across stores."""

    def test_single_filter_consistency(self, all_vector_stores, mock_embedder, sample_documents):
        """Test single metadata filter across all stores."""
        # Add documents
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            store.add_documents(sample_documents)

        query_embedding = mock_embedder.embed_query("test")

        # Filter by category
        filter_dict = {"category": "programming"}

        results_by_store = {}
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            results = store.search(query_embedding, top_k=10, filter=filter_dict)
            results_by_store[store_name] = results

        # All stores should return same number of filtered results
        result_counts = {name: len(results) for name, results in results_by_store.items()}
        assert (
            len(set(result_counts.values())) == 1
        ), f"Inconsistent filtered results: {result_counts}"

        # Verify all results match the filter
        for store_name, results in results_by_store.items():
            for result in results:
                assert (
                    result.document.metadata["category"] == "programming"
                ), f"{store_name} failed filter"

    def test_multiple_filters_consistency(self, all_vector_stores, mock_embedder, sample_documents):
        """Test multiple metadata filters."""
        # Add documents
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            store.add_documents(sample_documents)

        query_embedding = mock_embedder.embed_query("test")
        filter_dict = {"category": "devops", "difficulty": "advanced"}

        results_by_store = {}
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            results = store.search(query_embedding, top_k=10, filter=filter_dict)
            results_by_store[store_name] = results

        # Should all return the Kubernetes document
        for store_name, results in results_by_store.items():
            assert len(results) == 1, f"{store_name} returned {len(results)} results"
            assert results[0].document.metadata["category"] == "devops"
            assert results[0].document.metadata["difficulty"] == "advanced"


# ============================================================================
# CRUD Operations Consistency
# ============================================================================


class TestCRUDConsistency:
    """Test CRUD operations consistency."""

    def test_delete_consistency(self, all_vector_stores, mock_embedder, sample_documents):
        """Test that delete works consistently."""
        # Add documents and get IDs
        doc_ids_by_store = {}
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            doc_ids = store.add_documents(sample_documents)
            doc_ids_by_store[store_name] = doc_ids

        # Delete first document from each store
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            doc_ids = doc_ids_by_store[store_name]
            store.delete([doc_ids[0]])

        # Verify all stores have 4 documents remaining
        query_embedding = mock_embedder.embed_query("test")
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            results = store.search(query_embedding, top_k=10)
            assert len(results) == 4, f"{store_name} has {len(results)} documents after delete"

    def test_clear_consistency(self, all_vector_stores, mock_embedder, sample_documents):
        """Test that clear works consistently."""
        # Add documents
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            store.add_documents(sample_documents)

        # Clear all stores
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            store.clear()

        # Verify all stores are empty
        query_embedding = mock_embedder.embed_query("test")
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            results = store.search(query_embedding, top_k=10)
            assert len(results) == 0, f"{store_name} not empty after clear"


# ============================================================================
# Edge Cases Consistency
# ============================================================================


class TestEdgeCasesConsistency:
    """Test edge cases across all stores."""

    def test_empty_store_search(self, all_vector_stores, mock_embedder):
        """Test searching in empty stores."""
        query_embedding = mock_embedder.embed_query("test")

        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            results = store.search(query_embedding, top_k=5)
            assert len(results) == 0, f"{store_name} returned results from empty store"

    def test_single_document(self, all_vector_stores, mock_embedder):
        """Test with single document."""
        doc = Document(text="Single document test", metadata={"test": True})

        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            doc_ids = store.add_documents([doc])
            assert len(doc_ids) == 1, f"{store_name} failed with single document"

        # Search
        query_embedding = mock_embedder.embed_query("test")
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            results = store.search(query_embedding, top_k=5)
            assert len(results) == 1, f"{store_name} failed single document search"

    def test_duplicate_documents(self, all_vector_stores, mock_embedder):
        """Test adding duplicate documents."""
        doc = Document(text="Duplicate test", metadata={"dup": True})
        docs = [doc, doc, doc]  # Same document 3 times

        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            doc_ids = store.add_documents(docs)
            assert len(doc_ids) == 3, f"{store_name} failed with duplicates"

        # Search should return 3 results
        query_embedding = mock_embedder.embed_query("duplicate")
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            results = store.search(query_embedding, top_k=10)
            assert len(results) == 3, f"{store_name} duplicate handling inconsistent"


# ============================================================================
# Performance Comparison Tests
# ============================================================================


class TestPerformanceCharacteristics:
    """Compare performance characteristics (not strict equality)."""

    def test_search_time_reasonable(self, all_vector_stores, mock_embedder, sample_documents):
        """Test that search completes in reasonable time for all stores."""
        import time

        # Add documents
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            store.add_documents(sample_documents)

        query_embedding = mock_embedder.embed_query("test")

        # Measure search time
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue

            start = time.time()
            results = store.search(query_embedding, top_k=3)
            elapsed = time.time() - start

            # Should complete in less than 1 second (very generous)
            assert elapsed < 1.0, f"{store_name} search took {elapsed:.3f}s"
            assert len(results) == 3

    def test_batch_insert_performance(self, all_vector_stores, mock_embedder):
        """Test batch insert performance."""
        import time

        # Create 50 documents
        docs = [
            Document(text=f"Document {i} with some content", metadata={"idx": i}) for i in range(50)
        ]

        # Test each store
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue

            start = time.time()
            doc_ids = store.add_documents(docs)
            elapsed = time.time() - start

            assert len(doc_ids) == 50, f"{store_name} failed batch insert"
            # Should complete in less than 5 seconds
            assert elapsed < 5.0, f"{store_name} batch insert took {elapsed:.3f}s"


# ============================================================================
# API Compatibility Tests
# ============================================================================


class TestAPICompatibility:
    """Test that all stores have compatible APIs."""

    def test_all_have_add_documents(self, all_vector_stores):
        """Test that all stores have add_documents method."""
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            assert hasattr(store, "add_documents")
            assert callable(store.add_documents)

    def test_all_have_search(self, all_vector_stores):
        """Test that all stores have search method."""
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            assert hasattr(store, "search")
            assert callable(store.search)

    def test_all_have_delete(self, all_vector_stores):
        """Test that all stores have delete method."""
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            assert hasattr(store, "delete")
            assert callable(store.delete)

    def test_all_have_clear(self, all_vector_stores):
        """Test that all stores have clear method."""
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            assert hasattr(store, "clear")
            assert callable(store.clear)

    def test_all_have_embedder(self, all_vector_stores):
        """Test that all stores have embedder attribute."""
        for store_name, store in all_vector_stores.items():
            if store_name.endswith("_db_path"):
                continue
            assert hasattr(store, "embedder")
            assert store.embedder is not None
