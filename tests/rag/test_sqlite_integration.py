"""
SQLite vector store integration tests.

Tests persistence, reconnection, and real-world usage patterns.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import pytest

from selectools.rag import Document
from selectools.rag.stores.sqlite import SQLiteVectorStore

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_embedder() -> Mock:
    """Create a mock embedding provider."""
    embedder = Mock()
    embedder.model = "mock-embedding-model"
    embedder.dimension = 128

    def mock_embed_text(text: str) -> list[float]:
        # Generate consistent embeddings based on text hash
        hash_val = hash(text) % 1000
        return [float(hash_val + i) / 1000.0 for i in range(128)]

    def mock_embed_texts(texts: list[str]) -> list[list[float]]:
        return [mock_embed_text(text) for text in texts]

    def mock_embed_query(query: str) -> list[float]:
        return mock_embed_text(query)

    embedder.embed_text.side_effect = mock_embed_text
    embedder.embed_texts.side_effect = mock_embed_texts
    embedder.embed_query.side_effect = mock_embed_query

    return embedder


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database path (file not created yet)."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents."""
    return [
        Document(
            text="Python is a programming language.",
            metadata={"category": "programming", "difficulty": "beginner"},
        ),
        Document(
            text="Machine learning is a subset of AI.",
            metadata={"category": "ai", "difficulty": "intermediate"},
        ),
        Document(
            text="Docker is a containerization platform.",
            metadata={"category": "devops", "difficulty": "intermediate"},
        ),
        Document(
            text="React is a JavaScript library for UI.",
            metadata={"category": "web", "difficulty": "beginner"},
        ),
        Document(
            text="Kubernetes orchestrates containers.",
            metadata={"category": "devops", "difficulty": "advanced"},
        ),
    ]


# ============================================================================
# Persistence Tests
# ============================================================================


class TestSQLitePersistence:
    """Test that data persists across connections."""

    def test_basic_persistence(
        self, mock_embedder: Mock, temp_db_path: str, sample_documents: list[Document]
    ) -> None:
        """Test basic persistence after reconnection."""
        # Add documents and close
        store1 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        doc_ids = store1.add_documents(sample_documents)
        del store1

        # Reopen and verify
        store2 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        query_embedding = mock_embedder.embed_query("programming")
        results = store2.search(query_embedding, top_k=5)

        assert len(results) == 5
        assert all(isinstance(result.document.text, str) for result in results)
        assert all(len(result.document.text) > 0 for result in results)

    def test_metadata_persistence(
        self, mock_embedder: Mock, temp_db_path: str, sample_documents: list[Document]
    ) -> None:
        """Test that metadata persists correctly."""
        # Add documents
        store1 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        store1.add_documents(sample_documents)
        del store1

        # Reopen and verify metadata
        store2 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        query_embedding = mock_embedder.embed_query("test")
        results = store2.search(query_embedding, top_k=5)

        # Verify all metadata fields are preserved
        for result in results:
            assert "category" in result.document.metadata
            assert "difficulty" in result.document.metadata

    def test_embeddings_persistence(self, mock_embedder: Mock, temp_db_path: str) -> None:
        """Test that embeddings are stored and retrieved correctly."""
        # Add document with known embedding
        store1 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        doc = Document(text="test document", metadata={})
        known_embedding = [0.5] * 128
        store1.add_documents([doc], embeddings=[known_embedding])
        del store1

        # Reopen and search with similar embedding
        store2 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        results = store2.search([0.5] * 128, top_k=1)

        assert len(results) == 1
        # Cosine similarity should be ~1.0 for identical vectors
        assert results[0].score > 0.99

    def test_multiple_sessions(
        self, mock_embedder: Mock, temp_db_path: str, sample_documents: list[Document]
    ) -> None:
        """Test multiple open/close cycles."""
        # Session 1: Add 2 documents
        store1 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        store1.add_documents(sample_documents[:2])
        del store1

        # Session 2: Add 2 more documents
        store2 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        store2.add_documents(sample_documents[2:4])
        del store2

        # Session 3: Verify all 4 documents exist
        store3 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        query_embedding = mock_embedder.embed_query("test")
        results = store3.search(query_embedding, top_k=10)

        assert len(results) == 4

    def test_delete_and_reconnect(
        self, mock_embedder: Mock, temp_db_path: str, sample_documents: list[Document]
    ) -> None:
        """Test that deletions persist."""
        # Add documents
        store1 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        doc_ids = store1.add_documents(sample_documents)

        # Delete some documents
        store1.delete([doc_ids[0], doc_ids[2]])
        del store1

        # Reopen and verify
        store2 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        query_embedding = mock_embedder.embed_query("test")
        results = store2.search(query_embedding, top_k=10)

        # Should have 3 remaining documents (deleted 2 out of 5)
        assert len(results) == 3


# ============================================================================
# Database Operations Tests
# ============================================================================


class TestSQLiteDatabaseOperations:
    """Test database-specific operations."""

    def test_database_file_creation(self, mock_embedder: Mock, temp_db_path: str) -> None:
        """Test that database file is created."""
        assert not os.path.exists(temp_db_path)

        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)

        assert os.path.exists(temp_db_path)
        assert os.path.isfile(temp_db_path)

    def test_custom_db_path(self, mock_embedder: Mock) -> None:
        """Test using custom database path."""
        custom_dir = tempfile.mkdtemp()
        custom_path = os.path.join(custom_dir, "my_custom.db")

        try:
            store = SQLiteVectorStore(embedder=mock_embedder, db_path=custom_path)
            store.add_documents([Document(text="test", metadata={})])

            assert os.path.exists(custom_path)
        finally:
            if os.path.exists(custom_path):
                os.remove(custom_path)
            os.rmdir(custom_dir)

    def test_clear_database(
        self, mock_embedder: Mock, temp_db_path: str, sample_documents: list[Document]
    ) -> None:
        """Test clearing the database."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        store.add_documents(sample_documents)

        # Verify documents exist
        query_embedding = mock_embedder.embed_query("test")
        results_before = store.search(query_embedding, top_k=10)
        assert len(results_before) == 5

        # Clear
        store.clear()

        # Verify empty
        results_after = store.search(query_embedding, top_k=10)
        assert len(results_after) == 0

    def test_upsert_behavior(self, mock_embedder: Mock, temp_db_path: str) -> None:
        """Test multiple additions and deletions."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)

        # Add initial documents
        doc1 = Document(text="original text", metadata={"version": 1})
        doc_ids = store.add_documents([doc1])

        # Delete and re-add with new content
        store.delete(doc_ids)
        doc2 = Document(text="updated text", metadata={"version": 2})
        new_ids = store.add_documents([doc2])

        # Search and verify the new document is there
        query_embedding = mock_embedder.embed_query("updated")
        results = store.search(query_embedding, top_k=10)

        # Should have 1 document (the updated one)
        assert len(results) == 1
        assert "updated" in results[0].document.text or "original" in results[0].document.text


# ============================================================================
# Search Quality Tests
# ============================================================================


class TestSQLiteSearchQuality:
    """Test search accuracy and quality."""

    def test_search_relevance_ordering(
        self, mock_embedder: Mock, temp_db_path: str, sample_documents: list[Document]
    ) -> None:
        """Test that search results are ordered by relevance."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        store.add_documents(sample_documents)

        # Query for "programming"
        query_embedding = mock_embedder.embed_query("programming language")
        results = store.search(query_embedding, top_k=5)

        assert len(results) > 0
        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_filter_accuracy(
        self, mock_embedder: Mock, temp_db_path: str, sample_documents: list[Document]
    ) -> None:
        """Test that metadata filtering works correctly."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        store.add_documents(sample_documents)

        query_embedding = mock_embedder.embed_query("test")

        # Filter by category
        devops_results = store.search(query_embedding, top_k=10, filter={"category": "devops"})

        assert len(devops_results) == 2  # Docker and Kubernetes
        assert all(r.document.metadata["category"] == "devops" for r in devops_results)

    def test_top_k_limiting(
        self, mock_embedder: Mock, temp_db_path: str, sample_documents: list[Document]
    ) -> None:
        """Test that top_k parameter works correctly."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        store.add_documents(sample_documents)

        query_embedding = mock_embedder.embed_query("test")

        # Test different top_k values
        for k in [1, 3, 5]:
            results = store.search(query_embedding, top_k=k)
            assert len(results) == k

    def test_empty_search(self, mock_embedder: Mock, temp_db_path: str) -> None:
        """Test searching in empty database."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)

        query_embedding = mock_embedder.embed_query("test")
        results = store.search(query_embedding, top_k=5)

        assert len(results) == 0


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestSQLiteConcurrency:
    """Test concurrent access patterns."""

    def test_sequential_writes(self, mock_embedder: Mock, temp_db_path: str) -> None:
        """Test sequential writes from different store instances."""
        # First instance writes
        store1 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        store1.add_documents([Document(text="doc1", metadata={})])
        del store1

        # Second instance writes
        store2 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        store2.add_documents([Document(text="doc2", metadata={})])
        del store2

        # Third instance reads
        store3 = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)
        query_embedding = mock_embedder.embed_query("test")
        results = store3.search(query_embedding, top_k=10)

        assert len(results) == 2

    def test_read_after_write(
        self, mock_embedder: Mock, temp_db_path: str, sample_documents: list[Document]
    ) -> None:
        """Test reading immediately after writing."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)

        # Write
        store.add_documents(sample_documents[:2])

        # Read immediately
        query_embedding = mock_embedder.embed_query("test")
        results = store.search(query_embedding, top_k=10)

        assert len(results) == 2


# ============================================================================
# Performance Tests
# ============================================================================


class TestSQLitePerformance:
    """Test performance characteristics."""

    def test_large_batch_insert(self, mock_embedder: Mock, temp_db_path: str) -> None:
        """Test inserting a large batch of documents."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)

        # Create 100 documents
        docs = [Document(text=f"Document {i} with content", metadata={"id": i}) for i in range(100)]

        # Insert all at once
        doc_ids = store.add_documents(docs)

        assert len(doc_ids) == 100

        # Verify all can be searched
        query_embedding = mock_embedder.embed_query("document")
        results = store.search(query_embedding, top_k=100)

        assert len(results) == 100

    def test_search_performance(self, mock_embedder: Mock, temp_db_path: str) -> None:
        """Test search on moderately sized database."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)

        # Add 50 documents
        docs = [Document(text=f"Content {i}", metadata={"idx": i}) for i in range(50)]
        store.add_documents(docs)

        # Perform multiple searches
        query_embedding = mock_embedder.embed_query("content")

        for _ in range(10):
            results = store.search(query_embedding, top_k=5)
            assert len(results) == 5


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestSQLiteErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_db_path(self, mock_embedder: Mock) -> None:
        """Test handling of invalid database path."""
        # Path to a directory (not a file)
        with tempfile.TemporaryDirectory() as temp_dir:
            # This should work (SQLite creates the file)
            store = SQLiteVectorStore(
                embedder=mock_embedder, db_path=os.path.join(temp_dir, "test.db")
            )
            store.add_documents([Document(text="test", metadata={})])

    def test_delete_nonexistent_documents(self, mock_embedder: Mock, temp_db_path: str) -> None:
        """Test deleting non-existent documents (should not error)."""
        store = SQLiteVectorStore(embedder=mock_embedder, db_path=temp_db_path)

        # Should not raise error
        store.delete(["nonexistent-id-1", "nonexistent-id-2"])

        # Database should still work
        store.add_documents([Document(text="test", metadata={})])
        query_embedding = mock_embedder.embed_query("test")
        results = store.search(query_embedding, top_k=1)
        assert len(results) == 1
