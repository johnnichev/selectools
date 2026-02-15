"""
Unit tests for vector store CRUD operations.

Tests all 4 vector store implementations:
- InMemoryVectorStore
- SQLiteVectorStore
- ChromaVectorStore
- PineconeVectorStore
"""

from __future__ import annotations

import os
import tempfile
from typing import List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from selectools.rag import Document, SearchResult, VectorStore
from selectools.rag.stores.memory import InMemoryVectorStore
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

    def mock_embed_texts(texts: List[str]) -> list[list[float]]:
        return [mock_embed_text(text) for text in texts]

    def mock_embed_query(query: str) -> list[float]:
        return mock_embed_text(query)

    embedder.embed_text.side_effect = mock_embed_text
    embedder.embed_texts.side_effect = mock_embed_texts
    embedder.embed_query.side_effect = mock_embed_query

    return embedder


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            text="The quick brown fox jumps over the lazy dog.",
            metadata={"category": "animals", "source": "test1.txt"},
        ),
        Document(
            text="Python is a high-level programming language.",
            metadata={"category": "programming", "source": "test2.txt"},
        ),
        Document(
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "ai", "source": "test3.txt"},
        ),
    ]


# ============================================================================
# InMemoryVectorStore Tests
# ============================================================================


class TestInMemoryVectorStore:
    """Test in-memory vector store implementation."""

    def test_initialization(self, mock_embedder: Mock) -> None:
        """Test store initialization."""
        store = InMemoryVectorStore(embedder=mock_embedder)
        assert store.embedder == mock_embedder
        assert len(store.documents) == 0
        # Verify empty store returns no results
        query_embedding = mock_embedder.embed_query("test")
        results = store.search(query_embedding, top_k=5)
        assert len(results) == 0

    def test_add_documents(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test adding documents to the store."""
        store = InMemoryVectorStore(embedder=mock_embedder)
        doc_ids = store.add_documents(sample_documents)

        assert len(doc_ids) == 3
        assert all(isinstance(id, str) for id in doc_ids)
        assert len(store.documents) == 3
        # Verify documents can be searched (embeddings were stored)
        query_embedding = mock_embedder.embed_query("test")
        results = store.search(query_embedding, top_k=5)
        assert len(results) == 3

    def test_add_documents_with_embeddings(
        self, mock_embedder: Mock, sample_documents: list[Document]
    ) -> None:
        """Test adding documents with pre-computed embeddings."""
        store = InMemoryVectorStore(embedder=mock_embedder)

        # Pre-compute embeddings
        embeddings = [[0.1] * 128, [0.2] * 128, [0.3] * 128]

        doc_ids = store.add_documents(sample_documents, embeddings=embeddings)

        assert len(doc_ids) == 3
        # Verify documents were added by searching
        query_embedding = embeddings[0]  # Search for first document
        results = store.search(query_embedding, top_k=1)
        assert len(results) == 1
        assert results[0].score > 0.99  # Should be very similar to itself

    def test_search_basic(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test basic search functionality."""
        store = InMemoryVectorStore(embedder=mock_embedder)
        store.add_documents(sample_documents)

        query_embedding = mock_embedder.embed_query("programming")
        results = store.search(query_embedding, top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(hasattr(r, "document") for r in results)
        assert all(hasattr(r, "score") for r in results)
        # Scores should be in descending order
        if len(results) > 1:
            assert results[0].score >= results[1].score

    def test_search_with_filter(
        self, mock_embedder: Mock, sample_documents: list[Document]
    ) -> None:
        """Test search with metadata filtering."""
        store = InMemoryVectorStore(embedder=mock_embedder)
        store.add_documents(sample_documents)

        query_embedding = mock_embedder.embed_query("test")

        # Search with category filter
        results = store.search(query_embedding, top_k=5, filter={"category": "programming"})

        assert len(results) <= 1  # Only one programming document
        if results:
            assert results[0].document.metadata["category"] == "programming"

    def test_search_empty_store(self, mock_embedder: Mock) -> None:
        """Test searching in empty store."""
        store = InMemoryVectorStore(embedder=mock_embedder)

        query_embedding = [0.1] * 128
        results = store.search(query_embedding, top_k=3)

        assert len(results) == 0

    def test_delete_documents(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test deleting documents from store."""
        store = InMemoryVectorStore(embedder=mock_embedder)
        doc_ids = store.add_documents(sample_documents)

        # Delete one document
        store.delete([doc_ids[0]])

        assert len(store.documents) == 2
        assert doc_ids[0] not in store.documents
        # Verify deletion by searching - should get 2 results now
        query_embedding = mock_embedder.embed_query("test")
        results = store.search(query_embedding, top_k=10)
        assert len(results) == 2

    def test_delete_multiple_documents(
        self, mock_embedder: Mock, sample_documents: list[Document]
    ) -> None:
        """Test deleting multiple documents."""
        store = InMemoryVectorStore(embedder=mock_embedder)
        doc_ids = store.add_documents(sample_documents)

        # Delete two documents
        store.delete([doc_ids[0], doc_ids[2]])

        assert len(store.documents) == 1
        # Verify the remaining document is the second one
        assert doc_ids[1] in store.ids
        # Verify via search that only 1 document remains
        query_embedding = mock_embedder.embed_query("test")
        results = store.search(query_embedding, top_k=10)
        assert len(results) == 1

    def test_delete_nonexistent_document(
        self, mock_embedder: Mock, sample_documents: list[Document]
    ) -> None:
        """Test deleting non-existent document (should not error)."""
        store = InMemoryVectorStore(embedder=mock_embedder)
        store.add_documents(sample_documents)

        # Should not raise error
        store.delete(["nonexistent-id"])

        assert len(store.documents) == 3

    def test_clear_store(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test clearing all documents."""
        store = InMemoryVectorStore(embedder=mock_embedder)
        store.add_documents(sample_documents)

        store.clear()

        assert len(store.documents) == 0
        # Verify clear by searching - should get no results
        query_embedding = mock_embedder.embed_query("test")
        results = store.search(query_embedding, top_k=10)
        assert len(results) == 0

    def test_cosine_similarity_accuracy(self, mock_embedder: Mock) -> None:
        """Test cosine similarity calculation accuracy."""
        store = InMemoryVectorStore(embedder=mock_embedder)

        # Create documents with known embeddings
        docs = [
            Document(text="doc1", metadata={}),
            Document(text="doc2", metadata={}),
        ]

        # Identical embeddings (should have score ~1.0)
        embeddings = [
            [1.0] * 128,
            [1.0] * 128,
        ]

        store.add_documents(docs, embeddings=embeddings)

        # Query with same embedding
        results = store.search([1.0] * 128, top_k=2)

        assert len(results) == 2
        # Cosine similarity should be ~1.0 for identical vectors
        assert results[0].score > 0.99
        assert results[1].score > 0.99

    def test_top_k_limiting(self, mock_embedder: Mock) -> None:
        """Test that top_k properly limits results."""
        store = InMemoryVectorStore(embedder=mock_embedder)

        # Add 10 documents
        docs = [Document(text=f"doc{i}", metadata={}) for i in range(10)]
        store.add_documents(docs)

        # Search with top_k=3
        query_embedding = mock_embedder.embed_query("test")
        results = store.search(query_embedding, top_k=3)

        assert len(results) == 3


# ============================================================================
# SQLiteVectorStore Tests
# ============================================================================


class TestSQLiteVectorStore:
    """Test SQLite vector store implementation."""

    def test_initialization(self, mock_embedder: Mock) -> None:
        """Test store initialization."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)
            assert store.embedder == mock_embedder
            assert store.db_path == db_path
            assert os.path.exists(db_path)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_add_documents(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test adding documents to SQLite store."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)
            doc_ids = store.add_documents(sample_documents)

            assert len(doc_ids) == 3
            assert all(isinstance(id, str) for id in doc_ids)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_persistence(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test that documents persist after reconnection."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            # Add documents and close
            store1 = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)
            doc_ids = store1.add_documents(sample_documents)
            del store1

            # Reopen and verify documents exist
            store2 = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)
            query_embedding = mock_embedder.embed_query("test")
            results = store2.search(query_embedding, top_k=5)

            assert len(results) == 3
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_search_basic(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test basic search functionality."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)
            store.add_documents(sample_documents)

            query_embedding = mock_embedder.embed_query("programming")
            results = store.search(query_embedding, top_k=2)

            assert len(results) <= 2
            assert all(isinstance(r, SearchResult) for r in results)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_search_with_filter(
        self, mock_embedder: Mock, sample_documents: list[Document]
    ) -> None:
        """Test search with metadata filtering."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)
            store.add_documents(sample_documents)

            query_embedding = mock_embedder.embed_query("test")
            results = store.search(query_embedding, top_k=5, filter={"category": "ai"})

            assert len(results) <= 1
            if results:
                assert results[0].document.metadata["category"] == "ai"
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_delete_documents(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test deleting documents."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)
            doc_ids = store.add_documents(sample_documents)

            store.delete([doc_ids[0]])

            query_embedding = mock_embedder.embed_query("test")
            results = store.search(query_embedding, top_k=5)

            assert len(results) == 2
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_clear_store(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test clearing all documents."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            store = SQLiteVectorStore(embedder=mock_embedder, db_path=db_path)
            store.add_documents(sample_documents)

            store.clear()

            query_embedding = mock_embedder.embed_query("test")
            results = store.search(query_embedding, top_k=5)

            assert len(results) == 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


# ============================================================================
# ChromaVectorStore Tests (with mocking)
# ============================================================================


class TestChromaVectorStore:
    """Test Chroma vector store wrapper."""

    def test_initialization(self, mock_embedder: Mock) -> None:
        """Test store initialization."""
        mock_chroma = MagicMock()
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.Client.return_value = mock_client

        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            from selectools.rag.stores.chroma import ChromaVectorStore

            store = ChromaVectorStore(embedder=mock_embedder, collection_name="test_collection")
            assert store.embedder == mock_embedder

    def test_add_documents(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test adding documents."""
        mock_chroma = MagicMock()
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.Client.return_value = mock_client

        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            from selectools.rag.stores.chroma import ChromaVectorStore

            store = ChromaVectorStore(embedder=mock_embedder)

            doc_ids = store.add_documents(sample_documents)

            assert len(doc_ids) == 3
            mock_collection.add.assert_called_once()

    def test_search_basic(self, mock_embedder: Mock) -> None:
        """Test basic search."""
        mock_chroma = MagicMock()
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.Client.return_value = mock_client

        # Mock query results
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"cat": "a"}, {"cat": "b"}]],
            "distances": [[0.1, 0.3]],
            "embeddings": [[]],
        }

        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            from selectools.rag.stores.chroma import ChromaVectorStore

            store = ChromaVectorStore(embedder=mock_embedder)

            query_embedding = [0.1] * 128
            results = store.search(query_embedding, top_k=2)

            assert len(results) == 2
            mock_collection.query.assert_called_once()

    def test_delete_documents(self, mock_embedder: Mock) -> None:
        """Test deleting documents."""
        mock_chroma = MagicMock()
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.Client.return_value = mock_client

        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            from selectools.rag.stores.chroma import ChromaVectorStore

            store = ChromaVectorStore(embedder=mock_embedder)

            store.delete(["id1", "id2"])

            mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])


# ============================================================================
# PineconeVectorStore Tests (with mocking)
# ============================================================================


class TestPineconeVectorStore:
    """Test Pinecone vector store wrapper."""

    def test_initialization(self, mock_embedder: Mock) -> None:
        """Test store initialization."""
        mock_pinecone_module = MagicMock()
        mock_pinecone = Mock()
        mock_index = Mock()
        mock_pinecone.list_indexes.return_value = ["test-index"]
        mock_pinecone.Index.return_value = mock_index
        mock_pinecone_module.Pinecone.return_value = mock_pinecone

        with patch.dict("sys.modules", {"pinecone": mock_pinecone_module}):
            from selectools.rag.stores.pinecone import PineconeVectorStore

            store = PineconeVectorStore(
                embedder=mock_embedder, index_name="test-index", api_key="test_key"
            )
            assert store.embedder == mock_embedder

    def test_add_documents(self, mock_embedder: Mock, sample_documents: list[Document]) -> None:
        """Test adding documents."""
        mock_pinecone_module = MagicMock()
        mock_pinecone = Mock()
        mock_index = Mock()
        mock_pinecone.list_indexes.return_value = []
        mock_pinecone.Index.return_value = mock_index
        mock_pinecone_module.Pinecone.return_value = mock_pinecone

        with patch.dict("sys.modules", {"pinecone": mock_pinecone_module}):
            from selectools.rag.stores.pinecone import PineconeVectorStore

            store = PineconeVectorStore(
                embedder=mock_embedder, index_name="test-index", api_key="test_key"
            )

            doc_ids = store.add_documents(sample_documents)

            assert len(doc_ids) == 3
            mock_index.upsert.assert_called_once()

    def test_search_basic(self, mock_embedder: Mock) -> None:
        """Test basic search."""
        mock_pinecone_module = MagicMock()
        mock_pinecone = Mock()
        mock_index = Mock()
        mock_pinecone.list_indexes.return_value = ["test-index"]
        mock_pinecone.Index.return_value = mock_index
        mock_pinecone_module.Pinecone.return_value = mock_pinecone

        # Mock query results
        mock_match1 = Mock()
        mock_match1.id = "id1"
        mock_match1.score = 0.9
        mock_match1.metadata = {"text": "doc1", "cat": "a"}

        mock_match2 = Mock()
        mock_match2.id = "id2"
        mock_match2.score = 0.7
        mock_match2.metadata = {"text": "doc2", "cat": "b"}

        mock_results = Mock()
        mock_results.matches = [mock_match1, mock_match2]
        mock_index.query.return_value = mock_results

        with patch.dict("sys.modules", {"pinecone": mock_pinecone_module}):
            from selectools.rag.stores.pinecone import PineconeVectorStore

            store = PineconeVectorStore(
                embedder=mock_embedder, index_name="test-index", api_key="test_key"
            )

            query_embedding = [0.1] * 128
            results = store.search(query_embedding, top_k=2)

            assert len(results) == 2
            assert results[0].score == 0.9
            mock_index.query.assert_called_once()


# ============================================================================
# Cross-Store Compatibility Tests
# ============================================================================


class TestVectorStoreCompatibility:
    """Test that all vector stores behave consistently."""

    def test_factory_method(self, mock_embedder: Mock) -> None:
        """Test VectorStore.create factory method."""
        # Memory store
        memory_store = VectorStore.create("memory", embedder=mock_embedder)
        assert isinstance(memory_store, InMemoryVectorStore)

        # SQLite store
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            sqlite_store = VectorStore.create("sqlite", embedder=mock_embedder, db_path=db_path)
            assert isinstance(sqlite_store, SQLiteVectorStore)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_invalid_backend(self, mock_embedder: Mock) -> None:
        """Test error on invalid backend name."""
        with pytest.raises(ValueError, match="Unknown vector store backend"):
            VectorStore.create("invalid_backend", embedder=mock_embedder)
