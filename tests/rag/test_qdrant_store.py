"""
Unit tests for QdrantVectorStore.

All tests mock the qdrant_client module completely — no Qdrant server needed.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from selectools.rag.vector_store import Document, SearchResult

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
def mock_qdrant_models() -> MagicMock:
    """Create a mock qdrant_client.models module."""
    models = MagicMock()

    # VectorParams
    models.VectorParams = MagicMock()

    # Distance enum
    models.Distance = MagicMock()
    models.Distance.COSINE = "Cosine"

    # PointStruct — return a simple namespace so we can inspect fields
    def _make_point(id: Any, vector: Any, payload: Any) -> MagicMock:
        pt = MagicMock()
        pt.id = id
        pt.vector = vector
        pt.payload = payload
        return pt

    models.PointStruct.side_effect = _make_point

    # PointIdsList
    def _make_point_ids_list(points: Any) -> MagicMock:
        obj = MagicMock()
        obj.points = points
        return obj

    models.PointIdsList.side_effect = _make_point_ids_list

    # Filter / FieldCondition / MatchValue
    models.Filter = MagicMock()
    models.Filter.side_effect = lambda **kw: MagicMock(__class__=models.Filter, **kw)
    models.FieldCondition = MagicMock()
    models.MatchValue = MagicMock()

    return models


@pytest.fixture
def mock_qdrant_client_module(mock_qdrant_models: MagicMock) -> MagicMock:
    """Create a full mock qdrant_client package."""
    module = MagicMock()
    module.models = mock_qdrant_models

    # QdrantClient constructor returns a mock client instance
    client_instance = MagicMock()
    module.QdrantClient.return_value = client_instance

    # Default: no existing collections
    collections_response = MagicMock()
    collections_response.collections = []
    client_instance.get_collections.return_value = collections_response

    return module


@pytest.fixture
def qdrant_store(mock_embedder: Mock, mock_qdrant_client_module: MagicMock) -> Any:
    """Create a QdrantVectorStore with all Qdrant dependencies mocked."""
    with patch.dict(
        sys.modules,
        {
            "qdrant_client": mock_qdrant_client_module,
            "qdrant_client.models": mock_qdrant_client_module.models,
        },
    ):
        from selectools.rag.stores.qdrant import QdrantVectorStore

        store = QdrantVectorStore(
            embedder=mock_embedder,
            collection_name="test_collection",
            url="http://localhost:6333",
            api_key="test-key",
            prefer_grpc=True,
        )
        yield store


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
# Constructor Tests
# ============================================================================


class TestQdrantVectorStoreInit:
    """Tests for QdrantVectorStore initialization."""

    def test_constructor_initializes_client(
        self, mock_embedder: Mock, mock_qdrant_client_module: MagicMock
    ) -> None:
        """Constructor creates a QdrantClient with provided parameters."""
        with patch.dict(
            sys.modules,
            {
                "qdrant_client": mock_qdrant_client_module,
                "qdrant_client.models": mock_qdrant_client_module.models,
            },
        ):
            from selectools.rag.stores.qdrant import QdrantVectorStore

            store = QdrantVectorStore(
                embedder=mock_embedder,
                collection_name="my_coll",
                url="http://remote:6333",
                api_key="secret",
                prefer_grpc=False,
            )

            mock_qdrant_client_module.QdrantClient.assert_called_once_with(
                url="http://remote:6333",
                api_key="secret",
                prefer_grpc=False,
            )
            assert store.collection_name == "my_coll"
            assert store.url == "http://remote:6333"
            assert store.embedder is mock_embedder

    def test_constructor_default_parameters(
        self, mock_embedder: Mock, mock_qdrant_client_module: MagicMock
    ) -> None:
        """Constructor uses sensible defaults."""
        with patch.dict(
            sys.modules,
            {
                "qdrant_client": mock_qdrant_client_module,
                "qdrant_client.models": mock_qdrant_client_module.models,
            },
        ):
            from selectools.rag.stores.qdrant import QdrantVectorStore

            store = QdrantVectorStore(embedder=mock_embedder)

            assert store.collection_name == "selectools"
            assert store.url == "http://localhost:6333"

    def test_import_error_without_qdrant(self, mock_embedder: Mock) -> None:
        """Raises ImportError with helpful message when qdrant-client missing."""
        with patch.dict(sys.modules, {"qdrant_client": None}):
            with pytest.raises(ImportError, match="qdrant-client"):
                from selectools.rag.stores.qdrant import QdrantVectorStore

                QdrantVectorStore(embedder=mock_embedder)

    def test_collection_not_created_on_init(self, qdrant_store: Any) -> None:
        """Collection is NOT created during __init__ (lazy creation)."""
        client = qdrant_store.client
        client.create_collection.assert_not_called()
        assert qdrant_store._collection_exists is False


# ============================================================================
# add_documents Tests
# ============================================================================


class TestQdrantAddDocuments:
    """Tests for add_documents method."""

    def test_add_empty_list_returns_empty(self, qdrant_store: Any) -> None:
        """Adding an empty list returns an empty list without API calls."""
        result = qdrant_store.add_documents([])
        assert result == []
        qdrant_store.client.upsert.assert_not_called()

    def test_add_documents_creates_collection(
        self,
        qdrant_store: Any,
        sample_documents: list[Document],
        mock_qdrant_client_module: MagicMock,
    ) -> None:
        """First add_documents call auto-creates the collection."""
        ids = qdrant_store.add_documents(sample_documents)

        # Collection should have been created
        qdrant_store.client.create_collection.assert_called_once()
        call_kwargs = (
            qdrant_store.client.create_collection.call_kwargs
            if hasattr(qdrant_store.client.create_collection, "call_kwargs")
            else qdrant_store.client.create_collection.call_args
        )

        assert len(ids) == 3
        assert qdrant_store._collection_exists is True

    def test_add_documents_skips_create_on_second_call(
        self, qdrant_store: Any, sample_documents: list[Document]
    ) -> None:
        """Second add_documents call does not re-create collection."""
        qdrant_store.add_documents(sample_documents)
        qdrant_store.client.create_collection.reset_mock()

        qdrant_store.add_documents(sample_documents[:1])
        qdrant_store.client.create_collection.assert_not_called()

    def test_add_documents_skips_create_if_collection_exists(
        self,
        mock_embedder: Mock,
        mock_qdrant_client_module: MagicMock,
        sample_documents: list[Document],
    ) -> None:
        """Skips create_collection if the collection already exists on server."""
        # Simulate existing collection
        existing_coll = MagicMock()
        existing_coll.name = "test_collection"
        collections_response = MagicMock()
        collections_response.collections = [existing_coll]
        mock_qdrant_client_module.QdrantClient.return_value.get_collections.return_value = (
            collections_response
        )

        with patch.dict(
            sys.modules,
            {
                "qdrant_client": mock_qdrant_client_module,
                "qdrant_client.models": mock_qdrant_client_module.models,
            },
        ):
            from selectools.rag.stores.qdrant import QdrantVectorStore

            store = QdrantVectorStore(
                embedder=mock_embedder,
                collection_name="test_collection",
            )
            store.add_documents(sample_documents)

            store.client.create_collection.assert_not_called()

    def test_add_documents_computes_embeddings(
        self,
        qdrant_store: Any,
        mock_embedder: Mock,
        sample_documents: list[Document],
    ) -> None:
        """Embeddings are computed via the embedder when not provided."""
        qdrant_store.add_documents(sample_documents)

        mock_embedder.embed_texts.assert_called_once()
        texts = mock_embedder.embed_texts.call_args[0][0]
        assert len(texts) == 3

    def test_add_documents_uses_precomputed_embeddings(
        self,
        qdrant_store: Any,
        mock_embedder: Mock,
        sample_documents: list[Document],
    ) -> None:
        """Pre-computed embeddings bypass the embedder."""
        pre_embeddings = [[0.1] * 4, [0.2] * 4, [0.3] * 4]

        qdrant_store.add_documents(sample_documents, embeddings=pre_embeddings)

        mock_embedder.embed_texts.assert_not_called()

    def test_add_documents_upserts_points(
        self, qdrant_store: Any, sample_documents: list[Document]
    ) -> None:
        """Documents are upserted as Qdrant PointStruct objects."""
        qdrant_store.add_documents(sample_documents)

        qdrant_store.client.upsert.assert_called_once()
        call_kwargs = qdrant_store.client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"
        assert len(call_kwargs["points"]) == 3

    def test_add_documents_stores_text_in_payload(
        self, qdrant_store: Any, sample_documents: list[Document]
    ) -> None:
        """Document text is stored in payload under __selectools_text__."""
        qdrant_store.add_documents(sample_documents)

        call_kwargs = qdrant_store.client.upsert.call_args[1]
        point = call_kwargs["points"][0]
        assert point.payload["__selectools_text__"] == sample_documents[0].text

    def test_add_documents_preserves_metadata(
        self, qdrant_store: Any, sample_documents: list[Document]
    ) -> None:
        """User metadata is preserved in the payload."""
        qdrant_store.add_documents(sample_documents)

        call_kwargs = qdrant_store.client.upsert.call_args[1]
        point = call_kwargs["points"][0]
        assert point.payload["category"] == "animals"
        assert point.payload["source"] == "test1.txt"

    def test_add_documents_returns_deterministic_ids(
        self, qdrant_store: Any, sample_documents: list[Document]
    ) -> None:
        """IDs are deterministic (SHA256 based)."""
        ids1 = qdrant_store.add_documents(sample_documents)
        qdrant_store._collection_exists = True  # skip re-check
        ids2 = qdrant_store.add_documents(sample_documents)

        assert ids1 == ids2
        for doc_id in ids1:
            assert doc_id.startswith("doc_")


# ============================================================================
# search Tests
# ============================================================================


class TestQdrantSearch:
    """Tests for search method."""

    def test_search_returns_results(
        self,
        qdrant_store: Any,
        mock_qdrant_client_module: MagicMock,
    ) -> None:
        """Search returns properly structured SearchResult objects."""
        # Set up mock search results
        scored_point = MagicMock()
        scored_point.payload = {
            "__selectools_text__": "Hello world",
            "source": "test.txt",
        }
        scored_point.score = 0.95
        qdrant_store.client.search.return_value = [scored_point]

        query_emb = [0.1] * 128
        results = qdrant_store.search(query_emb, top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].document.text == "Hello world"
        assert results[0].document.metadata == {"source": "test.txt"}
        assert results[0].score == 0.95

    def test_search_passes_correct_parameters(self, qdrant_store: Any) -> None:
        """Search forwards collection name, vector, limit, and payload flag."""
        qdrant_store.client.search.return_value = []

        query_emb = [0.5] * 128
        qdrant_store.search(query_emb, top_k=10)

        qdrant_store.client.search.assert_called_once()
        call_kwargs = qdrant_store.client.search.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"
        assert call_kwargs["query_vector"] == query_emb
        assert call_kwargs["limit"] == 10
        assert call_kwargs["with_payload"] is True

    def test_search_empty_results(self, qdrant_store: Any) -> None:
        """Search returns empty list when no matches found."""
        qdrant_store.client.search.return_value = []

        results = qdrant_store.search([0.1] * 128)
        assert results == []

    def test_search_strips_internal_metadata(self, qdrant_store: Any) -> None:
        """__selectools_text__ is stripped from metadata in results."""
        scored_point = MagicMock()
        scored_point.payload = {
            "__selectools_text__": "Some text",
            "author": "Alice",
        }
        scored_point.score = 0.8
        qdrant_store.client.search.return_value = [scored_point]

        results = qdrant_store.search([0.1] * 128)

        assert "__selectools_text__" not in results[0].document.metadata
        assert results[0].document.metadata["author"] == "Alice"

    def test_search_handles_none_payload(self, qdrant_store: Any) -> None:
        """Search handles points with None payload gracefully."""
        scored_point = MagicMock()
        scored_point.payload = None
        scored_point.score = 0.5
        qdrant_store.client.search.return_value = [scored_point]

        results = qdrant_store.search([0.1] * 128)

        assert len(results) == 1
        assert results[0].document.text == ""
        assert results[0].document.metadata == {}

    def test_search_with_simple_filter(
        self, qdrant_store: Any, mock_qdrant_client_module: MagicMock
    ) -> None:
        """Simple dict filters are converted to Qdrant Filter objects."""
        qdrant_store.client.search.return_value = []

        qdrant_store.search(
            [0.1] * 128,
            top_k=5,
            filter={"category": "ai"},
        )

        call_kwargs = qdrant_store.client.search.call_args[1]
        # Filter should have been converted (not None)
        assert call_kwargs["query_filter"] is not None

    def test_search_with_no_filter(self, qdrant_store: Any) -> None:
        """Search with no filter passes None as query_filter."""
        qdrant_store.client.search.return_value = []

        qdrant_store.search([0.1] * 128, top_k=5)

        call_kwargs = qdrant_store.client.search.call_args[1]
        assert call_kwargs["query_filter"] is None

    def test_search_multiple_results_ordering(self, qdrant_store: Any) -> None:
        """Multiple results maintain order from Qdrant response."""
        points = []
        for _i, (text, score) in enumerate(
            [
                ("First", 0.99),
                ("Second", 0.85),
                ("Third", 0.70),
            ]
        ):
            pt = MagicMock()
            pt.payload = {"__selectools_text__": text}
            pt.score = score
            points.append(pt)

        qdrant_store.client.search.return_value = points

        results = qdrant_store.search([0.1] * 128, top_k=3)

        assert len(results) == 3
        assert results[0].score == 0.99
        assert results[1].score == 0.85
        assert results[2].score == 0.70


# ============================================================================
# delete Tests
# ============================================================================


class TestQdrantDelete:
    """Tests for delete method."""

    def test_delete_calls_client(self, qdrant_store: Any) -> None:
        """Delete forwards IDs to the Qdrant client."""
        qdrant_store.delete(["id1", "id2"])

        qdrant_store.client.delete.assert_called_once()
        call_kwargs = qdrant_store.client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"

    def test_delete_empty_list_noop(self, qdrant_store: Any) -> None:
        """Deleting an empty list does not call the client."""
        qdrant_store.delete([])
        qdrant_store.client.delete.assert_not_called()


# ============================================================================
# clear Tests
# ============================================================================


class TestQdrantClear:
    """Tests for clear method."""

    def test_clear_deletes_collection(self, qdrant_store: Any) -> None:
        """Clear deletes the collection on the server."""
        qdrant_store._collection_exists = True

        qdrant_store.clear()

        qdrant_store.client.delete_collection.assert_called_once_with(
            collection_name="test_collection"
        )
        assert qdrant_store._collection_exists is False

    def test_clear_handles_missing_collection(self, qdrant_store: Any) -> None:
        """Clear does not raise if the collection doesn't exist."""
        qdrant_store.client.delete_collection.side_effect = Exception("Not found")

        # Should not raise
        qdrant_store.clear()
        assert qdrant_store._collection_exists is False

    def test_clear_then_add_recreates_collection(
        self,
        qdrant_store: Any,
        sample_documents: list[Document],
    ) -> None:
        """After clear(), add_documents() recreates the collection."""
        # First add to create collection
        qdrant_store.add_documents(sample_documents)
        assert qdrant_store._collection_exists is True

        # Clear
        qdrant_store.clear()
        assert qdrant_store._collection_exists is False

        # Add again — should trigger create_collection
        qdrant_store.client.create_collection.reset_mock()
        qdrant_store.add_documents(sample_documents[:1])
        qdrant_store.client.create_collection.assert_called_once()


# ============================================================================
# Metadata filtering Tests
# ============================================================================


class TestQdrantFiltering:
    """Tests for metadata filter construction."""

    def test_build_filter_none(self, qdrant_store: Any) -> None:
        """None filter returns None."""
        result = qdrant_store._build_filter(None)
        assert result is None

    def test_build_filter_simple_dict(
        self, qdrant_store: Any, mock_qdrant_client_module: MagicMock
    ) -> None:
        """Simple dict is converted to Filter with FieldCondition entries."""
        models = mock_qdrant_client_module.models

        result = qdrant_store._build_filter({"source": "test.txt", "lang": "en"})

        assert result is not None
        # FieldCondition should have been called twice
        assert models.FieldCondition.call_count == 2
        assert models.MatchValue.call_count == 2

    def test_build_filter_native_passthrough(
        self, qdrant_store: Any, mock_qdrant_client_module: MagicMock
    ) -> None:
        """Native Qdrant Filter objects are passed through unchanged."""
        models = mock_qdrant_client_module.models

        native_filter = models.Filter(must=[])
        # isinstance check — we need the mock to identify as Filter
        native_filter.__class__ = models.Filter

        result = qdrant_store._build_filter(native_filter)
        assert result is native_filter


# ============================================================================
# Utility Tests
# ============================================================================


class TestQdrantUtilities:
    """Tests for utility methods."""

    def test_count_returns_points_count(self, qdrant_store: Any) -> None:
        """count() returns the collection's points_count."""
        info = MagicMock()
        info.points_count = 42
        qdrant_store.client.get_collection.return_value = info

        assert qdrant_store.count() == 42

    def test_count_returns_zero_on_error(self, qdrant_store: Any) -> None:
        """count() returns 0 if the collection doesn't exist."""
        qdrant_store.client.get_collection.side_effect = Exception("Not found")

        assert qdrant_store.count() == 0

    def test_repr(self, qdrant_store: Any) -> None:
        """__repr__ includes collection name and URL."""
        r = repr(qdrant_store)
        assert "test_collection" in r
        assert "localhost:6333" in r

    def test_beta_stability_marker(self, mock_qdrant_client_module: MagicMock) -> None:
        """QdrantVectorStore has @beta stability marker."""
        with patch.dict(
            sys.modules,
            {
                "qdrant_client": mock_qdrant_client_module,
                "qdrant_client.models": mock_qdrant_client_module.models,
            },
        ):
            from selectools.rag.stores.qdrant import QdrantVectorStore

            assert getattr(QdrantVectorStore, "__stability__", None) == "beta"
