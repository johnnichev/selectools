"""
Regression tests for RAG module bug hunt — Pass 1.

Bug #1: VectorStore.create() missing faiss, qdrant, pgvector backends
Bug #2: QdrantVectorStore._build_filter uses bare keys instead of _st_meta.key
Bug #3: DocumentLoader.from_json turns None text_field values into "None" string
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from selectools.rag.vector_store import Document, SearchResult, VectorStore

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_embedder() -> Mock:
    """Create a mock embedding provider."""
    embedder = Mock()
    embedder.model = "mock-embedding-model"
    embedder.dimension = 4

    def mock_embed_texts(texts: List[str]) -> list[list[float]]:
        return [[float(hash(t) % 100 + i) / 100.0 for i in range(4)] for t in texts]

    def mock_embed_query(query: str) -> list[float]:
        return mock_embed_texts([query])[0]

    embedder.embed_texts.side_effect = mock_embed_texts
    embedder.embed_query.side_effect = mock_embed_query
    return embedder


# ============================================================================
# Bug #1: VectorStore.create() missing faiss, qdrant, pgvector
# ============================================================================


class TestVectorStoreCreateNewBackends:
    """VectorStore.create() must accept faiss, qdrant, and pgvector."""

    def test_create_faiss_backend(self, mock_embedder: Mock) -> None:
        """VectorStore.create('faiss') returns a FAISSVectorStore."""
        try:
            store = VectorStore.create("faiss", embedder=mock_embedder, dimension=4)
            from selectools.rag.stores.faiss import FAISSVectorStore

            assert isinstance(store, FAISSVectorStore)
        except ImportError:
            pytest.skip("faiss-cpu not installed")

    def test_create_qdrant_backend(self, mock_embedder: Mock) -> None:
        """VectorStore.create('qdrant') returns a QdrantVectorStore."""
        mock_qdrant = MagicMock()
        mock_qdrant.QdrantClient.return_value = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_qdrant.models,
            },
        ):
            store = VectorStore.create("qdrant", embedder=mock_embedder)
            from selectools.rag.stores.qdrant import QdrantVectorStore

            assert isinstance(store, QdrantVectorStore)

    def test_create_pgvector_backend(self, mock_embedder: Mock) -> None:
        """VectorStore.create('pgvector') returns a PgVectorStore."""
        mock_psycopg2 = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "psycopg2": mock_psycopg2,
                "psycopg2.extras": mock_psycopg2.extras,
            },
        ):
            store = VectorStore.create(
                "pgvector",
                embedder=mock_embedder,
                connection_string="postgresql://user:pass@localhost/db",
            )
            from selectools.rag.stores.pgvector import PgVectorStore

            assert isinstance(store, PgVectorStore)

    def test_create_invalid_backend_error_lists_all(self, mock_embedder: Mock) -> None:
        """Error message for unknown backend lists all 7 supported backends."""
        with pytest.raises(ValueError, match="faiss") as exc_info:
            VectorStore.create("nonexistent", embedder=mock_embedder)
        msg = str(exc_info.value)
        for backend in ("memory", "sqlite", "chroma", "pinecone", "faiss", "qdrant", "pgvector"):
            assert backend in msg, f"Missing backend {backend!r} in error message"


# ============================================================================
# Bug #2: QdrantVectorStore._build_filter must prefix keys with _st_meta.
# ============================================================================


class TestQdrantFilterKeyPrefix:
    """Filter keys must be prefixed with _st_meta. to match payload layout."""

    @pytest.fixture
    def qdrant_store_and_models(self, mock_embedder: Mock) -> Any:
        """Create a QdrantVectorStore with mocked qdrant_client."""
        mock_qdrant = MagicMock()
        mock_client = MagicMock()
        mock_qdrant.QdrantClient.return_value = mock_client

        # No existing collections
        coll_resp = MagicMock()
        coll_resp.collections = []
        mock_client.get_collections.return_value = coll_resp

        with patch.dict(
            sys.modules,
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_qdrant.models,
            },
        ):
            from selectools.rag.stores.qdrant import QdrantVectorStore

            store = QdrantVectorStore(
                embedder=mock_embedder,
                collection_name="test",
                api_key="key",
            )
            yield store, mock_qdrant.models

    def test_simple_dict_filter_keys_prefixed(self, qdrant_store_and_models: Any) -> None:
        """FieldCondition keys must start with '_st_meta.' for nested payload."""
        store, models = qdrant_store_and_models

        store._build_filter({"source": "test.txt", "category": "ai"})

        # Inspect the FieldCondition calls
        calls = models.FieldCondition.call_args_list
        assert len(calls) == 2

        keys_used = [c[1]["key"] if "key" in c[1] else c[0][0] for c in calls]
        assert "_st_meta.source" in keys_used
        assert "_st_meta.category" in keys_used
        # Bare keys must NOT be used
        assert "source" not in keys_used
        assert "category" not in keys_used


# ============================================================================
# Bug #3: DocumentLoader.from_json None text_field produces "None" documents
# ============================================================================


class TestJsonLoaderNoneTextField:
    """from_json must skip items where text_field value is None, not create 'None' docs."""

    def test_none_text_field_skipped(self, tmp_path: Any) -> None:
        """Items with text_field=None should be skipped, not produce 'None' text."""
        json_data = [
            {"text": "valid document", "id": 1},
            {"text": None, "id": 2},
            {"text": "", "id": 3},
            {"id": 4},  # missing text field entirely
        ]
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(json_data))

        from selectools.rag.loaders import DocumentLoader

        docs = DocumentLoader.from_json(str(json_file), text_field="text")

        # Only the first item has a valid non-empty text
        assert len(docs) == 1
        assert docs[0].text == "valid document"
        # Crucially, no document should have text "None" (string)
        for doc in docs:
            assert doc.text != "None", (
                "from_json must skip None values, not convert to 'None' string"
            )

    def test_numeric_text_field_converted(self, tmp_path: Any) -> None:
        """Numeric values in text_field should be converted to string (valid)."""
        json_data = [{"text": 42, "id": 1}]
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(json_data))

        from selectools.rag.loaders import DocumentLoader

        docs = DocumentLoader.from_json(str(json_file), text_field="text")

        assert len(docs) == 1
        assert docs[0].text == "42"
