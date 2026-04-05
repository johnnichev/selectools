"""Tests for PgVectorStore — fully mocked, no PostgreSQL required."""

from __future__ import annotations

import json
import sys
import types
from typing import Any, List
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Build a fake psycopg2 module so the real one is never needed
# ---------------------------------------------------------------------------


def _make_mock_psycopg2() -> types.ModuleType:
    """Create a mock psycopg2 module with extras submodule."""
    mod = types.ModuleType("psycopg2")
    extras = types.ModuleType("psycopg2.extras")
    mod.extras = extras  # type: ignore[attr-defined]
    mod.connect = MagicMock()
    sys.modules["psycopg2"] = mod
    sys.modules["psycopg2.extras"] = extras
    return mod


_mock_pg = _make_mock_psycopg2()


# Now we can safely import the store
from selectools.rag.stores.pgvector import PgVectorStore  # noqa: E402
from selectools.rag.vector_store import Document, SearchResult  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Minimal embedder stub."""

    def __init__(self, dim: int = 3) -> None:
        self.dim = dim

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * self.dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * self.dim for _ in texts]


class FakeCursor:
    """Simulates a psycopg2 cursor with recorded SQL."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, Any]] = []
        self._fetchall_result: list[Any] = []
        self._fetchone_result: Any = None

    def execute(self, sql: str, params: Any = None) -> None:
        self.queries.append((sql, params))

    def fetchall(self) -> list[Any]:
        return self._fetchall_result

    def fetchone(self) -> Any:
        return self._fetchone_result


class FakeConnection:
    """Simulates a psycopg2 connection."""

    def __init__(self) -> None:
        self.autocommit = False
        self._cursor = FakeCursor()
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _make_store(
    dim: int = 3,
    table_name: str = "selectools_documents",
    pre_initialized: bool = False,
) -> tuple[PgVectorStore, FakeConnection]:
    """Create a PgVectorStore backed by fakes."""
    embedder = FakeEmbedder(dim=dim)
    conn = FakeConnection()
    _mock_pg.connect = MagicMock(return_value=conn)

    store = PgVectorStore(
        embedder=embedder,
        connection_string="postgresql://test:test@localhost/testdb",
        table_name=table_name,
        dimensions=dim,
    )
    if pre_initialized:
        store._initialized = True
    return store, conn


# ===========================================================================
# Test: initialisation & schema
# ===========================================================================


class TestInit:
    """Tests for constructor and schema initialisation."""

    def test_constructor_stores_fields(self) -> None:
        store, _ = _make_store(dim=5, table_name="my_docs")
        assert store.table_name == "my_docs"
        assert store._dimensions == 5
        assert store._initialized is False

    def test_ensure_initialized_creates_extension_and_table(self) -> None:
        store, conn = _make_store(dim=4)
        store._ensure_initialized()

        sqls = [sql.strip() for sql, _params in conn._cursor.queries]
        sql_text = "\n".join(sqls)

        assert "CREATE EXTENSION IF NOT EXISTS vector" in sql_text
        assert "CREATE TABLE IF NOT EXISTS selectools_documents" in sql_text
        assert "vector(4)" in sql_text
        assert "CREATE INDEX IF NOT EXISTS" in sql_text
        assert "hnsw" in sql_text.lower()
        assert store._initialized is True

    def test_ensure_initialized_idempotent(self) -> None:
        store, conn = _make_store(dim=3)
        store._ensure_initialized()
        query_count_after_first = len(conn._cursor.queries)

        # Second call should be a no-op
        store._ensure_initialized()
        assert len(conn._cursor.queries) == query_count_after_first

    def test_auto_detect_dimensions(self) -> None:
        embedder = FakeEmbedder(dim=7)
        conn = FakeConnection()
        _mock_pg.connect = MagicMock(return_value=conn)

        store = PgVectorStore(
            embedder=embedder,
            connection_string="postgresql://test:test@localhost/testdb",
            dimensions=None,  # force auto-detect
        )
        store._ensure_initialized()

        assert store._dimensions == 7
        sqls = " ".join(q[0] for q in conn._cursor.queries)
        assert "vector(7)" in sqls

    def test_invalid_table_name_special_chars(self) -> None:
        with pytest.raises(ValueError, match="Invalid table name"):
            PgVectorStore(
                embedder=FakeEmbedder(),
                connection_string="postgresql://x@localhost/db",
                table_name="drop;--",
            )

    def test_invalid_table_name_starts_with_digit(self) -> None:
        with pytest.raises(ValueError, match="must not start with a digit"):
            PgVectorStore(
                embedder=FakeEmbedder(),
                connection_string="postgresql://x@localhost/db",
                table_name="1bad",
            )

    def test_empty_table_name(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            PgVectorStore(
                embedder=FakeEmbedder(),
                connection_string="postgresql://x@localhost/db",
                table_name="",
            )

    def test_valid_table_name_with_underscores(self) -> None:
        store, _ = _make_store(table_name="my_custom_table_123")
        assert store.table_name == "my_custom_table_123"


# ===========================================================================
# Test: add_documents
# ===========================================================================


class TestAddDocuments:
    """Tests for add_documents."""

    def test_add_empty_list_returns_empty(self) -> None:
        store, _ = _make_store(pre_initialized=True)
        result = store.add_documents([])
        assert result == []

    def test_add_single_document(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        docs = [Document(text="hello world", metadata={"source": "test"})]
        ids = store.add_documents(docs)

        assert len(ids) == 1
        assert ids[0].startswith("doc_")
        assert conn.committed is True
        assert conn.closed is True

    def test_add_multiple_documents(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        docs = [
            Document(text="first", metadata={"i": 1}),
            Document(text="second", metadata={"i": 2}),
            Document(text="third", metadata={"i": 3}),
        ]
        ids = store.add_documents(docs)
        assert len(ids) == 3
        assert len(set(ids)) == 3  # all unique

    def test_add_with_precomputed_embeddings(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        docs = [Document(text="hello")]
        embeddings = [[0.5, 0.6, 0.7]]
        ids = store.add_documents(docs, embeddings=embeddings)

        assert len(ids) == 1
        # Verify the embedding was used in the SQL
        insert_query = conn._cursor.queries[0]
        assert "[0.5,0.6,0.7]" in str(insert_query[1])

    def test_add_mismatched_embeddings_raises(self) -> None:
        store, _ = _make_store(dim=3, pre_initialized=True)
        docs = [Document(text="a"), Document(text="b")]
        embeddings = [[0.1, 0.2, 0.3]]  # only 1 embedding for 2 docs
        with pytest.raises(ValueError, match="does not match"):
            store.add_documents(docs, embeddings=embeddings)

    def test_add_uses_parameterized_queries(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        docs = [Document(text="safe'; DROP TABLE --", metadata={"x": 1})]
        store.add_documents(docs)

        insert_sql, insert_params = conn._cursor.queries[0]
        # The text should be in params, not interpolated into SQL
        assert "%s" in insert_sql
        assert "safe'; DROP TABLE --" in insert_params

    def test_add_rollback_on_error(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)

        # Make execute raise on the insert
        original_execute = conn._cursor.execute
        call_count = [0]

        def failing_execute(sql: str, params: Any = None) -> None:
            call_count[0] += 1
            raise RuntimeError("DB error")

        conn._cursor.execute = failing_execute

        with pytest.raises(RuntimeError, match="DB error"):
            store.add_documents([Document(text="fail")])

        assert conn.rolled_back is True
        assert conn.closed is True


# ===========================================================================
# Test: search
# ===========================================================================


class TestSearch:
    """Tests for search."""

    def test_search_basic(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)

        conn._cursor._fetchall_result = [
            ("doc_1", "hello world", {"source": "test"}, 0.95),
            ("doc_2", "goodbye", {"source": "other"}, 0.80),
        ]

        results = store.search([0.1, 0.2, 0.3], top_k=2)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].document.text == "hello world"
        assert results[0].score == 0.95
        assert results[1].document.text == "goodbye"
        assert results[1].score == 0.80

    def test_search_with_filter(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        conn._cursor._fetchall_result = [
            ("doc_1", "matched", {"source": "pdf"}, 0.9),
        ]

        results = store.search(
            [0.1, 0.2, 0.3],
            top_k=5,
            filter={"source": "pdf"},
        )

        # Verify the filter clause is present in the SQL
        search_sql = conn._cursor.queries[0][0]
        assert "metadata @> %s::jsonb" in search_sql

        # Verify the filter JSON is passed as parameter
        search_params = conn._cursor.queries[0][1]
        assert json.dumps({"source": "pdf"}) in search_params

    def test_search_uses_cosine_distance_operator(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        conn._cursor._fetchall_result = []

        store.search([0.1, 0.2, 0.3], top_k=3)

        search_sql = conn._cursor.queries[0][0]
        assert "<=>" in search_sql
        assert "ORDER BY" in search_sql
        assert "LIMIT" in search_sql

    def test_search_parameterized_queries(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        conn._cursor._fetchall_result = []

        store.search([0.1, 0.2, 0.3], top_k=5)

        search_sql, search_params = conn._cursor.queries[0]
        assert "%s" in search_sql
        # top_k should be in params, not interpolated
        assert 5 in search_params

    def test_search_metadata_as_string(self) -> None:
        """psycopg2 may return JSONB as a string or dict."""
        store, conn = _make_store(dim=3, pre_initialized=True)
        conn._cursor._fetchall_result = [
            ("doc_1", "text", '{"key": "value"}', 0.9),
        ]

        results = store.search([0.1, 0.2, 0.3])
        assert results[0].document.metadata == {"key": "value"}

    def test_search_metadata_as_dict(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        conn._cursor._fetchall_result = [
            ("doc_1", "text", {"key": "value"}, 0.9),
        ]

        results = store.search([0.1, 0.2, 0.3])
        assert results[0].document.metadata == {"key": "value"}

    def test_search_metadata_none(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        conn._cursor._fetchall_result = [
            ("doc_1", "text", None, 0.85),
        ]

        results = store.search([0.1, 0.2, 0.3])
        assert results[0].document.metadata == {}

    def test_search_empty_results(self) -> None:
        store, conn = _make_store(dim=3, pre_initialized=True)
        conn._cursor._fetchall_result = []

        results = store.search([0.1, 0.2, 0.3], top_k=10)
        assert results == []


# ===========================================================================
# Test: delete
# ===========================================================================


class TestDelete:
    """Tests for delete."""

    def test_delete_empty_list_is_noop(self) -> None:
        store, conn = _make_store(pre_initialized=True)
        store.delete([])
        # No queries should have been executed
        assert len(conn._cursor.queries) == 0

    def test_delete_single_id(self) -> None:
        store, conn = _make_store(pre_initialized=True)
        store.delete(["doc_abc"])

        sql, params = conn._cursor.queries[0]
        assert "DELETE FROM" in sql
        assert "WHERE id IN" in sql
        assert "%s" in sql
        assert "doc_abc" in params
        assert conn.committed is True

    def test_delete_multiple_ids(self) -> None:
        store, conn = _make_store(pre_initialized=True)
        ids_to_delete = ["doc_1", "doc_2", "doc_3"]
        store.delete(ids_to_delete)

        sql, params = conn._cursor.queries[0]
        assert sql.count("%s") == 3
        assert list(params) == ids_to_delete

    def test_delete_rollback_on_error(self) -> None:
        store, conn = _make_store(pre_initialized=True)

        def failing_execute(sql: str, params: Any = None) -> None:
            raise RuntimeError("delete failed")

        conn._cursor.execute = failing_execute

        with pytest.raises(RuntimeError, match="delete failed"):
            store.delete(["doc_x"])

        assert conn.rolled_back is True
        assert conn.closed is True


# ===========================================================================
# Test: clear
# ===========================================================================


class TestClear:
    """Tests for clear."""

    def test_clear(self) -> None:
        store, conn = _make_store(pre_initialized=True)
        store.clear()

        sql = conn._cursor.queries[0][0]
        assert "DELETE FROM selectools_documents" in sql
        assert conn.committed is True
        assert conn.closed is True

    def test_clear_rollback_on_error(self) -> None:
        store, conn = _make_store(pre_initialized=True)

        def failing_execute(sql: str, params: Any = None) -> None:
            raise RuntimeError("clear failed")

        conn._cursor.execute = failing_execute

        with pytest.raises(RuntimeError, match="clear failed"):
            store.clear()

        assert conn.rolled_back is True


# ===========================================================================
# Test: utility methods
# ===========================================================================


class TestUtilities:
    """Tests for count, get, and repr."""

    def test_count(self) -> None:
        store, conn = _make_store(pre_initialized=True)
        conn._cursor._fetchone_result = (42,)

        assert store.count() == 42

    def test_count_empty(self) -> None:
        store, conn = _make_store(pre_initialized=True)
        conn._cursor._fetchone_result = (0,)

        assert store.count() == 0

    def test_get_existing(self) -> None:
        store, conn = _make_store(pre_initialized=True)
        conn._cursor._fetchone_result = ("doc_1", "hello", {"src": "a"})

        doc = store.get("doc_1")
        assert doc is not None
        assert doc.text == "hello"
        assert doc.metadata == {"src": "a"}

    def test_get_missing(self) -> None:
        store, conn = _make_store(pre_initialized=True)
        conn._cursor._fetchone_result = None

        assert store.get("nonexistent") is None

    def test_repr(self) -> None:
        store, _ = _make_store(dim=128, table_name="my_vecs")
        r = repr(store)
        assert "PgVectorStore" in r
        assert "my_vecs" in r
        assert "128" in r


# ===========================================================================
# Test: stability marker
# ===========================================================================


class TestStability:
    """Verify the @beta marker is applied."""

    def test_beta_marker(self) -> None:
        assert hasattr(PgVectorStore, "__stability__")
        assert PgVectorStore.__stability__ == "beta"


# ===========================================================================
# Test: VectorStore ABC compliance
# ===========================================================================


class TestABCCompliance:
    """Verify PgVectorStore is a valid VectorStore subclass."""

    def test_is_subclass(self) -> None:
        from selectools.rag.vector_store import VectorStore

        assert issubclass(PgVectorStore, VectorStore)

    def test_instance(self) -> None:
        from selectools.rag.vector_store import VectorStore

        store, _ = _make_store()
        assert isinstance(store, VectorStore)


# ===========================================================================
# Test: lazy import in __init__.py
# ===========================================================================


class TestLazyImport:
    """Verify PgVectorStore is accessible from the stores package."""

    def test_pgvector_in_stores_all(self) -> None:
        from selectools.rag import stores

        assert "PgVectorStore" in stores.__all__

    def test_import_from_stores(self) -> None:
        from selectools.rag.stores import PgVectorStore as Imported

        assert Imported is PgVectorStore
