"""Tests for selectools.checkpoint_postgres — PostgresCheckpointStore with mocked psycopg2."""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest

from selectools.orchestration.checkpoint import CheckpointMetadata
from selectools.orchestration.state import GraphState

# ===================================================================
# Fake psycopg2 module
# ===================================================================


def _make_fake_psycopg2():
    """Create a fake psycopg2 module with a mock connection."""
    mod = ModuleType("psycopg2")

    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.autocommit = False

    mod.connect = MagicMock(return_value=mock_conn)
    return mod, mock_conn, mock_cursor


# ===================================================================
# Helper: import with fake psycopg2
# ===================================================================


def _import_store(fake_psycopg2_mod):
    """Import PostgresCheckpointStore with fake psycopg2 in sys.modules."""
    with patch.dict(sys.modules, {"psycopg2": fake_psycopg2_mod}):
        # Force re-import to pick up our fake
        if "selectools.checkpoint_postgres" in sys.modules:
            del sys.modules["selectools.checkpoint_postgres"]
        from selectools.checkpoint_postgres import PostgresCheckpointStore

        return PostgresCheckpointStore


def _make_store(fake_psycopg2_mod):
    """Create a PostgresCheckpointStore instance with mocked psycopg2."""
    cls = _import_store(fake_psycopg2_mod)
    with patch.dict(sys.modules, {"psycopg2": fake_psycopg2_mod}):
        store = cls(dsn="postgresql://test:test@localhost/testdb")
    return store


def _make_state(**kwargs) -> GraphState:
    """Build a minimal GraphState for testing."""
    state = GraphState()
    state.current_node = kwargs.get("current_node", "node_a")
    for k, v in kwargs.get("data", {}).items():
        state.data[k] = v
    for k, v in kwargs.get("metadata", {}).items():
        state.metadata[k] = v
    return state


# ===================================================================
# Tests
# ===================================================================


class TestPostgresCheckpointStoreInit:
    """Tests for initialization and table creation."""

    def test_creates_table_on_init(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)
        # _init_table should have been called, executing CREATE TABLE and CREATE INDEX
        assert mock_cursor.execute.call_count >= 2
        sql_calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("CREATE TABLE" in s for s in sql_calls)
        assert any("CREATE INDEX" in s for s in sql_calls)

    def test_autocommit_set(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        _make_store(fake_pg)
        assert mock_conn.autocommit is True

    def test_invalid_table_name_raises(self):
        fake_pg, _, _ = _make_fake_psycopg2()
        cls = _import_store(fake_pg)
        with patch.dict(sys.modules, {"psycopg2": fake_pg}):
            with pytest.raises(ValueError, match="Invalid table name"):
                cls(dsn="postgresql://x@y/z", table="drop table; --")

    def test_valid_table_name_accepted(self):
        fake_pg, _, _ = _make_fake_psycopg2()
        cls = _import_store(fake_pg)
        with patch.dict(sys.modules, {"psycopg2": fake_pg}):
            store = cls(dsn="postgresql://x@y/z", table="my_checkpoints_v2")
            assert store._table == "my_checkpoints_v2"

    def test_custom_table_name(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        cls = _import_store(fake_pg)
        with patch.dict(sys.modules, {"psycopg2": fake_pg}):
            store = cls(dsn="postgresql://x@y/z", table="custom_cp")
        sql_calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("custom_cp" in s for s in sql_calls)

    def test_import_error_no_psycopg2(self):
        """When neither psycopg2 nor psycopg2cffi is available, ImportError is raised."""
        with patch.dict(sys.modules, {"psycopg2": None, "psycopg2cffi": None}):
            if "selectools.checkpoint_postgres" in sys.modules:
                del sys.modules["selectools.checkpoint_postgres"]
            with pytest.raises(ImportError, match="psycopg2"):
                from selectools.checkpoint_postgres import PostgresCheckpointStore

                PostgresCheckpointStore(dsn="postgresql://x@y/z")

    def test_fallback_to_psycopg2cffi(self):
        """When psycopg2 is not available but psycopg2cffi is, it should work."""
        fake_cffi = ModuleType("psycopg2cffi")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        fake_cffi.connect = MagicMock(return_value=mock_conn)

        with patch.dict(sys.modules, {"psycopg2": None, "psycopg2cffi": fake_cffi}):
            if "selectools.checkpoint_postgres" in sys.modules:
                del sys.modules["selectools.checkpoint_postgres"]
            from selectools.checkpoint_postgres import PostgresCheckpointStore

            store = PostgresCheckpointStore(dsn="postgresql://x@y/z")
            assert store is not None


class TestPostgresCheckpointStoreSave:
    """Tests for save()."""

    def test_save_returns_checkpoint_id(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)
        state = _make_state()

        checkpoint_id = store.save("graph-1", state, step=3)
        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) == 32  # uuid4().hex

    def test_save_inserts_row(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)
        state = _make_state(current_node="process")

        # Reset call history after init
        mock_cursor.execute.reset_mock()

        store.save("graph-1", state, step=5)

        # Should have called INSERT
        insert_calls = [c for c in mock_cursor.execute.call_args_list if "INSERT" in str(c)]
        assert len(insert_calls) == 1

        # Check the args
        insert_call = insert_calls[0]
        args = insert_call[0]  # positional args to execute()
        sql = args[0]
        params = args[1]
        assert "INSERT INTO checkpoints" in sql
        assert params[1] == "graph-1"  # graph_id
        assert params[2] == 5  # step
        assert params[3] == "process"  # node_name

    def test_save_with_interrupted_state(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)
        state = _make_state(metadata={"__pending_interrupt_key__": "ask_user"})

        mock_cursor.execute.reset_mock()
        store.save("graph-2", state, step=1)

        insert_calls = [c for c in mock_cursor.execute.call_args_list if "INSERT" in str(c)]
        params = insert_calls[0][0][1]
        assert params[4] is True  # interrupted flag

    def test_save_state_json_is_valid_json(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)
        state = _make_state(data={"count": 42, "items": ["a", "b"]})

        mock_cursor.execute.reset_mock()
        store.save("graph-3", state, step=2)

        insert_calls = [c for c in mock_cursor.execute.call_args_list if "INSERT" in str(c)]
        state_json = insert_calls[0][0][1][5]  # 6th param
        parsed = json.loads(state_json)
        assert "__step__" in parsed
        assert parsed["__step__"] == 2


class TestPostgresCheckpointStoreLoad:
    """Tests for load()."""

    def test_load_not_found_raises(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        mock_cursor.fetchone.return_value = None

        with pytest.raises(ValueError, match="not found"):
            store.load("nonexistent-id")

    def test_load_returns_state_and_step(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        # Simulate stored data
        stored_data = {
            "data": {"key": "value"},
            "metadata": {},
            "current_node": "node_b",
            "history": [],
            "__step__": 7,
            "__interrupt__": {},
            "__node_name__": "node_b",
            "__interrupted__": False,
        }
        mock_cursor.fetchone.return_value = (json.dumps(stored_data),)

        state, step = store.load("some-id")
        assert step == 7
        assert isinstance(state, GraphState)

    def test_load_with_dict_row(self):
        """When psycopg2 returns a dict instead of string (JSONB)."""
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        stored_data = {
            "data": {},
            "metadata": {},
            "current_node": "",
            "history": [],
            "__step__": 3,
            "__interrupt__": {},
            "__node_name__": "",
            "__interrupted__": False,
        }
        # Return as dict (JSONB auto-deserialized by psycopg2)
        mock_cursor.fetchone.return_value = (stored_data,)

        state, step = store.load("dict-row-id")
        assert step == 3


class TestPostgresCheckpointStoreList:
    """Tests for list()."""

    def test_list_empty(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        mock_cursor.fetchall.return_value = []
        results = store.list("graph-1")
        assert results == []

    def test_list_returns_metadata(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        now = datetime.now(timezone.utc)
        mock_cursor.fetchall.return_value = [
            ("cp-1", "graph-1", 1, "start", False, now),
            ("cp-2", "graph-1", 5, "end", True, now),
        ]

        results = store.list("graph-1")
        assert len(results) == 2
        assert all(isinstance(r, CheckpointMetadata) for r in results)
        assert results[0].checkpoint_id == "cp-1"
        assert results[0].step == 1
        assert results[0].node_name == "start"
        assert results[0].interrupted is False
        assert results[1].checkpoint_id == "cp-2"
        assert results[1].interrupted is True

    def test_list_string_datetime_parsing(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        # Sometimes the driver returns ISO string instead of datetime
        mock_cursor.fetchall.return_value = [
            ("cp-1", "graph-1", 1, "node", False, "2024-06-15T10:30:00+00:00"),
        ]

        results = store.list("graph-1")
        assert len(results) == 1
        assert isinstance(results[0].created_at, datetime)


class TestPostgresCheckpointStoreDelete:
    """Tests for delete()."""

    def test_delete_existing(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        mock_cursor.rowcount = 1
        result = store.delete("cp-1")
        assert result is True

    def test_delete_nonexistent(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        mock_cursor.rowcount = 0
        result = store.delete("nonexistent")
        assert result is False

    def test_delete_executes_correct_sql(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        mock_cursor.rowcount = 1
        mock_cursor.execute.reset_mock()
        store.delete("cp-42")

        delete_calls = [c for c in mock_cursor.execute.call_args_list if "DELETE" in str(c)]
        assert len(delete_calls) == 1
        params = delete_calls[0][0][1]
        assert params == ("cp-42",)


class TestPostgresCheckpointStoreClose:
    """Tests for close()."""

    def test_close_calls_conn_close(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        store.close()
        mock_conn.close.assert_called_once()


class TestPostgresCheckpointStoreThreadSafety:
    """Verify the lock is used."""

    def test_save_uses_lock(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        original_lock = store._lock
        acquired = []

        class TrackingLock:
            def __enter__(self):
                acquired.append(True)
                return original_lock.__enter__()

            def __exit__(self, *args):
                return original_lock.__exit__(*args)

        store._lock = TrackingLock()
        state = _make_state()
        store.save("g1", state, step=1)
        assert len(acquired) == 1

    def test_load_uses_lock(self):
        fake_pg, mock_conn, mock_cursor = _make_fake_psycopg2()
        store = _make_store(fake_pg)

        mock_cursor.fetchone.return_value = (
            json.dumps(
                {
                    "data": {},
                    "metadata": {},
                    "current_node": "",
                    "history": [],
                    "__step__": 0,
                    "__interrupt__": {},
                }
            ),
        )

        original_lock = store._lock
        acquired = []

        class TrackingLock:
            def __enter__(self):
                acquired.append(True)
                return original_lock.__enter__()

            def __exit__(self, *args):
                return original_lock.__exit__(*args)

        store._lock = TrackingLock()
        store.load("x")
        assert len(acquired) == 1


class TestPostgresCheckpointStoreAllExport:
    """Verify __all__ export."""

    def test_all_contains_store(self):
        fake_pg, _, _ = _make_fake_psycopg2()
        with patch.dict(sys.modules, {"psycopg2": fake_pg}):
            if "selectools.checkpoint_postgres" in sys.modules:
                del sys.modules["selectools.checkpoint_postgres"]
            import selectools.checkpoint_postgres as mod

            assert "PostgresCheckpointStore" in mod.__all__
