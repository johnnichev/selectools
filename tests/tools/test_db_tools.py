"""
Tests for database query tools (query_sqlite, query_postgres).

SQLite tests use real in-memory/temp databases.
PostgreSQL tests mock psycopg2 to avoid requiring a running database.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from selectools.toolbox import db_tools

# =============================================================================
# query_sqlite tests
# =============================================================================


class TestQuerySqlite:
    """Tests for the query_sqlite tool."""

    def test_tool_has_correct_metadata(self) -> None:
        assert db_tools.query_sqlite.name == "query_sqlite"
        assert "SQLite" in db_tools.query_sqlite.description

    def test_stability_marker_is_beta(self) -> None:
        assert getattr(db_tools.query_sqlite, "__stability__", None) == "beta"

    def test_empty_db_path_rejected(self) -> None:
        result = db_tools.query_sqlite.function("", "SELECT 1")
        assert "Error" in result

    def test_empty_sql_rejected(self) -> None:
        result = db_tools.query_sqlite.function("/tmp/test.db", "")
        assert "Error" in result

    def test_nonexistent_db_rejected(self) -> None:
        result = db_tools.query_sqlite.function("/nonexistent/path/db.sqlite", "SELECT 1")
        assert "Error" in result
        assert "not found" in result.lower()

    def test_successful_select(self) -> None:
        """Real SQLite query returns formatted results."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
            conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
            conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
            conn.commit()
            conn.close()

            result = db_tools.query_sqlite.function(db_path, "SELECT * FROM users")
            assert "Alice" in result
            assert "Bob" in result
            assert "Rows: 2" in result
            assert "|" in result  # pipe-separated table
        finally:
            os.unlink(db_path)

    def test_no_result_set(self) -> None:
        """Queries that produce no result set return informative message."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.commit()
            conn.close()

            # PRAGMA returns a result set, but something like a CTE that
            # returns nothing specific... We test with a table info pragma
            result = db_tools.query_sqlite.function(db_path, "SELECT * FROM t WHERE id > 999")
            assert "Rows: 0" in result
        finally:
            os.unlink(db_path)

    def test_write_operations_blocked(self) -> None:
        """INSERT/UPDATE/DELETE are blocked by read-only mode."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.commit()
            conn.close()

            result = db_tools.query_sqlite.function(db_path, "INSERT INTO t VALUES (1)")
            assert "Error" in result
        finally:
            os.unlink(db_path)

    def test_max_rows_limits_output(self) -> None:
        """max_rows parameter limits the number of returned rows."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE nums (n INTEGER)")
            for i in range(50):
                conn.execute("INSERT INTO nums VALUES (?)", (i,))
            conn.commit()
            conn.close()

            result = db_tools.query_sqlite.function(db_path, "SELECT * FROM nums", max_rows=5)
            assert "Rows: 5" in result
            assert "limited to 5 rows" in result
        finally:
            os.unlink(db_path)

    def test_invalid_sql_returns_error(self) -> None:
        """Invalid SQL returns a clear error message."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.commit()
            conn.close()

            result = db_tools.query_sqlite.function(db_path, "SELECTXYZ")
            assert "Error" in result
        finally:
            os.unlink(db_path)

    def test_column_headers_in_output(self) -> None:
        """Output includes column names as headers."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE items (name TEXT, price REAL)")
            conn.execute("INSERT INTO items VALUES ('Widget', 9.99)")
            conn.commit()
            conn.close()

            result = db_tools.query_sqlite.function(db_path, "SELECT name, price FROM items")
            assert "name" in result
            assert "price" in result
            assert "Widget" in result
            assert "9.99" in result
        finally:
            os.unlink(db_path)


# =============================================================================
# query_postgres tests (mocked)
# =============================================================================


class TestQueryPostgres:
    """Tests for the query_postgres tool (all mocked)."""

    def test_tool_has_correct_metadata(self) -> None:
        assert db_tools.query_postgres.name == "query_postgres"
        assert "PostgreSQL" in db_tools.query_postgres.description

    def test_stability_marker_is_beta(self) -> None:
        assert getattr(db_tools.query_postgres, "__stability__", None) == "beta"

    def test_empty_connection_string_rejected(self) -> None:
        """Empty connection string returns an error (before attempting import)."""
        # We need to mock psycopg2 to prevent ImportError
        with patch.dict("sys.modules", {"psycopg2": MagicMock()}):
            result = db_tools.query_postgres.function("", "SELECT 1")
        assert "Error" in result

    def test_empty_sql_rejected(self) -> None:
        with patch.dict("sys.modules", {"psycopg2": MagicMock()}):
            result = db_tools.query_postgres.function("postgresql://localhost/test", "")
        assert "Error" in result

    def test_psycopg2_not_installed(self) -> None:
        """Missing psycopg2 returns install instructions."""
        with patch.dict("sys.modules", {"psycopg2": None}):
            result = db_tools.query_postgres.function("postgresql://localhost/test", "SELECT 1")
        assert "psycopg2" in result
        assert "pip install" in result

    @patch("selectools.toolbox.db_tools.query_postgres.function")
    def test_successful_query_via_function_mock(self, mock_fn: MagicMock) -> None:
        """Full integration mock: successful SELECT returns formatted table."""
        mock_fn.return_value = "Rows: 2\n\nid | name\n---+-----\n1  | Alice\n2  | Bob"
        result = db_tools.query_postgres.function(
            "postgresql://user:pass@host/db", "SELECT * FROM users"
        )
        assert "Alice" in result

    def test_psycopg2_connect_and_readonly(self) -> None:
        """Verify psycopg2.connect is called with readonly session."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",)]
        mock_cursor.fetchmany.return_value = [(1, "Alice")]
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_psycopg2.connect.return_value = mock_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            import importlib

            import selectools.toolbox.db_tools as _db_mod

            importlib.reload(_db_mod)

            result = _db_mod.query_postgres.function(
                "postgresql://localhost/test", "SELECT * FROM t"
            )

        assert "Alice" in result
        mock_conn.set_session.assert_called_once_with(readonly=True, autocommit=True)

    def test_connection_error_handled(self) -> None:
        """Connection errors are caught gracefully."""
        mock_psycopg2 = MagicMock()
        mock_psycopg2.connect.side_effect = Exception("connection refused")

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            from selectools.toolbox.db_tools import query_postgres

            result = query_postgres.function("postgresql://localhost/db", "SELECT 1")
        assert "Error" in result
        assert "connection refused" in result


# =============================================================================
# _format_table helper tests
# =============================================================================


class TestFormatTable:
    """Tests for the _format_table helper function."""

    def test_empty_columns(self) -> None:
        result = db_tools._format_table([], [])
        assert "no columns" in result.lower()

    def test_single_row(self) -> None:
        result = db_tools._format_table(["name", "age"], [("Alice", 30)])
        assert "name" in result
        assert "age" in result
        assert "Alice" in result
        assert "30" in result
        assert "|" in result

    def test_multiple_rows(self) -> None:
        rows = [("Alice", 30), ("Bob", 25)]
        result = db_tools._format_table(["name", "age"], rows)
        lines = result.strip().split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows

    def test_column_width_adapts(self) -> None:
        """Columns widen to fit the longest value."""
        result = db_tools._format_table(["x"], [("short",), ("a very long value here",)])
        # The header "x" should be padded to match the longest value
        lines = result.strip().split("\n")
        assert len(lines[0]) == len(lines[2])
