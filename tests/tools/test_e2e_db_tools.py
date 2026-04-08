"""End-to-end tests for the database tools against real SQLite.

The existing ``test_db_tools.py`` relies on mocked ``psycopg2`` and limited
SQLite coverage. These tests create real on-disk SQLite databases with real
schemas and verify that:

- ``query_sqlite`` reads actual rows from a real file
- The ``PRAGMA query_only = ON`` enforcement rejects writes
- ``max_rows`` genuinely limits the returned result set
- The table formatting matches what the LLM will see

``query_postgres`` lives in test_e2e_pgvector_store.py's tier because it
requires a running Postgres instance with credentials.

Run with:

    pytest tests/tools/test_e2e_db_tools.py --run-e2e -v
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from selectools.toolbox import db_tools

pytestmark = pytest.mark.e2e


@pytest.fixture
def real_sqlite_db(tmp_path: Path) -> Path:
    """Create a real SQLite database on disk with sample data."""
    db_path = tmp_path / "e2e.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)")
    conn.executemany(
        "INSERT INTO users (id, name, age) VALUES (?, ?, ?)",
        [
            (1, "alice", 30),
            (2, "bob", 25),
            (3, "carol", 40),
            (4, "dave", 35),
            (5, "eve", 28),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


class TestQuerySqliteReal:
    def test_select_returns_rows(self, real_sqlite_db: Path) -> None:
        """A real SELECT returns all rows formatted as a text table."""
        result = db_tools.query_sqlite.function(
            str(real_sqlite_db), "SELECT id, name, age FROM users ORDER BY id"
        )
        for name in ("alice", "bob", "carol", "dave", "eve"):
            assert name in result
        # Column headers appear in output
        assert "id" in result
        assert "name" in result

    def test_select_where_clause(self, real_sqlite_db: Path) -> None:
        """WHERE clauses filter rows as expected."""
        result = db_tools.query_sqlite.function(
            str(real_sqlite_db), "SELECT name FROM users WHERE age > 30"
        )
        assert "carol" in result
        assert "dave" in result
        assert "alice" not in result
        assert "bob" not in result

    def test_count_query(self, real_sqlite_db: Path) -> None:
        """Aggregate queries return single-row results."""
        result = db_tools.query_sqlite.function(
            str(real_sqlite_db), "SELECT COUNT(*) AS total FROM users"
        )
        assert "5" in result

    def test_insert_rejected_readonly(self, real_sqlite_db: Path) -> None:
        """INSERT is rejected by the read-only validator."""
        result = db_tools.query_sqlite.function(
            str(real_sqlite_db), "INSERT INTO users (id, name) VALUES (99, 'mallory')"
        )
        assert "error" in result.lower() or "read-only" in result.lower()

        # Verify the row was NOT inserted (sanity-check the enforcement worked)
        conn = sqlite3.connect(str(real_sqlite_db))
        (count,) = conn.execute("SELECT COUNT(*) FROM users WHERE name = 'mallory'").fetchone()
        conn.close()
        assert count == 0

    def test_update_rejected_readonly(self, real_sqlite_db: Path) -> None:
        """UPDATE is rejected by the read-only validator."""
        result = db_tools.query_sqlite.function(
            str(real_sqlite_db), "UPDATE users SET age = 999 WHERE id = 1"
        )
        assert "error" in result.lower() or "read-only" in result.lower()

    def test_max_rows_truncates(self, real_sqlite_db: Path) -> None:
        """max_rows caps the result set."""
        result = db_tools.query_sqlite.function(
            str(real_sqlite_db), "SELECT name FROM users ORDER BY id", max_rows=2
        )
        assert "alice" in result
        assert "bob" in result
        # Rows 3-5 should NOT be present
        assert "carol" not in result
