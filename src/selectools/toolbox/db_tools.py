"""
Database query tools for SQLite and PostgreSQL.

SQLite support uses the standard-library ``sqlite3`` module. PostgreSQL
support requires the ``psycopg2`` package (lazy-imported at call time).
"""

from __future__ import annotations

import os
import re
import sqlite3

from ..tools import tool


def _validate_sql_readonly(sql: str) -> str | None:
    """Validate that SQL is a single read-only SELECT statement.

    Returns an error message if invalid, or None if the query is acceptable.
    """
    # Strip trailing whitespace and trailing semicolons
    stripped = sql.strip().rstrip(";").strip()

    if not stripped:
        return "Error: No SQL query provided."

    # Reject multi-statement queries (semicolons in the body)
    if ";" in stripped:
        return "Error: Multi-statement queries are not allowed. Submit one SELECT at a time."

    # Only allow SELECT statements
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        return "Error: Only SELECT queries are allowed."

    return None


def _format_table(columns: list[str], rows: list[tuple]) -> str:
    """Format query results as a pipe-separated text table."""
    if not columns:
        return "(no columns)"

    # Calculate column widths
    widths = [len(str(c)) for c in columns]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(val)))

    # Build header
    header = " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns))
    separator = "-+-".join("-" * w for w in widths)

    # Build rows
    lines = [header, separator]
    for row in rows:
        line = " | ".join(
            str(val).ljust(widths[i]) if i < len(widths) else str(val) for i, val in enumerate(row)
        )
        lines.append(line)

    return "\n".join(lines)


@tool(description="Execute a read-only SQL query against a SQLite database")
def query_sqlite(db_path: str, sql: str, max_rows: int = 100) -> str:
    """
    Execute a read-only SQL query against a SQLite database.

    The connection is opened with ``PRAGMA query_only = ON`` to prevent
    any writes. Results are formatted as a pipe-separated table.

    Args:
        db_path: Path to the SQLite database file.
        sql: SQL query to execute (must be read-only).
        max_rows: Maximum number of rows to return (default: 100, max: 10000).

    Returns:
        Formatted query results or an error message.
    """
    if not db_path or not db_path.strip():
        return "Error: No database path provided."

    if not sql or not sql.strip():
        return "Error: No SQL query provided."

    sql_error = _validate_sql_readonly(sql)
    if sql_error:
        return sql_error

    if not os.path.isfile(db_path):
        return f"Error: Database file not found: {db_path}"

    max_rows = max(1, min(max_rows, 10000))

    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")

        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows)

        if not columns:
            return "Query executed successfully (no result set)."

        total_hint = ""
        extra = cursor.fetchone()
        if extra is not None:
            total_hint = f"\n... (results limited to {max_rows} rows)"

        table = _format_table(columns, rows)
        return f"Rows: {len(rows)}{total_hint}\n\n{table}"

    except sqlite3.OperationalError as e:
        msg = str(e)
        if "readonly" in msg.lower() or "not authorized" in msg.lower():
            return f"Error: Write operations are not allowed (read-only mode): {e}"
        return f"Error: SQLite query failed: {e}"
    except sqlite3.Error as e:
        return f"Error: SQLite error: {e}"
    except Exception as e:
        return f"Error executing query: {e}"
    finally:
        if conn:
            conn.close()


@tool(description="Execute a read-only SQL query against PostgreSQL")
def query_postgres(connection_string: str, sql: str, max_rows: int = 100) -> str:
    """
    Execute a read-only SQL query against a PostgreSQL database.

    Requires the ``psycopg2`` library. The connection is opened with
    ``SET default_transaction_read_only = on`` to prevent writes.

    Args:
        connection_string: PostgreSQL connection string
            (e.g. ``"postgresql://user:pass@host:5432/dbname"``).
        sql: SQL query to execute (must be read-only).
        max_rows: Maximum number of rows to return (default: 100, max: 10000).

    Returns:
        Formatted query results or an error message.
    """
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError:
        return "Error: 'psycopg2' library not installed. Run: pip install psycopg2-binary"

    if not connection_string or not connection_string.strip():
        return "Error: No connection string provided."

    if not sql or not sql.strip():
        return "Error: No SQL query provided."

    sql_error = _validate_sql_readonly(sql)
    if sql_error:
        return sql_error

    max_rows = max(1, min(max_rows, 10000))

    conn = None
    try:
        conn = psycopg2.connect(connection_string)
        conn.set_session(autocommit=False)

        cursor = conn.cursor()
        # Use per-transaction read-only (cannot be overridden within the transaction,
        # unlike session-level SET default_transaction_read_only)
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute(sql)

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows)

        if not columns:
            return "Query executed successfully (no result set)."

        total_hint = ""
        extra = cursor.fetchone()
        if extra is not None:
            total_hint = f"\n... (results limited to {max_rows} rows)"

        table = _format_table(columns, rows)
        return f"Rows: {len(rows)}{total_hint}\n\n{table}"

    except Exception as e:
        error_msg = str(e).strip()
        if "read-only" in error_msg.lower() or "readonly" in error_msg.lower():
            return f"Error: Write operations are not allowed (read-only mode): {e}"
        return f"Error: PostgreSQL query failed: {e}"
    finally:
        if conn:
            conn.close()
