#!/usr/bin/env python3
"""
Database Query Tools -- SQL queries from agents (read-only).

No API key needed. Creates a sample SQLite database and queries it.
Also supports PostgreSQL with psycopg2.

Run: python examples/85_database_query.py
"""

import os
import sqlite3
import tempfile

from selectools.toolbox.db_tools import query_sqlite

print("=== Database Query Tools Example ===\n")

# Create a sample database
db_path = os.path.join(tempfile.mkdtemp(), "sample.db")
conn = sqlite3.connect(db_path)
conn.execute(
    """CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary REAL
)"""
)
conn.executemany(
    "INSERT INTO employees VALUES (?, ?, ?, ?)",
    [
        (1, "Alice", "Engineering", 120000),
        (2, "Bob", "Marketing", 95000),
        (3, "Charlie", "Engineering", 115000),
        (4, "Diana", "Sales", 105000),
        (5, "Eve", "Engineering", 130000),
    ],
)
conn.commit()
conn.close()

# 1. Basic query
print("--- All employees ---")
result = query_sqlite.function(db_path, "SELECT * FROM employees")
print(result)

# 2. Filtered query
print("\n--- Engineering team ---")
result = query_sqlite.function(
    db_path, "SELECT name, salary FROM employees WHERE department = 'Engineering'"
)
print(result)

# 3. Aggregation
print("\n--- Department stats ---")
result = query_sqlite.function(
    db_path,
    "SELECT department, COUNT(*) as count, AVG(salary) as avg_salary FROM employees GROUP BY department",
)
print(result)

# Cleanup
os.unlink(db_path)

print(
    """
--- PostgreSQL Pattern ---
from selectools.toolbox.db_tools import query_postgres

result = query_postgres.function(
    "postgresql://user:pass@localhost:5432/mydb",
    "SELECT * FROM users LIMIT 10"
)

--- Agent Pattern ---
from selectools import Agent
from selectools.providers import OpenAIProvider
from selectools.toolbox.db_tools import query_sqlite

agent = Agent(tools=[query_sqlite], provider=OpenAIProvider())
result = agent.run("What's the average salary by department?")
# Agent generates SQL and executes it (read-only mode)
"""
)

print("Done!")
