"""
Edge-case tests for JsonFileSessionStore and SQLiteSessionStore.

Covers error paths not hit by the main test_sessions.py:
- Corrupt JSON files
- OS errors during list/exists
- TTL edge cases
- Tool-pair boundary trimming in memory round-trips
"""

from __future__ import annotations

import json
import os
import time

import pytest

from selectools.memory import ConversationMemory
from selectools.sessions import JsonFileSessionStore, SQLiteSessionStore
from selectools.types import Message, Role, ToolCall


def _memory_with_messages(*contents: str) -> ConversationMemory:
    mem = ConversationMemory(max_messages=50)
    for c in contents:
        mem.add(Message(role=Role.USER, content=c))
    return mem


# ======================================================================
# JsonFileSessionStore edge cases
# ======================================================================


class TestJsonFileCorruptJSON:
    def test_save_over_corrupt_existing_file(self, tmp_path: "os.PathLike[str]") -> None:
        """save() should handle corrupt existing JSON gracefully."""
        store = JsonFileSessionStore(directory=str(tmp_path))
        # Write corrupt JSON to the session file
        path = os.path.join(str(tmp_path), "s1.json")
        with open(path, "w") as f:
            f.write("not valid json{{{")

        # Save should succeed despite corrupt existing file
        store.save("s1", _memory_with_messages("Hello"))
        loaded = store.load("s1")
        assert loaded is not None
        assert len(loaded) == 1

    def test_list_skips_corrupt_json_files(self, tmp_path: "os.PathLike[str]") -> None:
        """list() should skip files with corrupt JSON."""
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("good", _memory_with_messages("ok"))

        # Write a corrupt JSON file
        corrupt_path = os.path.join(str(tmp_path), "bad.json")
        with open(corrupt_path, "w") as f:
            f.write("{invalid json")

        sessions = store.list()
        assert len(sessions) == 1
        assert sessions[0].session_id == "good"

    def test_exists_returns_false_for_corrupt_json(self, tmp_path: "os.PathLike[str]") -> None:
        """exists() should return False if the file has corrupt JSON."""
        store = JsonFileSessionStore(directory=str(tmp_path))
        path = os.path.join(str(tmp_path), "bad.json")
        with open(path, "w") as f:
            f.write("{{not json}}")

        assert store.exists("bad") is False


class TestJsonFileTTLEdgeCases:
    def test_list_removes_expired_files(self, tmp_path: "os.PathLike[str]") -> None:
        """list() should delete expired session files."""
        store = JsonFileSessionStore(directory=str(tmp_path), default_ttl=1)
        store.save("expired", _memory_with_messages("old"))

        path = os.path.join(str(tmp_path), "expired.json")
        with open(path, "r") as f:
            data = json.load(f)
        data["updated_at"] = time.time() - 100
        with open(path, "w") as f:
            json.dump(data, f)

        # list() should clean up expired files
        sessions = store.list()
        assert len(sessions) == 0
        # File should be deleted
        assert not os.path.exists(path)

    def test_list_handles_os_error_on_expired_delete(self, tmp_path: "os.PathLike[str]") -> None:
        """list() should not crash if it can't delete an expired file."""
        from unittest.mock import patch

        store = JsonFileSessionStore(directory=str(tmp_path), default_ttl=1)
        store.save("expired", _memory_with_messages("old"))

        path = os.path.join(str(tmp_path), "expired.json")
        with open(path, "r") as f:
            data = json.load(f)
        data["updated_at"] = time.time() - 100
        with open(path, "w") as f:
            json.dump(data, f)

        # Mock os.remove to raise OSError
        original_remove = os.remove

        def failing_remove(p):
            if p == path:
                raise OSError("Permission denied")
            return original_remove(p)

        with patch("os.remove", side_effect=failing_remove):
            # Should not crash
            sessions = store.list()
            assert len(sessions) == 0

    def test_list_skips_non_json_files(self, tmp_path: "os.PathLike[str]") -> None:
        """list() should skip non-.json files in the directory."""
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("real", _memory_with_messages("ok"))

        # Create a non-JSON file
        other_path = os.path.join(str(tmp_path), "readme.txt")
        with open(other_path, "w") as f:
            f.write("not a session")

        sessions = store.list()
        assert len(sessions) == 1
        assert sessions[0].session_id == "real"


class TestJsonFileSessionMetadata:
    def test_metadata_message_count(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        store.save("s1", _memory_with_messages("A", "B", "C"))
        sessions = store.list()
        assert sessions[0].message_count == 3

    def test_metadata_timestamps(self, tmp_path: "os.PathLike[str]") -> None:
        store = JsonFileSessionStore(directory=str(tmp_path))
        before = time.time()
        store.save("s1", _memory_with_messages("A"))
        after = time.time()

        sessions = store.list()
        meta = sessions[0]
        assert before <= meta.created_at <= after
        assert before <= meta.updated_at <= after


# ======================================================================
# SQLiteSessionStore edge cases
# ======================================================================


class TestSQLiteEdgeCases:
    def test_preserves_created_at_on_overwrite(self, tmp_path: "os.PathLike[str]") -> None:
        import sqlite3

        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        store.save("s1", _memory_with_messages("v1"))

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT created_at FROM sessions WHERE session_id = ?", ("s1",)
        ).fetchone()
        first_created = row[0]
        conn.close()

        time.sleep(0.01)
        store.save("s1", _memory_with_messages("v2"))

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT created_at, updated_at FROM sessions WHERE session_id = ?", ("s1",)
        ).fetchone()
        conn.close()

        assert row[0] == first_created
        assert row[1] > first_created

    def test_exists_expired_returns_false(self, tmp_path: "os.PathLike[str]") -> None:
        import sqlite3

        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db, default_ttl=1)
        store.save("s1", _memory_with_messages("Hello"))

        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (time.time() - 100, "s1"),
        )
        conn.commit()
        conn.close()

        assert store.exists("s1") is False

    def test_metadata_message_count(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        store.save("s1", _memory_with_messages("A", "B", "C"))
        sessions = store.list()
        assert sessions[0].message_count == 3

    def test_list_cleans_up_expired(self, tmp_path: "os.PathLike[str]") -> None:
        import sqlite3

        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db, default_ttl=1)
        store.save("fresh", _memory_with_messages("ok"))
        store.save("stale", _memory_with_messages("old"))

        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (time.time() - 100, "stale"),
        )
        conn.commit()
        conn.close()

        sessions = store.list()
        assert len(sessions) == 1
        assert sessions[0].session_id == "fresh"

        # Expired session should be deleted from DB
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        conn.close()
        assert count == 1

    def test_empty_memory_round_trip(self, tmp_path: "os.PathLike[str]") -> None:
        db = os.path.join(str(tmp_path), "test.db")
        store = SQLiteSessionStore(db_path=db)
        mem = ConversationMemory()
        store.save("empty", mem)

        loaded = store.load("empty")
        assert loaded is not None
        assert len(loaded) == 0


# ======================================================================
# Regression tests for fixed bugs
# ======================================================================


class TestJsonFileSessionStoreLoadCorruptJSON:
    def test_load_returns_none_for_corrupt_json(self, tmp_path: "os.PathLike[str]") -> None:
        """Regression: load() must return None instead of raising on corrupt JSON."""
        store = JsonFileSessionStore(directory=str(tmp_path))
        path = os.path.join(str(tmp_path), "corrupt.json")
        with open(path, "w") as f:
            f.write("{not valid json{{{")

        # Should return None, not raise JSONDecodeError
        result = store.load("corrupt")
        assert result is None

    def test_load_returns_none_for_truncated_json(self, tmp_path: "os.PathLike[str]") -> None:
        """Regression: load() must return None for truncated/incomplete JSON."""
        store = JsonFileSessionStore(directory=str(tmp_path))
        path = os.path.join(str(tmp_path), "truncated.json")
        with open(path, "w") as f:
            f.write('{"session_id": "truncated", "memory": {')  # truncated

        result = store.load("truncated")
        assert result is None
