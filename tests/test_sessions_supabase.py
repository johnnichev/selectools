"""
Tests for SupabaseSessionStore backend.

Uses a FakeSupabaseClient to test all SessionStore protocol methods
without requiring a real Supabase connection or the ``supabase`` package
to be installed.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from selectools.memory import ConversationMemory
from selectools.sessions import SessionMetadata, SupabaseSessionStore
from selectools.types import Message, Role, ToolCall

# ======================================================================
# Fake Supabase client
# ======================================================================


class FakeResponse:
    """Mimics the supabase-py response object (has a .data attribute)."""

    def __init__(self, data: Any) -> None:
        self.data = data


class FakeQueryBuilder:
    """Chains .select/.eq/.limit/.upsert/.delete and returns a FakeResponse.

    Stores the in-memory table and applies filters / mutations lazily when
    .execute() is called.
    """

    def __init__(self, store: Dict[str, Dict[str, Any]], table: str, op: str) -> None:
        self._store = store
        self._table = table
        self._op = op  # "select" | "upsert" | "delete"
        self._upsert_payload: Optional[Dict[str, Any]] = None
        self._upsert_kwargs: Dict[str, Any] = {}
        self._filters: List[tuple] = []
        self._select_cols: Optional[str] = None
        self._limit_n: Optional[int] = None

    # builder methods

    def select(self, cols: str = "*") -> "FakeQueryBuilder":
        self._select_cols = cols
        return self

    def eq(self, col: str, val: Any) -> "FakeQueryBuilder":
        self._filters.append((col, val))
        return self

    def limit(self, n: int) -> "FakeQueryBuilder":
        self._limit_n = n
        return self

    def upsert(self, payload: Dict[str, Any], **kwargs: Any) -> "FakeQueryBuilder":
        self._op = "upsert"
        self._upsert_payload = payload
        self._upsert_kwargs = kwargs
        return self

    def delete(self) -> "FakeQueryBuilder":
        self._op = "delete"
        return self

    # terminal

    def execute(self) -> FakeResponse:
        rows = list(self._store.get(self._table, {}).values())

        if self._op == "upsert":
            payload = self._upsert_payload or {}
            key = payload.get("session_id")
            if key is None:
                return FakeResponse(data=None)
            table_rows = self._store.setdefault(self._table, {})
            if key in table_rows:
                # preserve created_at if already set
                existing = table_rows[key]
                merged = {**existing, **payload}
                if "created_at" not in payload and "created_at" in existing:
                    merged["created_at"] = existing["created_at"]
                table_rows[key] = merged
            else:
                if "created_at" not in payload:
                    payload = {**payload, "created_at": datetime.now(timezone.utc).isoformat()}
                table_rows[key] = payload
            return FakeResponse(data=[table_rows[key]])

        # Apply eq filters
        for col, val in self._filters:
            rows = [r for r in rows if r.get(col) == val]

        if self._op == "delete":
            deleted = rows[:]
            table_rows = self._store.get(self._table, {})
            for row in deleted:
                key = row.get("session_id")
                if key and key in table_rows:
                    del table_rows[key]
            return FakeResponse(data=deleted)

        # select
        if self._limit_n is not None:
            rows = rows[: self._limit_n]

        if self._select_cols and self._select_cols != "*":
            cols = [c.strip() for c in self._select_cols.split(",")]
            rows = [{c: r[c] for c in cols if c in r} for r in rows]

        return FakeResponse(data=rows)


class FakeSupabaseClient:
    """Minimal fake for supabase.Client — supports .table(name)."""

    def __init__(self) -> None:
        # Shared in-memory store: {table_name: {session_id: row_dict}}
        self._data: Dict[str, Dict[str, Any]] = {}

    def table(self, name: str) -> "FakeQueryBuilder":
        return FakeQueryBuilder(self._data, name, "select")


# ======================================================================
# Helpers
# ======================================================================


def _make_store(
    client: Optional[FakeSupabaseClient] = None,
    table_name: str = "selectools_sessions",
) -> SupabaseSessionStore:
    """Create a SupabaseSessionStore with a patched supabase import."""
    if client is None:
        client = FakeSupabaseClient()
    fake_module = MagicMock()
    with patch.dict("sys.modules", {"supabase": fake_module}):
        store = SupabaseSessionStore(client=client, table_name=table_name)
    return store


def _memory_with_messages(*contents: str) -> ConversationMemory:
    mem = ConversationMemory(max_messages=50)
    for c in contents:
        mem.add(Message(role=Role.USER, content=c))
    return mem


# ======================================================================
# Save / Load
# ======================================================================


class TestSupabaseSessionStoreSaveLoad:
    def test_save_and_load_round_trip(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        mem = _memory_with_messages("Hello", "World")
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded.get_history()[0].content == "Hello"
        assert loaded.get_history()[1].content == "World"

    def test_load_nonexistent_returns_none(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        assert store.load("nonexistent") is None

    def test_save_overwrites_existing(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("s1", _memory_with_messages("v1"))
        store.save("s1", _memory_with_messages("v2"))

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.get_history()[0].content == "v2"

    def test_save_upsert_called_with_correct_payload(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        mem = _memory_with_messages("Hi")
        store.save("s1", mem)

        table_data = client._data.get("selectools_sessions", {})
        assert "s1" in table_data
        row = table_data["s1"]
        assert row["session_id"] == "s1"
        assert row["message_count"] == 1
        assert "memory_json" in row
        assert "updated_at" in row

    def test_preserves_tool_calls(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        mem = ConversationMemory()
        mem.add(Message(role=Role.USER, content="Search"))
        tc = ToolCall(tool_name="search", parameters={"q": "ai"}, id="tc1")
        mem.add(Message(role=Role.ASSISTANT, content="", tool_calls=[tc]))
        mem.add(Message(role=Role.TOOL, content="result", tool_name="search", tool_call_id="tc1"))
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        history = loaded.get_history()
        assert history[1].tool_calls[0].tool_name == "search"
        assert history[2].tool_name == "search"

    def test_preserves_summary(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        mem = _memory_with_messages("Hello")
        mem.summary = "User said hello"
        store.save("s1", mem)

        loaded = store.load("s1")
        assert loaded is not None
        assert loaded.summary == "User said hello"

    def test_custom_table_name(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client, table_name="my_sessions")
        store.save("s1", _memory_with_messages("Hello"))

        assert "my_sessions" in client._data
        assert "s1" in client._data["my_sessions"]


# ======================================================================
# Delete / List / Exists
# ======================================================================


class TestSupabaseSessionStoreDeleteListExists:
    def test_delete_existing_returns_true(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("s1", _memory_with_messages("Hello"))
        assert store.delete("s1") is True
        assert store.load("s1") is None

    def test_delete_nonexistent_returns_false(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        assert store.delete("nope") is False

    def test_exists_true(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("s1", _memory_with_messages("Hello"))
        assert store.exists("s1") is True

    def test_exists_false(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        assert store.exists("s1") is False

    def test_list_empty(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        assert store.list() == []

    def test_list_sessions(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("s1", _memory_with_messages("A"))
        store.save("s2", _memory_with_messages("B", "C"))

        sessions = store.list()
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert ids == {"s1", "s2"}
        for s in sessions:
            assert isinstance(s, SessionMetadata)

    def test_list_message_counts(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("s1", _memory_with_messages("A"))
        store.save("s2", _memory_with_messages("B", "C"))

        sessions = store.list()
        by_id = {s.session_id: s for s in sessions}
        assert by_id["s1"].message_count == 1
        assert by_id["s2"].message_count == 2

    def test_list_iso_timestamps_parsed(self) -> None:
        """ISO timestamp strings from the DB are converted to Unix floats."""
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("s1", _memory_with_messages("Hello"))

        sessions = store.list()
        assert len(sessions) == 1
        s = sessions[0]
        # timestamps should be positive Unix floats (year 2020+)
        assert s.created_at > 1_580_000_000.0
        assert s.updated_at > 1_580_000_000.0

    def test_list_handles_null_timestamps(self) -> None:
        """Rows with None timestamps get 0.0 rather than raising."""
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("s1", _memory_with_messages("Hello"))
        # Overwrite timestamps with None to simulate a DB row without them
        client._data["selectools_sessions"]["s1"]["created_at"] = None
        client._data["selectools_sessions"]["s1"]["updated_at"] = None

        sessions = store.list()
        assert len(sessions) == 1
        assert sessions[0].created_at == 0.0
        assert sessions[0].updated_at == 0.0

    def test_list_with_numeric_timestamps(self) -> None:
        """Numeric timestamps (float) pass through unchanged."""
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("s1", _memory_with_messages("Hello"))
        client._data["selectools_sessions"]["s1"]["created_at"] = 1_700_000_000.0
        client._data["selectools_sessions"]["s1"]["updated_at"] = 1_700_000_001.5

        sessions = store.list()
        assert sessions[0].created_at == 1_700_000_000.0
        assert sessions[0].updated_at == 1_700_000_001.5


# ======================================================================
# Branch
# ======================================================================


class TestSupabaseSessionStoreBranch:
    def test_branch_copies_session(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        mem = _memory_with_messages("Hello", "World")
        store.save("orig", mem)
        store.branch("orig", "copy")

        orig = store.load("orig")
        copy = store.load("copy")
        assert orig is not None
        assert copy is not None
        assert len(copy) == 2
        assert copy.get_history()[0].content == "Hello"

    def test_branch_copies_are_independent(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("orig", _memory_with_messages("A"))
        store.branch("orig", "fork")

        store.save("fork", _memory_with_messages("B", "C"))
        orig = store.load("orig")
        fork = store.load("fork")
        assert orig is not None
        assert fork is not None
        assert len(orig) == 1
        assert len(fork) == 2

    def test_branch_missing_source_raises(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        with pytest.raises(ValueError, match="not found"):
            store.branch("ghost", "new")


# ======================================================================
# Namespace prefix
# ======================================================================


class TestSupabaseSessionStoreNamespace:
    def test_namespace_prefix_applied_to_storage_key(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("user1", _memory_with_messages("Hi"), namespace="agent-a")

        table_data = client._data.get("selectools_sessions", {})
        # Row key should be "agent-a:user1", not "user1"
        assert "agent-a:user1" in table_data
        assert "user1" not in table_data

    def test_namespace_isolates_same_session_id(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("sid", _memory_with_messages("ns-a"), namespace="a")
        store.save("sid", _memory_with_messages("ns-b"), namespace="b")

        a = store.load("sid", namespace="a")
        b = store.load("sid", namespace="b")
        assert a is not None
        assert b is not None
        assert a.get_history()[0].content == "ns-a"
        assert b.get_history()[0].content == "ns-b"

    def test_no_namespace_and_namespace_do_not_collide(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("sid", _memory_with_messages("bare"))
        store.save("sid", _memory_with_messages("ns"), namespace="myns")

        bare = store.load("sid")
        ns = store.load("sid", namespace="myns")
        assert bare is not None
        assert ns is not None
        assert bare.get_history()[0].content == "bare"
        assert ns.get_history()[0].content == "ns"

    def test_delete_with_namespace(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("sid", _memory_with_messages("Hi"), namespace="ns")
        assert store.exists("sid", namespace="ns") is True
        store.delete("sid", namespace="ns")
        assert store.exists("sid", namespace="ns") is False

    def test_exists_with_namespace(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        store.save("sid", _memory_with_messages("Hi"), namespace="ns")
        assert store.exists("sid", namespace="ns") is True
        assert store.exists("sid") is False  # no namespace -> different key


# ======================================================================
# Validation
# ======================================================================


class TestSupabaseSessionStoreValidation:
    def _store_without_supabase(self) -> SupabaseSessionStore:
        """Return an uninitialised store instance (bypasses lazy import)."""
        client = FakeSupabaseClient()
        return _make_store(client)

    def test_empty_session_id_raises(self) -> None:
        store = self._store_without_supabase()
        with pytest.raises(ValueError, match="must not be empty"):
            store._key("")

    def test_null_byte_in_session_id_raises(self) -> None:
        store = self._store_without_supabase()
        with pytest.raises(ValueError, match="null bytes"):
            store._key("bad\x00id")

    def test_session_id_too_long_raises(self) -> None:
        store = self._store_without_supabase()
        with pytest.raises(ValueError, match="too long"):
            store._key("x" * 513)

    def test_session_id_at_limit_passes(self) -> None:
        store = self._store_without_supabase()
        key = store._key("x" * 512)
        assert len(key) == 512

    def test_empty_namespace_raises(self) -> None:
        store = self._store_without_supabase()
        with pytest.raises(ValueError, match="must not be empty"):
            store._validate_namespace("")

    def test_null_byte_in_namespace_raises(self) -> None:
        store = self._store_without_supabase()
        with pytest.raises(ValueError, match="null bytes"):
            store._validate_namespace("bad\x00ns")

    def test_namespace_too_long_raises(self) -> None:
        store = self._store_without_supabase()
        with pytest.raises(ValueError, match="too long"):
            store._validate_namespace("n" * 513)

    def test_none_namespace_passes(self) -> None:
        store = self._store_without_supabase()
        store._validate_namespace(None)  # must not raise

    def test_save_empty_session_id_raises(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        with pytest.raises(ValueError, match="must not be empty"):
            store.save("", _memory_with_messages("Hi"))

    def test_load_null_byte_session_id_raises(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        with pytest.raises(ValueError, match="null bytes"):
            store.load("bad\x00id")

    def test_delete_too_long_session_id_raises(self) -> None:
        client = FakeSupabaseClient()
        store = _make_store(client)
        with pytest.raises(ValueError, match="too long"):
            store.delete("x" * 513)


# ======================================================================
# Import error
# ======================================================================


class TestSupabaseSessionStoreImportError:
    def test_missing_supabase_raises_import_error(self) -> None:
        client = FakeSupabaseClient()
        with patch.dict("sys.modules", {"supabase": None}):
            with pytest.raises(ImportError, match="supabase"):
                SupabaseSessionStore(client=client)
