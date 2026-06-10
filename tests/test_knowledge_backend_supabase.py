"""
Tests for SupabaseKnowledgeBackend.

Uses a FakeSupabaseClient (mirroring tests/test_sessions_supabase.py) so no
network access or ``supabase`` package install is required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from selectools.knowledge import KnowledgeBackend, KnowledgeMemory
from selectools.knowledge_backends import SupabaseKnowledgeBackend

# ======================================================================
# Fake Supabase client
# ======================================================================


class FakeResponse:
    """Mimics the supabase-py response object (has a .data attribute)."""

    def __init__(self, data: Any) -> None:
        self.data = data


class FakeQueryBuilder:
    """Chains .select/.eq/.limit/.upsert and returns a FakeResponse."""

    def __init__(self, store: Dict[str, Dict[str, Any]], table: str) -> None:
        self._store = store
        self._table = table
        self._op = "select"
        self._upsert_payload: Optional[Dict[str, Any]] = None
        self._upsert_kwargs: Dict[str, Any] = {}
        self._filters: List[tuple] = []
        self._select_cols: Optional[str] = None
        self._limit_n: Optional[int] = None

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

    def execute(self) -> FakeResponse:
        if self._op == "upsert":
            payload = self._upsert_payload or {}
            conflict_col = self._upsert_kwargs.get("on_conflict")
            key = payload.get(conflict_col) if conflict_col else None
            if key is None:
                return FakeResponse(data=None)
            table_rows = self._store.setdefault(self._table, {})
            existing = table_rows.get(key, {})
            table_rows[key] = {**existing, **payload}
            return FakeResponse(data=[table_rows[key]])

        rows = list(self._store.get(self._table, {}).values())
        for col, val in self._filters:
            rows = [r for r in rows if r.get(col) == val]
        if self._limit_n is not None:
            rows = rows[: self._limit_n]
        if self._select_cols and self._select_cols != "*":
            cols = [c.strip() for c in self._select_cols.split(",")]
            rows = [{c: r[c] for c in cols if c in r} for r in rows]
        return FakeResponse(data=rows)


class FakeSupabaseClient:
    """Minimal fake for supabase.Client — supports .table(name)."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def table(self, name: str) -> FakeQueryBuilder:
        return FakeQueryBuilder(self._data, name)


# ======================================================================
# Helpers
# ======================================================================


def _make_backend(
    client: Optional[FakeSupabaseClient] = None,
    key: str = "user-1",
    **kwargs: Any,
) -> SupabaseKnowledgeBackend:
    if client is None:
        client = FakeSupabaseClient()
    fake_module = MagicMock()
    with patch.dict("sys.modules", {"supabase": fake_module}):
        backend = SupabaseKnowledgeBackend(client=client, key=key, **kwargs)
    return backend


# ======================================================================
# Construction
# ======================================================================


class TestConstruction:
    def test_import_error_when_supabase_missing(self) -> None:
        with patch.dict("sys.modules", {"supabase": None}):
            with pytest.raises(ImportError, match="supabase"):
                SupabaseKnowledgeBackend(client=FakeSupabaseClient(), key="u1")

    def test_satisfies_protocol_structurally(self) -> None:
        backend = _make_backend()
        assert callable(backend.load_bytes)
        assert callable(backend.save_bytes)

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_backend(key="")

    def test_null_byte_key_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_backend(key="bad\x00key")

    def test_too_long_key_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_backend(key="x" * 513)


# ======================================================================
# Save / Load
# ======================================================================


class TestSaveLoad:
    def test_load_missing_returns_none(self) -> None:
        backend = _make_backend()
        assert backend.load_bytes() is None

    def test_round_trip_utf8(self) -> None:
        backend = _make_backend()
        backend.save_bytes(b'{"version": 1, "files": {}}')
        assert backend.load_bytes() == b'{"version": 1, "files": {}}'

    def test_utf8_stored_as_plain_text(self) -> None:
        client = FakeSupabaseClient()
        backend = _make_backend(client)
        backend.save_bytes(b'{"hello": "world"}')
        row = client._data["selectools_knowledge"]["user-1"]
        assert row["data"] == '{"hello": "world"}'

    def test_round_trip_binary_uses_base64_marker(self) -> None:
        client = FakeSupabaseClient()
        backend = _make_backend(client)
        payload = b"\x00\xff\xfe"
        backend.save_bytes(payload)
        row = client._data["selectools_knowledge"]["user-1"]
        assert row["data"].startswith("b64:")
        assert backend.load_bytes() == payload

    def test_text_starting_with_marker_is_escaped(self) -> None:
        backend = _make_backend()
        payload = b"b64:not actually base64"
        backend.save_bytes(payload)
        assert backend.load_bytes() == payload

    def test_save_overwrites(self) -> None:
        backend = _make_backend()
        backend.save_bytes(b"v1")
        backend.save_bytes(b"v2")
        assert backend.load_bytes() == b"v2"

    def test_keys_are_isolated(self) -> None:
        client = FakeSupabaseClient()
        b1 = _make_backend(client, key="user-1")
        b2 = _make_backend(client, key="user-2")
        b1.save_bytes(b"alpha")
        b2.save_bytes(b"beta")
        assert b1.load_bytes() == b"alpha"
        assert b2.load_bytes() == b"beta"

    def test_custom_table_and_columns(self) -> None:
        client = FakeSupabaseClient()
        backend = _make_backend(
            client,
            key="u42",
            table_name="user_memory",
            key_column="user_id",
            data_column="memory_text",
        )
        backend.save_bytes(b"custom layout")
        row = client._data["user_memory"]["u42"]
        assert row["user_id"] == "u42"
        assert row["memory_text"] == "custom layout"
        assert backend.load_bytes() == b"custom layout"

    def test_upsert_sets_updated_at(self) -> None:
        client = FakeSupabaseClient()
        backend = _make_backend(client)
        backend.save_bytes(b"x")
        row = client._data["selectools_knowledge"]["user-1"]
        assert "updated_at" in row


# ======================================================================
# Integration with KnowledgeMemory
# ======================================================================


class TestKnowledgeMemoryIntegration:
    def test_end_to_end_round_trip(self, tmp_path) -> None:
        client = FakeSupabaseClient()
        backend = _make_backend(client, key="user-7")

        km1 = KnowledgeMemory(directory=str(tmp_path / "d1"), backend=backend)
        km1.remember("supabase persisted fact", persistent=True, importance=0.9)

        km2 = KnowledgeMemory(directory=str(tmp_path / "d2"), backend=backend)
        assert "supabase persisted fact" in km2.build_context()
        assert "supabase persisted fact" in km2.get_persistent_facts()
