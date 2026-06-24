"""
Tests for SupabaseKnowledgeBackend.

Uses a FakeSupabaseClient (mirroring tests/test_sessions_supabase.py) so no
network access or ``supabase`` package install is required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from selectools.knowledge import KnowledgeMemory
from selectools.knowledge_backends import SupabaseKnowledgeBackend

# ======================================================================
# Fake Supabase client
# ======================================================================


class FakeResponse:
    """Mimics the supabase-py response object (has a .data attribute)."""

    def __init__(self, data: Any) -> None:
        self.data = data


class FakeNotNullViolation(Exception):
    """Stand-in for postgrest's APIError code 23502 (not_null_violation).

    Mirrors Sheriff's FakeSupabase (sheriff#302): a fake table configured
    with NOT-NULL-no-default columns rejects partial-payload upserts exactly
    like PostgREST, so the partial-column-upsert bug class can't pass tests.
    """


def _raise_not_null(col: str, table: str) -> None:
    raise FakeNotNullViolation(
        f'null value in column "{col}" of relation "{table}" '
        "violates not-null constraint (code 23502)"
    )


class FakeQueryBuilder:
    """Chains .select/.eq/.limit/.upsert/.update and returns a FakeResponse."""

    def __init__(
        self,
        store: Dict[str, Dict[str, Any]],
        table: str,
        not_null_no_default: tuple = (),
    ) -> None:
        self._store = store
        self._table = table
        self._not_null_no_default = not_null_no_default
        self._op = "select"
        self._upsert_payload: Optional[Dict[str, Any]] = None
        self._upsert_kwargs: Dict[str, Any] = {}
        self._update_payload: Optional[Dict[str, Any]] = None
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

    def update(self, payload: Dict[str, Any]) -> "FakeQueryBuilder":
        self._op = "update"
        self._update_payload = payload
        return self

    def execute(self) -> FakeResponse:
        if self._op == "upsert":
            payload = self._upsert_payload or {}
            # INSERT ... ON CONFLICT semantics: Postgres validates the
            # proposed insert tuple against NOT NULL constraints BEFORE
            # conflict arbitration.  A NOT-NULL-no-default column absent
            # from the payload fails the whole statement (23502) even when
            # the conflicting row already exists.
            for col in self._not_null_no_default:
                if payload.get(col) is None:
                    _raise_not_null(col, self._table)
            conflict_col = self._upsert_kwargs.get("on_conflict")
            key = payload.get(conflict_col) if conflict_col else None
            if key is None:
                return FakeResponse(data=None)
            table_rows = self._store.setdefault(self._table, {})
            existing = table_rows.get(key, {})
            table_rows[key] = {**existing, **payload}
            return FakeResponse(data=[table_rows[key]])

        if self._op == "update":
            payload = self._update_payload or {}
            for col in self._not_null_no_default:
                if col in payload and payload[col] is None:
                    _raise_not_null(col, self._table)
            matched = []
            for row in self._store.get(self._table, {}).values():
                if all(row.get(col) == val for col, val in self._filters):
                    row.update(payload)
                    matched.append(dict(row))
            # PostgREST UPDATE matching zero rows: empty data, no error.
            return FakeResponse(data=matched)

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
    """Minimal fake for supabase.Client — supports .table(name).

    ``not_null_no_default`` maps table name -> columns that are NOT NULL
    with no default, enforced on upserts like real Postgres.
    """

    def __init__(self, not_null_no_default: Optional[Dict[str, tuple]] = None) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self._not_null_no_default = not_null_no_default or {}

    def table(self, name: str) -> FakeQueryBuilder:
        return FakeQueryBuilder(self._data, name, self._not_null_no_default.get(name, ()))


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

    def test_invalid_write_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="write_mode"):
            _make_backend(write_mode="merge")

    def test_valid_write_modes_accepted(self) -> None:
        _make_backend(write_mode="upsert")
        _make_backend(write_mode="update")


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
# write_mode (issue #89 — NOT NULL columns break partial-column upserts)
# ======================================================================


def _users_table_client() -> FakeSupabaseClient:
    """A fake client whose ``users`` table has ``email NOT NULL`` (no
    default) — the Sheriff-style shared table from the docstring recipe."""
    return FakeSupabaseClient(not_null_no_default={"users": ("email",)})


def _seed_user_row(client: FakeSupabaseClient, user_id: str = "u1") -> None:
    client._data.setdefault("users", {})[user_id] = {
        "user_id": user_id,
        "email": "john@example.com",
        "memory_text": None,
    }


def _users_backend(client: FakeSupabaseClient, **kwargs: Any) -> SupabaseKnowledgeBackend:
    return _make_backend(
        client,
        key="u1",
        table_name="users",
        key_column="user_id",
        data_column="memory_text",
        **kwargs,
    )


class TestWriteMode:
    def test_upsert_against_not_null_table_raises(self) -> None:
        # Regression pin for issue #89: the partial {key, data, updated_at}
        # upsert fails NOT NULL validation on the proposed insert tuple
        # even though the row already exists.
        client = _users_table_client()
        _seed_user_row(client)
        backend = _users_backend(client)  # default write_mode="upsert"
        with pytest.raises(FakeNotNullViolation, match="23502"):
            backend.save_bytes(b"memory payload")

    def test_update_mode_succeeds_against_not_null_table(self) -> None:
        client = _users_table_client()
        _seed_user_row(client)
        backend = _users_backend(client, write_mode="update")
        backend.save_bytes(b"memory payload")
        row = client._data["users"]["u1"]
        assert row["memory_text"] == "memory payload"
        assert row["email"] == "john@example.com"  # untouched
        assert backend.load_bytes() == b"memory payload"

    def test_update_mode_round_trip_overwrites(self) -> None:
        client = _users_table_client()
        _seed_user_row(client)
        backend = _users_backend(client, write_mode="update")
        backend.save_bytes(b"v1")
        backend.save_bytes(b"v2")
        assert backend.load_bytes() == b"v2"

    def test_update_mode_zero_rows_raises_clear_error(self) -> None:
        client = _users_table_client()  # no row seeded
        backend = _users_backend(client, write_mode="update")
        with pytest.raises(RuntimeError, match="must pre-exist"):
            backend.save_bytes(b"memory payload")

    def test_update_mode_sets_updated_at(self) -> None:
        client = _users_table_client()
        _seed_user_row(client)
        backend = _users_backend(client, write_mode="update")
        backend.save_bytes(b"x")
        assert "updated_at" in client._data["users"]["u1"]

    def test_default_dedicated_table_upsert_unchanged(self) -> None:
        # Dedicated selectools_knowledge table (key + data NOT NULL, both
        # always present in the payload): default upsert keeps working.
        client = FakeSupabaseClient(not_null_no_default={"selectools_knowledge": ("key", "data")})
        backend = _make_backend(client)
        backend.save_bytes(b"v1")
        backend.save_bytes(b"v2")  # idempotent re-save, no pre-existing row needed
        assert backend.load_bytes() == b"v2"


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
