"""Tests for knowledge store error handling (Redis + Supabase).

Verifies that both RedisKnowledgeStore and SupabaseKnowledgeStore
degrade gracefully on connection/API errors rather than crashing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from selectools.knowledge import KnowledgeEntry
from selectools.knowledge_store_redis import RedisKnowledgeStore
from selectools.knowledge_store_supabase import SupabaseKnowledgeStore

# ======================================================================
# RedisKnowledgeStore error handling
# ======================================================================


class TestRedisKnowledgeStoreErrorHandling:
    """RedisKnowledgeStore must not crash on connection errors."""

    def _make_store(self) -> RedisKnowledgeStore:
        store = RedisKnowledgeStore.__new__(RedisKnowledgeStore)
        store._client = MagicMock()
        store._prefix = "test"
        return store

    def test_save_handles_connection_error(self) -> None:
        store = self._make_store()
        store._client.hget.side_effect = Exception("Connection refused")
        entry = KnowledgeEntry(content="hello")
        # Should not raise -- returns entry.id gracefully
        result = store.save(entry)
        assert result is not None
        assert result == entry.id

    def test_get_handles_connection_error(self) -> None:
        store = self._make_store()
        store._client.hgetall.side_effect = Exception("Connection refused")
        result = store.get("nonexistent")
        assert result is None

    def test_query_handles_connection_error_on_category(self) -> None:
        store = self._make_store()
        store._client.smembers.side_effect = Exception("Timeout")
        result = store.query(category="facts")
        assert result == []

    def test_query_handles_connection_error_on_sorted_set(self) -> None:
        store = self._make_store()
        store._client.zrevrangebyscore.side_effect = Exception("Timeout")
        result = store.query()
        assert result == []

    def test_count_handles_connection_error(self) -> None:
        store = self._make_store()
        store._client.scard.side_effect = Exception("Timeout")
        result = store.count()
        assert result == 0

    def test_delete_handles_connection_error(self) -> None:
        store = self._make_store()
        store._client.hgetall.side_effect = Exception("Connection refused")
        result = store.delete("some-id")
        assert result is False

    def test_prune_handles_connection_error(self) -> None:
        store = self._make_store()
        store._client.smembers.side_effect = Exception("Timeout")
        result = store.prune()
        assert result == 0


# ======================================================================
# SupabaseKnowledgeStore error handling
# ======================================================================


class TestSupabaseKnowledgeStoreErrorHandling:
    """SupabaseKnowledgeStore must not crash on API errors."""

    def _make_store(self) -> SupabaseKnowledgeStore:
        store = SupabaseKnowledgeStore.__new__(SupabaseKnowledgeStore)
        store._client = MagicMock()
        store._table = "knowledge"
        return store

    def test_save_handles_api_error(self) -> None:
        store = self._make_store()
        store._client.table.return_value.upsert.return_value.execute.side_effect = Exception(
            "API Error"
        )
        entry = KnowledgeEntry(content="hello")
        result = store.save(entry)
        assert result is not None
        assert result == entry.id

    def test_get_handles_api_error(self) -> None:
        store = self._make_store()
        store._client.table.return_value.select.return_value.eq.return_value.execute.side_effect = (
            Exception("API Error")
        )
        result = store.get("test")
        assert result is None

    def test_query_handles_api_error(self) -> None:
        store = self._make_store()
        store._client.table.return_value.select.return_value.gte.side_effect = Exception(
            "API Error"
        )
        result = store.query()
        assert result == []

    def test_count_handles_api_error(self) -> None:
        store = self._make_store()
        store._client.table.return_value.select.return_value.execute.side_effect = Exception(
            "API Error"
        )
        result = store.count()
        assert result == 0

    def test_delete_handles_api_error(self) -> None:
        store = self._make_store()
        store._client.table.return_value.delete.return_value.eq.return_value.execute.side_effect = (
            Exception("API Error")
        )
        result = store.delete("test-id")
        assert result is False

    def test_prune_handles_api_error(self) -> None:
        store = self._make_store()
        store._client.table.return_value.select.return_value.eq.return_value.execute.side_effect = (
            Exception("API Error")
        )
        result = store.prune()
        assert result == 0
