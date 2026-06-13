"""
Tests for DynamoDBSessionStore backend.

Uses an in-memory fake DynamoDB Table (injected via sys.modules, the same
mechanism the Redis/Mongo backend tests use) so the full SessionStore protocol
is exercised without real AWS access.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from selectools.memory import ConversationMemory
from selectools.sessions import DynamoDBSessionStore, SessionMetadata
from selectools.types import Message, Role

# ======================================================================
# Fake DynamoDB
# ======================================================================


class FakeTable:
    """In-memory stand-in for a boto3 DynamoDB Table (the subset used)."""

    def __init__(self) -> None:
        self.items: Dict[str, Dict[str, Any]] = {}

    def put_item(self, Item: Dict[str, Any]) -> Dict[str, Any]:  # noqa: N803 (boto3 kwarg)
        self.items[Item["session_key"]] = dict(Item)
        return {}

    def get_item(self, Key: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:  # noqa: N803
        item = self.items.get(Key["session_key"])
        return {"Item": dict(item)} if item is not None else {}

    def delete_item(
        self,
        Key: Dict[str, Any],
        ReturnValues: Optional[str] = None,  # noqa: N803
    ) -> Dict[str, Any]:
        old = self.items.pop(Key["session_key"], None)
        if old is not None and ReturnValues == "ALL_OLD":
            return {"Attributes": old}
        return {}

    def scan(self, **kwargs: Any) -> Dict[str, Any]:
        # Single-page scan; no LastEvaluatedKey.
        return {"Items": [dict(v) for v in self.items.values()]}


def _make_store(table: FakeTable, default_ttl: Optional[int] = None):
    fake_resource = MagicMock()
    fake_resource.Table.return_value = table
    fake_boto3 = MagicMock()
    fake_boto3.resource = MagicMock(return_value=fake_resource)
    with patch.dict("sys.modules", {"boto3": fake_boto3}):
        return DynamoDBSessionStore(default_ttl=default_ttl)


def _memory(*texts: str) -> ConversationMemory:
    mem = ConversationMemory()
    for i, t in enumerate(texts):
        mem.add(Message(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content=t))
    return mem


# ======================================================================
# Save / load
# ======================================================================


def test_save_and_load_round_trip():
    store = _make_store(FakeTable())
    store.save("s1", _memory("Hello", "World"))
    loaded = store.load("s1")
    assert loaded is not None
    assert [m.content for m in loaded.get_history()] == ["Hello", "World"]


def test_load_missing_returns_none():
    assert _make_store(FakeTable()).load("nope") is None


def test_save_is_upsert_idempotent():
    table = FakeTable()
    store = _make_store(table)
    store.save("s1", _memory("a"))
    store.save("s1", _memory("a", "b", "c"))
    assert len(table.items) == 1
    assert len(store.load("s1").get_history()) == 3


def test_memory_stored_as_json_string():
    table = FakeTable()
    store = _make_store(table)
    store.save("s1", _memory("x"))
    assert isinstance(table.items["s1"]["memory_json"], str)


def test_save_preserves_created_at():
    table = FakeTable()
    store = _make_store(table)
    store.save("s1", _memory("a"))
    created = table.items["s1"]["created_at"]
    store.save("s1", _memory("a", "b"))
    assert table.items["s1"]["created_at"] == created
    assert table.items["s1"]["updated_at"] >= created


def test_numbers_are_decimal_on_write():
    table = FakeTable()
    store = _make_store(table)
    store.save("s1", _memory("a"))
    assert isinstance(table.items["s1"]["created_at"], Decimal)


# ======================================================================
# Namespace isolation
# ======================================================================


def test_namespace_isolates_same_session_id():
    store = _make_store(FakeTable())
    store.save("s1", _memory("alpha"), namespace="A")
    store.save("s1", _memory("beta"), namespace="B")
    assert store.load("s1", namespace="A").get_history()[0].content == "alpha"
    assert store.load("s1", namespace="B").get_history()[0].content == "beta"
    assert store.load("s1") is None


# ======================================================================
# list / delete / exists / branch
# ======================================================================


def test_list_returns_metadata():
    store = _make_store(FakeTable())
    store.save("s1", _memory("a", "b"))
    store.save("s2", _memory("c"))
    meta = {m.session_id: m for m in store.list()}
    assert set(meta) == {"s1", "s2"}
    assert isinstance(meta["s1"], SessionMetadata)
    assert meta["s1"].message_count == 2
    assert isinstance(meta["s1"].created_at, float)


def test_delete():
    store = _make_store(FakeTable())
    store.save("s1", _memory("a"))
    assert store.delete("s1") is True
    assert store.delete("s1") is False
    assert store.load("s1") is None


def test_exists():
    store = _make_store(FakeTable())
    assert store.exists("s1") is False
    store.save("s1", _memory("a"))
    assert store.exists("s1") is True


def test_branch_copies_and_is_independent():
    store = _make_store(FakeTable())
    store.save("src", _memory("hello"))
    store.branch("src", "dst")
    assert store.load("dst").get_history()[0].content == "hello"
    store.save("dst", _memory("changed"))
    assert store.load("src").get_history()[0].content == "hello"


def test_branch_missing_source_raises():
    store = _make_store(FakeTable())
    with pytest.raises(ValueError, match="not found"):
        store.branch("missing", "dst")


# ======================================================================
# search
# ======================================================================


def test_search_matches_and_ranks():
    store = _make_store(FakeTable())
    store.save("s1", _memory("billing discrepancy on the invoice", "checking billing"))
    store.save("s2", _memory("the weather is nice"))
    results = store.search("billing")
    assert [r.session_id for r in results] == ["s1"]
    assert results[0].score > 0
    assert any("billing" in s.lower() for s in results[0].matched_messages)


def test_search_namespace_filter():
    store = _make_store(FakeTable())
    store.save("s1", _memory("shared keyword"), namespace="A")
    store.save("s2", _memory("shared keyword"), namespace="B")
    assert {r.session_id for r in store.search("keyword", namespace="A")} == {"s1"}


def test_search_empty_query_returns_empty():
    store = _make_store(FakeTable())
    store.save("s1", _memory("anything"))
    assert store.search("") == []
    assert store.search("x", limit=0) == []


# ======================================================================
# Validation / TTL / missing dependency
# ======================================================================


def test_validation_rejects_bad_ids():
    store = _make_store(FakeTable())
    with pytest.raises(ValueError):
        store.save("", _memory("a"))
    with pytest.raises(ValueError):
        store.save("bad\x00", _memory("a"))


def test_ttl_stamps_expires_at():
    table = FakeTable()
    store = _make_store(table, default_ttl=3600)
    store.save("s1", _memory("a"))
    assert "expires_at" in table.items["s1"]
    assert isinstance(table.items["s1"]["expires_at"], int)


def test_missing_boto3_raises_helpful_error():
    with patch.dict("sys.modules", {"boto3": None}):
        with pytest.raises(ImportError, match=r"selectools\[aws\]"):
            DynamoDBSessionStore()
