"""
Tests for MongoSessionStore backend.

Uses an in-memory fake MongoDB collection (injected via sys.modules, the same
mechanism the Redis backend tests use) so the full SessionStore protocol is
exercised without a real MongoDB server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from selectools.memory import ConversationMemory
from selectools.sessions import MongoSessionStore, SessionMetadata
from selectools.types import Message, Role

# ======================================================================
# Fake MongoDB
# ======================================================================


class FakeMongoCollection:
    """In-memory stand-in for a pymongo Collection (the subset used)."""

    def __init__(self) -> None:
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.indexes: List[Any] = []

    def create_index(self, key: Any, **kwargs: Any) -> None:
        self.indexes.append((key, kwargs))

    def find_one(
        self, flt: Dict[str, Any], projection: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        doc = self.docs.get(flt.get("_id"))
        return dict(doc) if doc is not None else None

    def replace_one(self, flt: Dict[str, Any], doc: Dict[str, Any], upsert: bool = False) -> Any:
        key = flt["_id"]
        existed = key in self.docs
        if existed or upsert:
            self.docs[key] = dict(doc)
        return MagicMock(matched_count=1 if existed else 0, upserted_id=None if existed else key)

    def find(self, flt: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        flt = flt or {}
        out = []
        for doc in self.docs.values():
            if all(doc.get(k) == v for k, v in flt.items()):
                out.append(dict(doc))
        return out

    def delete_one(self, flt: Dict[str, Any]) -> Any:
        key = flt.get("_id")
        if key in self.docs:
            del self.docs[key]
            return MagicMock(deleted_count=1)
        return MagicMock(deleted_count=0)

    def count_documents(self, flt: Dict[str, Any], limit: int = 0) -> int:
        key = flt.get("_id")
        return 1 if key in self.docs else 0


def _make_store(collection: FakeMongoCollection, default_ttl: Optional[int] = None):
    """Build a MongoSessionStore backed by the fake collection."""
    fake_client = MagicMock()
    fake_client.__getitem__.return_value.__getitem__.return_value = collection
    fake_module = MagicMock()
    fake_module.MongoClient = MagicMock(return_value=fake_client)
    with patch.dict("sys.modules", {"pymongo": fake_module}):
        return MongoSessionStore(default_ttl=default_ttl)


def _memory(*texts: str) -> ConversationMemory:
    mem = ConversationMemory()
    for i, t in enumerate(texts):
        mem.add(Message(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content=t))
    return mem


# ======================================================================
# Save / load
# ======================================================================


def test_save_and_load_round_trip():
    coll = FakeMongoCollection()
    store = _make_store(coll)
    store.save("s1", _memory("Hello", "World"))
    loaded = store.load("s1")
    assert loaded is not None
    assert [m.content for m in loaded.get_history()] == ["Hello", "World"]


def test_load_missing_returns_none():
    store = _make_store(FakeMongoCollection())
    assert store.load("nope") is None


def test_save_is_upsert_idempotent():
    coll = FakeMongoCollection()
    store = _make_store(coll)
    store.save("s1", _memory("a"))
    store.save("s1", _memory("a", "b", "c"))
    assert len(coll.docs) == 1
    assert store.load("s1") is not None
    assert len(store.load("s1").get_history()) == 3


def test_save_preserves_created_at():
    coll = FakeMongoCollection()
    store = _make_store(coll)
    store.save("s1", _memory("a"))
    created = coll.docs["s1"]["created_at"]
    store.save("s1", _memory("a", "b"))
    assert coll.docs["s1"]["created_at"] == created
    assert coll.docs["s1"]["updated_at"] >= created


# ======================================================================
# Namespace isolation
# ======================================================================


def test_namespace_isolates_same_session_id():
    coll = FakeMongoCollection()
    store = _make_store(coll)
    store.save("s1", _memory("alpha"), namespace="agentA")
    store.save("s1", _memory("beta"), namespace="agentB")
    assert store.load("s1", namespace="agentA").get_history()[0].content == "alpha"
    assert store.load("s1", namespace="agentB").get_history()[0].content == "beta"
    assert store.load("s1") is None  # bare key untouched


# ======================================================================
# list / delete / exists / branch
# ======================================================================


def test_list_returns_metadata():
    coll = FakeMongoCollection()
    store = _make_store(coll)
    store.save("s1", _memory("a", "b"))
    store.save("s2", _memory("c"))
    meta = {m.session_id: m for m in store.list()}
    assert set(meta) == {"s1", "s2"}
    assert isinstance(meta["s1"], SessionMetadata)
    assert meta["s1"].message_count == 2


def test_delete():
    store = _make_store(FakeMongoCollection())
    store.save("s1", _memory("a"))
    assert store.delete("s1") is True
    assert store.delete("s1") is False
    assert store.load("s1") is None


def test_exists():
    store = _make_store(FakeMongoCollection())
    assert store.exists("s1") is False
    store.save("s1", _memory("a"))
    assert store.exists("s1") is True


def test_branch_copies_session():
    store = _make_store(FakeMongoCollection())
    store.save("src", _memory("hello"))
    store.branch("src", "dst")
    assert store.load("dst").get_history()[0].content == "hello"
    # independent after branch
    store.save("dst", _memory("changed"))
    assert store.load("src").get_history()[0].content == "hello"


def test_branch_missing_source_raises():
    store = _make_store(FakeMongoCollection())
    with pytest.raises(ValueError, match="not found"):
        store.branch("missing", "dst")


# ======================================================================
# search
# ======================================================================


def test_search_matches_and_ranks():
    coll = FakeMongoCollection()
    store = _make_store(coll)
    store.save("s1", _memory("billing discrepancy on the invoice", "I will check billing"))
    store.save("s2", _memory("the weather is nice"))
    results = store.search("billing")
    assert len(results) == 1
    assert results[0].session_id == "s1"
    assert results[0].score > 0
    assert any("billing" in s.lower() for s in results[0].matched_messages)


def test_search_namespace_filter():
    coll = FakeMongoCollection()
    store = _make_store(coll)
    store.save("s1", _memory("shared keyword here"), namespace="A")
    store.save("s2", _memory("shared keyword here"), namespace="B")
    results = store.search("keyword", namespace="A")
    assert {r.session_id for r in results} == {"s1"}


def test_search_empty_query_returns_empty():
    store = _make_store(FakeMongoCollection())
    store.save("s1", _memory("anything"))
    assert store.search("") == []
    assert store.search("x", limit=0) == []


# ======================================================================
# Validation + TTL + missing dependency
# ======================================================================


def test_validation_rejects_bad_ids():
    store = _make_store(FakeMongoCollection())
    with pytest.raises(ValueError):
        store.save("", _memory("a"))
    with pytest.raises(ValueError):
        store.save("ok\x00null", _memory("a"))


def test_ttl_creates_index_and_stamps_expiry():
    coll = FakeMongoCollection()
    store = _make_store(coll, default_ttl=3600)
    assert any(idx[0] == "expires_at" for idx in coll.indexes)
    store.save("s1", _memory("a"))
    assert "expires_at" in coll.docs["s1"]


def test_missing_pymongo_raises_helpful_error():
    with patch.dict("sys.modules", {"pymongo": None}):
        with pytest.raises(ImportError, match=r"selectools\[mongo\]"):
            MongoSessionStore()
