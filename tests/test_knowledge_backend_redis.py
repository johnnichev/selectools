"""
Tests for RedisKnowledgeBackend.

Uses a FakeRedis (mirroring tests/test_sessions_redis.py) so no Redis server
or ``redis`` package install is required.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from selectools.knowledge import KnowledgeBackend, KnowledgeMemory
from selectools.knowledge_backends import RedisKnowledgeBackend

# ======================================================================
# Fake Redis client (bytes mode — decode_responses=False)
# ======================================================================


class FakeRedis:
    """In-memory stand-in for redis.Redis operating on bytes values."""

    def __init__(self) -> None:
        self.data: Dict[str, bytes] = {}
        self.ttls: Dict[str, int] = {}

    def get(self, key: str) -> Optional[bytes]:
        return self.data.get(key)

    def set(self, key: str, value: bytes) -> None:
        self.data[key] = value
        self.ttls.pop(key, None)

    def setex(self, key: str, ttl: int, value: bytes) -> None:
        self.data[key] = value
        self.ttls[key] = ttl


# ======================================================================
# Helpers
# ======================================================================


def _make_backend(
    fake_redis: Optional[FakeRedis] = None,
    key: str = "user-1",
    **kwargs: Any,
) -> RedisKnowledgeBackend:
    if fake_redis is None:
        fake_redis = FakeRedis()
    fake_module = MagicMock()
    fake_module.from_url = MagicMock(return_value=fake_redis)
    with patch.dict("sys.modules", {"redis": fake_module}):
        backend = RedisKnowledgeBackend(key=key, **kwargs)
    return backend


# ======================================================================
# Construction
# ======================================================================


class TestConstruction:
    def test_import_error_when_redis_missing(self) -> None:
        with patch.dict("sys.modules", {"redis": None}):
            with pytest.raises(ImportError, match="redis"):
                RedisKnowledgeBackend(key="u1")

    def test_satisfies_protocol_structurally(self) -> None:
        backend = _make_backend()
        assert callable(backend.load_bytes)
        assert callable(backend.save_bytes)

    def test_from_url_called_without_decode_responses(self) -> None:
        fake_redis = FakeRedis()
        fake_module = MagicMock()
        fake_module.from_url = MagicMock(return_value=fake_redis)
        with patch.dict("sys.modules", {"redis": fake_module}):
            RedisKnowledgeBackend(key="u1", url="redis://example:6379/2")
        args, kwargs = fake_module.from_url.call_args
        assert args[0] == "redis://example:6379/2"
        # Bytes contract: values must NOT be decoded to str.
        assert kwargs.get("decode_responses", False) is False

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

    def test_round_trip(self) -> None:
        backend = _make_backend()
        backend.save_bytes(b"\x00\xffraw bytes")
        assert backend.load_bytes() == b"\x00\xffraw bytes"

    def test_save_overwrites(self) -> None:
        backend = _make_backend()
        backend.save_bytes(b"v1")
        backend.save_bytes(b"v2")
        assert backend.load_bytes() == b"v2"

    def test_key_uses_prefix(self) -> None:
        fake_redis = FakeRedis()
        backend = _make_backend(fake_redis, key="user-9")
        backend.save_bytes(b"x")
        assert "selectools:knowledge:user-9" in fake_redis.data

    def test_custom_prefix(self) -> None:
        fake_redis = FakeRedis()
        backend = _make_backend(fake_redis, key="u1", prefix="app:mem:")
        backend.save_bytes(b"x")
        assert "app:mem:u1" in fake_redis.data

    def test_ttl_uses_setex(self) -> None:
        fake_redis = FakeRedis()
        backend = _make_backend(fake_redis, key="u1", ttl=3600)
        backend.save_bytes(b"x")
        assert fake_redis.ttls["selectools:knowledge:u1"] == 3600

    def test_no_ttl_uses_plain_set(self) -> None:
        fake_redis = FakeRedis()
        backend = _make_backend(fake_redis, key="u1")
        backend.save_bytes(b"x")
        assert "selectools:knowledge:u1" not in fake_redis.ttls

    def test_keys_are_isolated(self) -> None:
        fake_redis = FakeRedis()
        b1 = _make_backend(fake_redis, key="user-1")
        b2 = _make_backend(fake_redis, key="user-2")
        b1.save_bytes(b"alpha")
        b2.save_bytes(b"beta")
        assert b1.load_bytes() == b"alpha"
        assert b2.load_bytes() == b"beta"


# ======================================================================
# Integration with KnowledgeMemory
# ======================================================================


class TestKnowledgeMemoryIntegration:
    def test_end_to_end_round_trip(self, tmp_path) -> None:
        fake_redis = FakeRedis()
        backend = _make_backend(fake_redis, key="user-7")

        km1 = KnowledgeMemory(directory=str(tmp_path / "d1"), backend=backend)
        km1.remember("redis persisted fact", persistent=True, importance=0.9)

        km2 = KnowledgeMemory(directory=str(tmp_path / "d2"), backend=backend)
        assert "redis persisted fact" in km2.build_context()
        assert "redis persisted fact" in km2.get_persistent_facts()
