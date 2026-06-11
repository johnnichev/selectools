"""Real-Redis smoke tests for RedisPendingStore (issue #58 / #82 residual).

tests/test_pending.py exercises RedisPendingStore against a FakeRedis whose
``eval`` REIMPLEMENTS the ``_TIGHTEN_TTL_LUA`` semantics in Python — the Lua
source itself is never executed there. These tests run the id-pinned
compare-and-set (and the GETDEL claim paths around it) against a real Redis
server so the script is execute-validated: cjson parsing, the id compare,
and the ``SET ... EX`` rewrite all happen server-side.

Skip-gating mirrors the external-service precedent in tests/integration/
(env-var + connection-attempt skip, e.g. the Ollama pattern in
test_live_native_tools.py): tests skip cleanly when the ``redis`` package is
missing or no server answers PING at ``SELECTOOLS_TEST_REDIS_URL`` /
``REDIS_URL`` (default ``redis://localhost:6379/0``).

Run locally with e.g.::

    docker run -d -p 6379:6379 redis
    pytest tests/test_pending_redis_smoke.py -v
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Iterator, List, Optional, Tuple

import pytest

from selectools.pending import (
    _TIGHTEN_TTL_LUA,
    ConfirmOutcome,
    RedisPendingStore,
)

redis = pytest.importorskip("redis", reason="redis package not installed")

REDIS_URL = os.environ.get(
    "SELECTOOLS_TEST_REDIS_URL",
    os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
)


@pytest.fixture(scope="module")
def redis_client() -> Iterator[Any]:
    """A real Redis client, or a clean skip when no server is reachable."""
    client = redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=2,
    )
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 - any connection failure means skip
        pytest.skip(f"no real Redis reachable at {REDIS_URL}: {exc}")
    yield client
    client.close()


@pytest.fixture
def store_and_client(redis_client: Any) -> Iterator[Tuple[RedisPendingStore, Any, str]]:
    """A RedisPendingStore on a unique key prefix, plus the raw client.

    The unique prefix isolates each test from concurrent runs against a
    shared server; teardown deletes every key under it.
    """
    prefix = f"selectools:test:pending:{uuid.uuid4().hex}:"
    store = RedisPendingStore(url=REDIS_URL, prefix=prefix)
    yield store, redis_client, prefix
    leftover = list(redis_client.scan_iter(match=f"{prefix}*"))
    if leftover:
        redis_client.delete(*leftover)


def _only_key(client: Any, prefix: str) -> str:
    keys = list(client.scan_iter(match=f"{prefix}*"))
    assert len(keys) == 1, f"expected exactly one key under {prefix}, got {keys}"
    return keys[0]


class TestTightenTtlLuaReal:
    def test_tighten_happy_path_shortens_record_and_server_ttl(
        self, store_and_client: Tuple[RedisPendingStore, Any, str]
    ) -> None:
        """(a) The Lua rewrites the matching record and refreshes SET EX."""
        store, client, prefix = store_and_client
        fired: List[str] = []
        record = store.stash(
            "user-1",
            kind="delete_invoice",
            preview="Delete invoice INV-42",
            executor=lambda: fired.append("boom") or "deleted",
            ttl_seconds=300.0,
        )
        key = _only_key(client, prefix)
        assert client.ttl(key) > 5  # sanity: server TTL reflects the long window

        before = time.time()
        updated = store.tighten_ttl("user-1", 5.0)
        assert updated is not None
        assert updated.pending_action_id == record.pending_action_id
        assert updated.expires_at < record.expires_at
        assert updated.expires_at <= before + 5.0 + 1.0
        # Server-side TTL was rewritten by the Lua's SET ... EX.
        server_ttl = client.ttl(key)
        assert 0 < server_ttl <= 5
        # The persisted payload is the tightened record (cjson round-trip
        # on the server did not mangle it).
        stored = json.loads(client.get(key))
        assert stored["pending_action_id"] == record.pending_action_id
        assert stored["expires_at"] == updated.expires_at
        # Executor survives: confirming inside the tightened window runs it.
        outcome = store.pop_if_confirmed("user-1", "yes")
        assert outcome is not None and outcome.executed
        assert fired == ["boom"]

    def test_lua_id_pin_misses_when_key_holds_a_different_record(
        self, store_and_client: Tuple[RedisPendingStore, Any, str]
    ) -> None:
        """(b) Direct EVAL: the script must NOT rewrite a re-stashed record.

        Simulates the GET -> EVAL race: record A is observed, a twin claims
        it via GETDEL and the same scope stashes B. Running the script with
        A's id while the key holds B must return 0 and leave B untouched
        (value AND TTL).
        """
        store, client, prefix = store_and_client
        record_a = store.stash(
            "user-1", kind="k", preview="A", executor=lambda: "a", ttl_seconds=300.0
        )
        key = _only_key(client, prefix)
        observed_payload = client.get(key)
        assert json.loads(observed_payload)["pending_action_id"] == record_a.pending_action_id

        # The race: a twin claims A, then the scope stashes a NEW pending B.
        assert client.getdel(key) is not None
        record_b = store.stash(
            "user-1", kind="k", preview="B", executor=lambda: "b", ttl_seconds=300.0
        )
        b_payload_before = client.get(key)
        b_ttl_before = client.ttl(key)

        # Replay tighten's write phase with A's observed identity: the Lua
        # compare must miss server-side.
        tightened = json.dumps({**json.loads(observed_payload), "expires_at": time.time() + 5.0})
        result = client.eval(_TIGHTEN_TTL_LUA, 1, key, record_a.pending_action_id, tightened, "5")
        assert result == 0
        # B untouched: same payload, TTL not shrunk to the 5s window.
        assert client.get(key) == b_payload_before
        assert client.ttl(key) > 5
        assert client.ttl(key) <= b_ttl_before
        live = store.get("user-1")
        assert live is not None
        assert live.pending_action_id == record_b.pending_action_id
        assert live.expires_at == record_b.expires_at

        # Control: pinned to B's actual id, the same script DOES rewrite.
        b_tightened = json.dumps({**json.loads(b_payload_before), "expires_at": time.time() + 5.0})
        result = client.eval(_TIGHTEN_TTL_LUA, 1, key, record_b.pending_action_id, b_tightened, "5")
        assert result == 1
        assert 0 < client.ttl(key) <= 5

    def test_store_tighten_misses_when_record_restashed_mid_flight(
        self, store_and_client: Tuple[RedisPendingStore, Any, str]
    ) -> None:
        """(b, store-level) tighten_ttl's own EVAL must miss after a re-stash.

        Same interleaving as the FakeRedis race test in test_pending.py,
        but the deciding compare-and-set runs inside a REAL Redis: the
        store's GET observes A; before the EVAL lands, A is claimed and B
        stashed. tighten_ttl must return None and B must survive untouched.
        """
        store, client, prefix = store_and_client
        fired: List[str] = []
        record_a = store.stash(
            "user-1",
            kind="delete_invoice",
            preview="Delete invoice INV-42",
            executor=lambda: fired.append("A") or "A-ran",
            ttl_seconds=300.0,
        )

        original_get = store._client.get
        state: dict = {}

        def racing_get(k: str) -> Optional[str]:
            value = original_get(k)
            store._client.get = original_get  # race fires once
            client.getdel(k)  # twin webhook claims A...
            store._executors.pop(record_a.pending_action_id, None)
            state["record_b"] = store.stash(  # ...and the scope stashes B
                "user-1",
                kind="delete_invoice",
                preview="Delete invoice INV-99",
                executor=lambda: fired.append("B") or "B-ran",
                ttl_seconds=300.0,
            )
            return value

        store._client.get = racing_get  # type: ignore[method-assign]
        try:
            assert store.tighten_ttl("user-1", 5.0) is None
        finally:
            store._client.get = original_get  # type: ignore[method-assign]

        record_b = state["record_b"]
        key = _only_key(client, prefix)
        # B survives untouched: identity, window, and server TTL.
        live = store.get("user-1")
        assert live is not None
        assert live.pending_action_id == record_b.pending_action_id
        assert live.expires_at == record_b.expires_at
        assert client.ttl(key) > 5
        # A never re-arms: the next confirm executes B, and only B.
        outcome = store.pop_if_intent("user-1", "confirm")
        assert outcome is not None and outcome.executed
        assert outcome.result == "B-ran"
        assert fired == ["B"]


class TestPopIfIntentReal:
    def test_confirm_path_end_to_end(
        self, store_and_client: Tuple[RedisPendingStore, Any, str]
    ) -> None:
        """(c) ignore -> tighten (EVAL) -> confirm, all against real Redis."""
        store, client, prefix = store_and_client
        fired: List[str] = []
        record = store.stash(
            "user-1",
            kind="delete_invoice",
            preview="Delete invoice INV-42",
            executor=lambda: fired.append("boom") or "deleted",
            ttl_seconds=300.0,
        )
        key = _only_key(client, prefix)

        # "ignore" preserves the pending but tightens its window via the
        # Lua EVAL — this is the production path that needs EVAL available.
        ignored = store.pop_if_intent("user-1", "ignore", ignore_ttl_seconds=30.0)
        assert ignored is not None and ignored.status == "ignored"
        assert ignored.record.pending_action_id == record.pending_action_id
        assert ignored.record.expires_at < record.expires_at
        assert 0 < client.ttl(key) <= 30
        assert fired == []

        # Structured confirm claims via GETDEL and executes exactly once.
        outcome = store.pop_if_intent("user-1", "confirm")
        assert outcome is not None and outcome.executed
        assert outcome.result == "deleted"
        assert fired == ["boom"]
        assert client.get(key) is None
        assert store.pop_if_intent("user-1", "confirm") is None


class TestDuplicateConfirmReal:
    def test_two_threads_confirm_exactly_once(
        self, store_and_client: Tuple[RedisPendingStore, Any, str]
    ) -> None:
        """(d) Concurrent twin webhooks against real Redis execute ONCE."""
        store, _client, _prefix = store_and_client
        fired: List[str] = []
        fired_lock = threading.Lock()

        def executor() -> str:
            with fired_lock:
                fired.append("boom")
            return "deleted"

        store.stash("user-1", kind="delete_invoice", preview="p", executor=executor)
        outcomes: List[Optional[ConfirmOutcome]] = []
        outcomes_lock = threading.Lock()
        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait()
            outcome = store.pop_if_confirmed("user-1", "yes")
            with outcomes_lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert fired == ["boom"]
        executed = [o for o in outcomes if o is not None and o.executed]
        assert len(executed) == 1
        assert executed[0].result == "deleted"
        # The twin saw the GETDEL miss.
        assert sum(1 for o in outcomes if o is None) == 1
        assert store.get("user-1") is None
