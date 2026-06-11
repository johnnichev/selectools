"""Tests for selectools.pending — deferred confirmation flow (issue #58).

The five mandatory failure tests from the review spec come first:

1. a late "yes" after TTL does not execute;
2. duplicate webhook delivery executes once;
3. an unrelated user's confirmation does not pop the action;
4. a changed action after preview (digest mismatch) asks for a fresh
   confirmation instead of executing;
5. non-confirmation messages fall through to the normal agent path.

Plus: serializer round-trip for PendingConfirmation (the ClassVar ``kind``
footgun #59 fixed) and RedisPendingStore tests against a FakeRedis (mirrors
tests/test_sessions_redis.py).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from selectools import Agent, AgentConfig
from selectools.pending import (
    DEFAULT_TTL_SECONDS,
    ChannelAgent,
    ConfirmOutcome,
    InMemoryPendingStore,
    PendingAction,
    PendingActionExistsError,
    PendingConfirmation,
    RedisPendingStore,
    RegexConfirmParser,
    compute_args_digest,
    stash_pending,
)
from selectools.providers.base import Provider
from selectools.tools import tool
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

_DUMMY_USAGE = UsageStats(
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
    cost_usd=0.0,
    model="test",
    provider="test",
)


class _ToolThenDoneProvider(Provider):
    """First call returns the given tool calls, second call returns plain text."""

    name = "tool-then-done"
    supports_streaming = False
    supports_async = True

    def __init__(self, tool_calls: List[ToolCall], final_text: str = "done") -> None:
        self.default_model = "test"
        self._tool_calls = tool_calls
        self._final_text = final_text
        self.call_count = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.call_count += 1
        if self.call_count == 1:
            return (
                Message(role=Role.ASSISTANT, content="", tool_calls=list(self._tool_calls)),
                _DUMMY_USAGE,
            )
        return Message(role=Role.ASSISTANT, content=self._final_text), _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


def _noop_tool() -> Any:
    @tool()
    def noop() -> str:
        """Do nothing."""
        return "noop"

    return noop


class _TextProvider(Provider):
    """Always returns plain text. Counts calls so tests can assert dispatch."""

    name = "text"
    supports_streaming = False
    supports_async = True

    def __init__(self, text: str = "llm reply") -> None:
        self.default_model = "test"
        self._text = text
        self.call_count = 0

    def complete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        self.call_count += 1
        return Message(role=Role.ASSISTANT, content=self._text), _DUMMY_USAGE

    async def acomplete(self, **kwargs: Any) -> Tuple[Message, UsageStats]:
        return self.complete(**kwargs)


# ---------------------------------------------------------------------------
# FakeRedis (mirrors tests/test_sessions_redis.py, plus GETDEL for the
# atomic-claim path)
# ---------------------------------------------------------------------------


class FakeRedis:
    """In-memory fake Redis client for testing."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._ttls: Dict[str, float] = {}

    def _evict_if_expired(self, key: str) -> None:
        if key in self._ttls and time.time() > self._ttls[key]:
            self._store.pop(key, None)
            self._ttls.pop(key, None)

    def get(self, key: str) -> Optional[str]:
        self._evict_if_expired(key)
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value
        self._ttls.pop(key, None)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._ttls[key] = time.time() + ttl

    def getdel(self, key: str) -> Optional[str]:
        self._evict_if_expired(key)
        value = self._store.pop(key, None)
        self._ttls.pop(key, None)
        return value

    def delete(self, *keys: str) -> int:
        removed = 0
        for k in keys:
            self._evict_if_expired(k)
            if k in self._store:
                del self._store[k]
                self._ttls.pop(k, None)
                removed += 1
        return removed


def _make_redis_store(fake_redis: FakeRedis, **kwargs: Any) -> RedisPendingStore:
    """Create a RedisPendingStore with a mocked redis import."""
    from unittest.mock import MagicMock, patch

    fake_module = MagicMock()
    fake_module.from_url = MagicMock(return_value=fake_redis)

    with patch.dict("sys.modules", {"redis": fake_module}):
        store = RedisPendingStore(url="redis://localhost:6379/0", **kwargs)
    return store


@pytest.fixture
def redis_store() -> RedisPendingStore:
    return _make_redis_store(FakeRedis())


# ---------------------------------------------------------------------------
# Mandatory failure tests (review spec, verbatim requirements)
# ---------------------------------------------------------------------------


class TestMandatoryFailures:
    def test_late_yes_after_ttl_does_not_execute(self) -> None:
        """1. A late "yes" after TTL does NOT execute."""
        store = InMemoryPendingStore()
        fired: List[str] = []
        store.stash(
            "user-1",
            kind="delete_invoice",
            preview="Delete invoice INV-42",
            executor=lambda: fired.append("boom") or "deleted",
            ttl_seconds=0.01,
        )
        time.sleep(0.05)
        outcome = store.pop_if_confirmed("user-1", "yes")
        assert fired == []
        assert outcome is not None
        assert outcome.status == "expired"
        assert outcome.result is None
        # The pending is gone; a fresh confirmation cycle is required.
        assert store.get("user-1") is None

    def test_duplicate_webhook_delivery_executes_once_sequential(self) -> None:
        """2. Duplicate webhook delivery executes ONCE (sequential redelivery)."""
        store = InMemoryPendingStore()
        fired: List[str] = []
        store.stash(
            "user-1",
            kind="delete_invoice",
            preview="Delete invoice INV-42",
            executor=lambda: fired.append("boom") or "deleted",
        )
        first = store.pop_if_confirmed("user-1", "yes")
        second = store.pop_if_confirmed("user-1", "yes")
        assert first is not None and first.status == "executed"
        assert first.result == "deleted"
        assert second is None
        assert fired == ["boom"]

    def test_duplicate_webhook_delivery_executes_once_concurrent(self) -> None:
        """2b. Concurrent twin webhooks (user double-taps) execute ONCE."""
        store = InMemoryPendingStore()
        fired: List[str] = []
        lock = threading.Lock()

        def executor() -> str:
            with lock:
                fired.append("boom")
            return "deleted"

        store.stash("user-1", kind="k", preview="p", executor=executor)
        results: List[Optional[ConfirmOutcome]] = []
        threads = [
            threading.Thread(target=lambda: results.append(store.pop_if_confirmed("user-1", "sim")))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert fired == ["boom"]
        executed = [r for r in results if r is not None and r.status == "executed"]
        assert len(executed) == 1

    def test_unrelated_user_confirmation_does_not_pop(self) -> None:
        """3. An unrelated user's confirmation does NOT pop the action."""
        store = InMemoryPendingStore()
        fired: List[str] = []
        store.stash(
            "user-a",
            kind="delete_invoice",
            preview="Delete invoice INV-42",
            executor=lambda: fired.append("boom") or "deleted",
        )
        outcome = store.pop_if_confirmed("user-b", "yes")
        assert outcome is None
        assert fired == []
        # user-a's pending action is untouched.
        record = store.get("user-a")
        assert record is not None and record.status == "pending"

    def test_unrelated_conversation_confirmation_does_not_pop(self) -> None:
        """3b. Same user, different conversation scope, does NOT pop."""
        store = InMemoryPendingStore()
        fired: List[str] = []
        store.stash(
            "user-a",
            kind="k",
            preview="p",
            executor=lambda: fired.append("boom") or "ok",
            channel_id="whatsapp",
            conversation_id="conv-1",
        )
        outcome = store.pop_if_confirmed(
            "user-a", "yes", channel_id="whatsapp", conversation_id="conv-2"
        )
        assert outcome is None
        assert fired == []
        assert store.get("user-a", channel_id="whatsapp", conversation_id="conv-1") is not None

    def test_digest_mismatch_requires_fresh_confirmation(self) -> None:
        """4. A changed action after preview (digest mismatch) does not execute."""
        store = InMemoryPendingStore()
        fired: List[str] = []
        store.stash(
            "user-1",
            kind="delete_last_expense",
            preview="Delete R$47 — mercado",
            executor=lambda: fired.append("boom") or "deleted",
            args={"expense_id": 47},
        )
        # Between preview and confirm the target changed (e.g. a newer
        # expense was logged). The confirming turn recomputes the digest of
        # what would execute NOW and it no longer matches the preview.
        changed = compute_args_digest({"expense_id": 99})
        outcome = store.pop_if_confirmed("user-1", "yes", args_digest=changed)
        assert fired == []
        assert outcome is not None
        assert outcome.status == "digest_mismatch"
        assert outcome.result is None
        # The stale pending is dropped — a fresh confirmation must be requested.
        assert store.get("user-1") is None
        store.stash("user-1", kind="k", preview="p", executor=lambda: "ok")

    def test_matching_digest_executes(self) -> None:
        """4b. When the recomputed digest still matches, execution proceeds."""
        store = InMemoryPendingStore()
        record = store.stash(
            "user-1",
            kind="k",
            preview="p",
            executor=lambda: "ok",
            args={"expense_id": 47},
        )
        assert record.args_digest == compute_args_digest({"expense_id": 47})
        outcome = store.pop_if_confirmed("user-1", "yes", args_digest=record.args_digest)
        assert outcome is not None and outcome.status == "executed"

    def test_non_confirmation_falls_through_to_agent_path(self) -> None:
        """5. Non-confirmation messages fall through to the normal agent path."""
        provider = _TextProvider("here is your summary")
        agent = Agent(
            tools=[_noop_tool()],
            provider=provider,
            config=AgentConfig(max_iterations=3),
        )
        store = InMemoryPendingStore()
        store.stash("user-1", kind="k", preview="p", executor=lambda: "deleted")
        channel = ChannelAgent(agent, store=store)

        result = channel.ask_channel("user-1", "what did I spend this month?")
        assert provider.call_count == 1
        assert result.content == "here is your summary"
        # The stale pending was dropped — the user moved on (Sheriff behavior:
        # any other reply must never leave a destructive action armed).
        assert store.get("user-1") is None


# ---------------------------------------------------------------------------
# RegexConfirmParser (PT/EN/ES)
# ---------------------------------------------------------------------------


class TestRegexConfirmParser:
    @pytest.fixture
    def parser(self) -> RegexConfirmParser:
        return RegexConfirmParser()

    @pytest.mark.parametrize(
        "msg",
        [
            "sim",
            "Sim, pode apagar",
            "confirmo",
            "confirmar",
            "confirmado",
            "pode apagar",
            "pode deletar",
            "pode cancelar",
            "pode remover",
            "yes",
            "Yes!",
            "yep",
            "yeah",
            "confirm",
            "confirmed",
            "sí",
            "si",
            "sí, borra",
            "puedes borrar",
            "puede eliminar",
        ],
    )
    def test_confirms(self, parser: RegexConfirmParser, msg: str) -> None:
        assert parser.is_confirm(msg)

    @pytest.mark.parametrize(
        "msg",
        [
            "não",
            "nao",
            "Não, deixa",
            "no",
            "cancel",
            "cancela",
            "cancelar",
            "deixa",
            "nada",
            "nunca",
            "never",
            "olvida",
        ],
    )
    def test_cancels(self, parser: RegexConfirmParser, msg: str) -> None:
        assert parser.is_cancel(msg)

    @pytest.mark.parametrize(
        "msg",
        [
            "",
            "ok",
            "claro",
            "pode",
            "isso",
            "what did I spend?",
            "yesterday was fun",
            "simples assim",
            "si quieres puedo mirar mañana",
            "nothing to see",
            "now what",
            "delete the other one instead",
        ],
    )
    def test_neither(self, parser: RegexConfirmParser, msg: str) -> None:
        """Ambiguous acknowledgments and ordinary messages must NOT confirm.

        "ok" / "claro" / "pode" / "isso" are common Brazilian-Portuguese
        acknowledgments in NON-destructive replies; bare Spanish "si" mid-
        sentence is the conditional "if", not "yes" (Sheriff bug-hunt #10).
        """
        assert not parser.is_confirm(msg)
        assert not parser.is_cancel(msg)

    def test_version_is_stamped(self, parser: RegexConfirmParser) -> None:
        assert parser.version
        store = InMemoryPendingStore(parser=parser)
        record = store.stash("u", kind="k", preview="p", executor=lambda: "ok")
        assert record.parser_version == parser.version


# ---------------------------------------------------------------------------
# compute_args_digest
# ---------------------------------------------------------------------------


class TestArgsDigest:
    def test_deterministic_and_order_insensitive(self) -> None:
        a = compute_args_digest({"x": 1, "y": "z"})
        b = compute_args_digest({"y": "z", "x": 1})
        assert a == b
        assert len(a) == 64  # sha256 hex

    def test_different_args_differ(self) -> None:
        assert compute_args_digest({"x": 1}) != compute_args_digest({"x": 2})

    def test_none_is_empty_digest(self) -> None:
        assert compute_args_digest(None) == compute_args_digest({})


# ---------------------------------------------------------------------------
# InMemoryPendingStore lifecycle
# ---------------------------------------------------------------------------


class TestInMemoryPendingStore:
    def test_stash_returns_pending_record(self) -> None:
        store = InMemoryPendingStore()
        record = store.stash(
            "u1",
            kind="delete_invoice",
            preview="Delete INV-42",
            executor=lambda: "ok",
            args={"id": 42},
        )
        assert record.pending_action_id
        assert record.user_id == "u1"
        assert record.kind == "delete_invoice"
        assert record.preview == "Delete INV-42"
        assert record.status == "pending"
        assert record.args_digest == compute_args_digest({"id": 42})
        assert record.expires_at > record.requested_at

    def test_stash_refuses_to_overwrite_unexpired_pending(self) -> None:
        """A second stash must not silently replace the previewed action."""
        store = InMemoryPendingStore()
        store.stash("u1", kind="a", preview="A", executor=lambda: "a")
        with pytest.raises(PendingActionExistsError):
            store.stash("u1", kind="b", preview="B", executor=lambda: "b")

    def test_stash_replaces_expired_pending(self) -> None:
        store = InMemoryPendingStore()
        store.stash("u1", kind="a", preview="A", executor=lambda: "a", ttl_seconds=0.01)
        time.sleep(0.05)
        record = store.stash("u1", kind="b", preview="B", executor=lambda: "b")
        assert record.kind == "b"

    def test_cancel_message_does_not_execute(self) -> None:
        store = InMemoryPendingStore()
        fired: List[str] = []
        store.stash("u1", kind="k", preview="p", executor=lambda: fired.append("x") or "ok")
        outcome = store.pop_if_confirmed("u1", "não")
        assert outcome is None
        assert fired == []

    def test_drop_removes_pending(self) -> None:
        store = InMemoryPendingStore()
        store.stash("u1", kind="k", preview="p", executor=lambda: "ok")
        dropped = store.drop("u1")
        assert dropped is not None and dropped.status == "cancelled"
        assert store.get("u1") is None
        assert store.drop("u1") is None

    def test_consumed_record_carries_outcome(self) -> None:
        store = InMemoryPendingStore()
        store.stash("u1", kind="k", preview="p", executor=lambda: "it is done")
        outcome = store.pop_if_confirmed("u1", "confirm")
        assert outcome is not None
        assert outcome.record.status == "consumed"
        assert outcome.record.outcome == "it is done"
        assert outcome.result == "it is done"

    def test_lru_eviction_bounds_map(self) -> None:
        store = InMemoryPendingStore(max_entries=3)
        for i in range(5):
            store.stash(f"u{i}", kind="k", preview="p", executor=lambda: "ok")
        assert store.get("u0") is None
        assert store.get("u1") is None
        assert store.get("u4") is not None

    def test_executor_exception_consumes_pending(self) -> None:
        """A failing executor must not leave the action re-confirmable."""
        store = InMemoryPendingStore()

        def boom() -> str:
            raise RuntimeError("db down")

        store.stash("u1", kind="k", preview="p", executor=boom)
        with pytest.raises(RuntimeError):
            store.pop_if_confirmed("u1", "yes")
        assert store.get("u1") is None
        assert store.pop_if_confirmed("u1", "yes") is None


# ---------------------------------------------------------------------------
# stash_pending contextvar plumbing + ChannelAgent
# ---------------------------------------------------------------------------


class TestChannelAgent:
    def _build_channel(
        self, store: InMemoryPendingStore, fired: List[str]
    ) -> Tuple[ChannelAgent, _ToolThenDoneProvider]:
        @tool()
        def delete_invoice(invoice_id: str) -> PendingConfirmation:
            """Delete an invoice (destructive — requires confirmation)."""
            preview = f"Delete invoice {invoice_id}"
            stash_pending(
                kind="delete_invoice",
                preview=preview,
                executor=lambda: fired.append(invoice_id) or f"Deleted {invoice_id}",
                args={"invoice_id": invoice_id},
            )
            return PendingConfirmation(
                action="delete_invoice",
                preview=preview,
                user_prompt="Reply 'yes' to confirm or 'no' to cancel.",
            )

        provider = _ToolThenDoneProvider(
            [ToolCall(tool_name="delete_invoice", parameters={"invoice_id": "INV-42"}, id="c1")],
            final_text="I need your confirmation: delete INV-42?",
        )
        agent = Agent(
            tools=[delete_invoice],
            provider=provider,
            config=AgentConfig(max_iterations=3),
        )
        return ChannelAgent(agent, store=store), provider

    def test_two_turn_webhook_confirm_flow(self) -> None:
        store = InMemoryPendingStore()
        fired: List[str] = []
        channel, provider = self._build_channel(store, fired)

        # Turn 1: webhook delivers the destructive request. The tool stashes
        # and the agent returns a confirmation prompt. Nothing executed yet.
        first = channel.ask_channel("user-1", "delete invoice INV-42")
        assert fired == []
        record = store.get("user-1")
        assert record is not None and record.kind == "delete_invoice"
        assert first.content == "I need your confirmation: delete INV-42?"

        # Turn 2: a SEPARATE webhook turn delivers the "yes". The LLM is
        # bypassed; the stashed executor runs.
        calls_before = provider.call_count
        second = channel.ask_channel("user-1", "sim")
        assert fired == ["INV-42"]
        assert second.content == "Deleted INV-42"
        assert provider.call_count == calls_before
        assert store.get("user-1") is None

    def test_two_turn_webhook_cancel_flow(self) -> None:
        store = InMemoryPendingStore()
        fired: List[str] = []
        channel, provider = self._build_channel(store, fired)

        channel.ask_channel("user-1", "delete invoice INV-42")
        calls_before = provider.call_count
        result = channel.ask_channel("user-1", "não")
        assert fired == []
        assert store.get("user-1") is None
        assert provider.call_count == calls_before
        assert "Delete invoice INV-42" in result.content

    def test_late_confirm_via_channel_does_not_execute(self) -> None:
        """A late "yes" must NOT execute. The expired record has been
        evicted (in-memory eviction mirrors Redis server-side TTL), so the
        message falls through to the normal agent path like any other."""
        store = InMemoryPendingStore(default_ttl_seconds=0.01)
        fired: List[str] = []
        channel, provider = self._build_channel(store, fired)

        channel.ask_channel("user-1", "delete invoice INV-42")
        calls_before = provider.call_count
        time.sleep(0.05)
        channel.ask_channel("user-1", "yes")
        assert fired == []
        # The "yes" was dispatched to the LLM, not to a stale executor.
        assert provider.call_count > calls_before

    def test_confirm_without_pending_falls_through_to_agent(self) -> None:
        provider = _TextProvider("hello!")
        agent = Agent(tools=[_noop_tool()], provider=provider, config=AgentConfig(max_iterations=3))
        channel = ChannelAgent(agent, store=InMemoryPendingStore())
        result = channel.ask_channel("user-1", "yes")
        assert provider.call_count == 1
        assert result.content == "hello!"

    def test_stash_pending_outside_channel_run_is_noop(self) -> None:
        assert stash_pending(kind="k", preview="p", executor=lambda: "ok") is None

    def test_pending_scoped_to_channel_and_conversation(self) -> None:
        store = InMemoryPendingStore()
        fired: List[str] = []
        channel, _provider = self._build_channel(store, fired)

        channel.ask_channel(
            "user-1", "delete invoice INV-42", channel_id="wa", conversation_id="c1"
        )
        assert store.get("user-1", channel_id="wa", conversation_id="c1") is not None
        # A "yes" from another conversation cannot fire it.
        channel.ask_channel("user-1", "yes", channel_id="wa", conversation_id="c2")
        assert fired == []
        assert store.get("user-1", channel_id="wa", conversation_id="c1") is not None


# ---------------------------------------------------------------------------
# PendingConfirmation serializer round-trip (the #59 ClassVar footgun)
# ---------------------------------------------------------------------------


class TestPendingConfirmationSerialization:
    def test_kind_survives_tool_serialize_result(self) -> None:
        @tool()
        def dummy() -> str:
            """Dummy."""
            return "ok"

        result = PendingConfirmation(
            action="delete_invoice",
            preview="Delete INV-42",
            user_prompt="Reply 'yes' to confirm.",
        )
        data = json.loads(dummy._serialize_result(result))
        assert data["kind"] == "pending_confirmation"
        assert data["action"] == "delete_invoice"
        assert data["preview"] == "Delete INV-42"
        assert data["user_prompt"] == "Reply 'yes' to confirm."

    def test_is_frozen_toolresult(self) -> None:
        result = PendingConfirmation(action="a", preview="p", user_prompt="u")
        assert PendingConfirmation.kind == "pending_confirmation"
        with pytest.raises(Exception):
            result.action = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RedisPendingStore (FakeRedis)
# ---------------------------------------------------------------------------


class TestRedisPendingStore:
    def test_stash_and_confirm_executes_once(self, redis_store: RedisPendingStore) -> None:
        fired: List[str] = []
        redis_store.stash("u1", kind="k", preview="p", executor=lambda: fired.append("x") or "done")
        first = redis_store.pop_if_confirmed("u1", "yes")
        second = redis_store.pop_if_confirmed("u1", "yes")
        assert first is not None and first.status == "executed" and first.result == "done"
        assert second is None
        assert fired == ["x"]

    def test_stash_refuses_unexpired_existing(self, redis_store: RedisPendingStore) -> None:
        redis_store.stash("u1", kind="a", preview="A", executor=lambda: "a")
        with pytest.raises(PendingActionExistsError):
            redis_store.stash("u1", kind="b", preview="B", executor=lambda: "b")

    def test_ttl_expiry_does_not_execute(self, redis_store: RedisPendingStore) -> None:
        fired: List[str] = []
        redis_store.stash(
            "u1",
            kind="k",
            preview="p",
            executor=lambda: fired.append("x") or "ok",
            ttl_seconds=0.01,
        )
        time.sleep(1.1)  # server-side TTL is whole seconds (rounded up to 1)
        outcome = redis_store.pop_if_confirmed("u1", "yes")
        assert fired == []
        assert outcome is None or outcome.status == "expired"

    def test_unrelated_user_does_not_pop(self, redis_store: RedisPendingStore) -> None:
        fired: List[str] = []
        redis_store.stash("u-a", kind="k", preview="p", executor=lambda: fired.append("x") or "ok")
        assert redis_store.pop_if_confirmed("u-b", "yes") is None
        assert fired == []
        assert redis_store.get("u-a") is not None

    def test_digest_mismatch_does_not_execute(self, redis_store: RedisPendingStore) -> None:
        fired: List[str] = []
        redis_store.stash(
            "u1",
            kind="k",
            preview="p",
            executor=lambda: fired.append("x") or "ok",
            args={"id": 1},
        )
        outcome = redis_store.pop_if_confirmed(
            "u1", "yes", args_digest=compute_args_digest({"id": 2})
        )
        assert fired == []
        assert outcome is not None and outcome.status == "digest_mismatch"
        assert redis_store.get("u1") is None

    def test_executor_factory_rebuilds_after_process_restart(
        self, redis_store: RedisPendingStore
    ) -> None:
        """The record survives in Redis; the closure does not. A factory
        registered by kind rebuilds the executor from the persisted record."""
        fired: List[str] = []
        redis_store.stash(
            "u1",
            kind="delete_invoice",
            preview="Delete INV-42",
            executor=lambda: "from-closure",
            args={"invoice_id": "INV-42"},
        )
        # Simulate the confirming webhook landing on a fresh process: the
        # process-local closure registry is empty.
        redis_store._executors.clear()

        def factory(record: PendingAction) -> Any:
            assert record.args == {"invoice_id": "INV-42"}
            return lambda: fired.append(record.args["invoice_id"]) or "from-factory"

        redis_store.register_executor_factory("delete_invoice", factory)
        outcome = redis_store.pop_if_confirmed("u1", "yes")
        assert outcome is not None and outcome.status == "executed"
        assert outcome.result == "from-factory"
        assert fired == ["INV-42"]

    def test_no_executor_and_no_factory_does_not_execute(
        self, redis_store: RedisPendingStore
    ) -> None:
        redis_store.stash("u1", kind="k", preview="p", executor=lambda: "ok")
        redis_store._executors.clear()
        outcome = redis_store.pop_if_confirmed("u1", "yes")
        assert outcome is not None and outcome.status == "no_executor"
        assert outcome.result is None
        # Claimed and dropped: a fresh confirmation cycle is required.
        assert redis_store.get("u1") is None

    def test_drop_removes_record_and_executor(self, redis_store: RedisPendingStore) -> None:
        record = redis_store.stash("u1", kind="k", preview="p", executor=lambda: "ok")
        assert record.pending_action_id in redis_store._executors
        dropped = redis_store.drop("u1")
        assert dropped is not None and dropped.status == "cancelled"
        assert redis_store.get("u1") is None
        assert record.pending_action_id not in redis_store._executors

    def test_record_round_trips_through_json(self, redis_store: RedisPendingStore) -> None:
        record = redis_store.stash(
            "u1",
            kind="k",
            preview="p",
            executor=lambda: "ok",
            args={"a": 1},
            channel_id="wa",
            conversation_id="c9",
        )
        loaded = redis_store.get("u1", channel_id="wa", conversation_id="c9")
        assert loaded is not None
        assert loaded.pending_action_id == record.pending_action_id
        assert loaded.args == {"a": 1}
        assert loaded.channel_id == "wa"
        assert loaded.conversation_id == "c9"
        assert loaded.args_digest == record.args_digest

    def test_missing_redis_raises_import_error(self) -> None:
        from unittest.mock import patch

        with patch.dict("sys.modules", {"redis": None}):
            with pytest.raises(ImportError, match="redis"):
                RedisPendingStore()

    def test_default_ttl_constant(self) -> None:
        assert DEFAULT_TTL_SECONDS == 60.0
