#!/usr/bin/env python3
"""
Supabase Session Store — Postgres-backed sessions via Supabase PostgREST.

Demonstrates SupabaseSessionStore: ConversationMemory persisted to a Supabase
Postgres table as JSONB, with idempotent upserts and optional namespace
isolation. Fourth backend alongside JSON file, SQLite, and Redis.

No API key needed. Runs entirely offline with the built-in LocalProvider and
an in-process fake Supabase client so the demo works without a live project.
Swap the fake for `supabase.create_client(SUPABASE_URL, SERVICE_ROLE_KEY)` in
production.

Table DDL (run once in your Supabase project):

    create table if not exists public.selectools_sessions (
        session_id    text        primary key,
        memory_json   jsonb       not null,
        message_count integer     not null default 0,
        created_at    timestamptz not null default now(),
        updated_at    timestamptz not null default now()
    );
    alter table public.selectools_sessions enable row level security;

Prerequisites: pip install selectools[supabase]
Run: python examples/96_supabase_session_store.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from selectools import Agent, AgentConfig, ConversationMemory, Message, Role, tool
from selectools.providers.stubs import LocalProvider
from selectools.sessions import SupabaseSessionStore

# ── Minimal in-process Supabase fake (demo only) ────────────────────────────
# In production, replace this with:
#   from supabase import create_client
#   client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


class _FakeResponse:
    def __init__(self, data: Any) -> None:
        self.data = data


class _FakeQuery:
    def __init__(self, store: Dict[str, Dict[str, Any]], table: str) -> None:
        self._store, self._table, self._op = store, table, "select"
        self._payload: Optional[Dict[str, Any]] = None
        self._filters: List[tuple] = []
        self._cols = "*"
        self._limit: Optional[int] = None

    def select(self, cols: str = "*") -> "_FakeQuery":
        self._cols = cols
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters.append((col, val))
        return self

    def limit(self, n: int) -> "_FakeQuery":
        self._limit = n
        return self

    def upsert(self, payload: Dict[str, Any], **_: Any) -> "_FakeQuery":
        self._op, self._payload = "upsert", payload
        return self

    def delete(self) -> "_FakeQuery":
        self._op = "delete"
        return self

    def execute(self) -> _FakeResponse:
        rows = self._store.setdefault(self._table, {})
        if self._op == "upsert":
            assert self._payload is not None
            key = self._payload["session_id"]
            existing = rows.get(key, {})
            merged = {**existing, **self._payload}
            if "created_at" not in merged:
                merged["created_at"] = datetime.now(timezone.utc).isoformat()
            rows[key] = merged
            return _FakeResponse([merged])

        matching = [r for r in rows.values() if all(r.get(c) == v for c, v in self._filters)]
        if self._op == "delete":
            for r in matching:
                rows.pop(r["session_id"], None)
            return _FakeResponse(matching)

        if self._limit is not None:
            matching = matching[: self._limit]
        if self._cols != "*":
            cols = [c.strip() for c in self._cols.split(",")]
            matching = [{c: r[c] for c in cols if c in r} for r in matching]
        return _FakeResponse(matching)


class _FakeSupabaseClient:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self._data, name)


@tool(description="Get the current weather for a city")
def get_weather(city: str) -> str:
    return {"paris": "18C, sunny", "london": "12C, cloudy"}.get(city.lower(), f"No data for {city}")


def main() -> None:
    client = _FakeSupabaseClient()
    # SupabaseSessionStore lazily imports `supabase`. Patch the import so this
    # demo runs without the package installed. In real use you just do:
    #   store = SupabaseSessionStore(client=client)
    with patch.dict(sys.modules, {"supabase": MagicMock()}):
        store = SupabaseSessionStore(client=client, table_name="selectools_sessions")

    session_id = "demo-session"

    # ── 1. Agent auto-save/load via AgentConfig ─────────────────────────────
    print("=== Turn 1 — agent auto-saves to Supabase ===\n")
    memory = ConversationMemory(max_messages=20)
    agent = Agent(
        tools=[get_weather],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=2, session_store=store, session_id=session_id),
        memory=memory,
    )
    result = agent.run([Message(role=Role.USER, content="Weather in Paris?")])
    print(f"Agent: {result.content}")
    print(f"Row upserted: {store.exists(session_id)}\n")

    print("=== Turn 2 — fresh agent loads from the same session_id ===\n")
    restored = store.load(session_id)
    assert restored is not None
    print(f"Restored {len(restored)} messages from Postgres")
    agent2 = Agent(
        tools=[get_weather],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=2, session_store=store, session_id=session_id),
        memory=restored,
    )
    agent2.run([Message(role=Role.USER, content="Now London.")])
    print(f"After turn 2: {len(restored)} messages persisted\n")

    # ── 2. Namespace isolation (store-level, not agent-level) ──────────────
    # Use namespaces to keep multiple agents' sessions from colliding on the
    # same session_id. Pass namespace= directly to save/load/exists/delete.
    print("=== Namespace isolation ===\n")
    mem_a = ConversationMemory(max_messages=10)
    mem_a.add(Message(role=Role.USER, content="agent-a note"))
    mem_b = ConversationMemory(max_messages=10)
    mem_b.add(Message(role=Role.USER, content="agent-b note"))
    store.save("shared", mem_a, namespace="agent-a")
    store.save("shared", mem_b, namespace="agent-b")
    loaded_a = store.load("shared", namespace="agent-a")
    loaded_b = store.load("shared", namespace="agent-b")
    assert loaded_a is not None and loaded_b is not None
    print(f"  agent-a sees: {loaded_a.get_history()[0].content}")
    print(f"  agent-b sees: {loaded_b.get_history()[0].content}")
    print(f"  Exists bare 'shared': {store.exists('shared')}  (no collision)\n")

    # ── 3. Branch a session ────────────────────────────────────────────────
    print("=== Branching ===\n")
    store.branch(session_id, f"{session_id}-fork")
    for meta in store.list():
        print(f"  id={meta.session_id}  messages={meta.message_count}")

    # ── 4. Cleanup ─────────────────────────────────────────────────────────
    print("\n=== Cleanup ===")
    store.delete(session_id)
    store.delete(f"{session_id}-fork")
    store.delete("shared", namespace="agent-a")
    store.delete("shared", namespace="agent-b")
    print(f"Rows remaining: {len(store.list())}")


if __name__ == "__main__":
    main()
