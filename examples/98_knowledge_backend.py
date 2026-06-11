#!/usr/bin/env python3
"""
Knowledge Backends — persist KnowledgeMemory across ephemeral deploys.

Demonstrates the KnowledgeBackend protocol: KnowledgeMemory keeps using a
fast local directory as scratch space, while a backend (Supabase or Redis)
persists a snapshot of that directory between requests. Railway, Lambda, and
Cloud Run wipe /tmp between deploys — the backend doesn't.

No API key needed. Runs entirely offline with an in-process fake Supabase
client so the demo works without a live project. Swap the fake for
`supabase.create_client(SUPABASE_URL, SERVICE_ROLE_KEY)` in production.

Table DDL (run once in your Supabase project):

    create table if not exists public.selectools_knowledge (
        key        text        primary key,
        data       text        not null,
        updated_at timestamptz not null default now()
    );
    alter table public.selectools_knowledge enable row level security;

Prerequisites: pip install selectools[supabase]
Run: python examples/98_knowledge_backend.py
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from selectools import KnowledgeMemory
from selectools.knowledge_backends import SupabaseKnowledgeBackend

# ── Minimal in-process Supabase fake (demo only) ────────────────────────────


class _FakeResponse:
    def __init__(self, data: Any) -> None:
        self.data = data


class _FakeQuery:
    def __init__(self, rows: Dict[str, Dict[str, Any]]) -> None:
        self._rows = rows
        self._filters: List[tuple] = []
        self._payload: Optional[Dict[str, Any]] = None
        self._conflict: Optional[str] = None

    def select(self, _cols: str = "*") -> "_FakeQuery":
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters.append((col, val))
        return self

    def limit(self, _n: int) -> "_FakeQuery":
        return self

    def upsert(self, payload: Dict[str, Any], on_conflict: str = "key") -> "_FakeQuery":
        self._payload = payload
        self._conflict = on_conflict
        return self

    def execute(self) -> _FakeResponse:
        if self._payload is not None:
            self._rows[self._payload[self._conflict]] = dict(self._payload)
            return _FakeResponse([self._payload])
        rows = list(self._rows.values())
        for col, val in self._filters:
            rows = [r for r in rows if r.get(col) == val]
        return _FakeResponse(rows)


class _FakeSupabaseClient:
    def __init__(self) -> None:
        self._tables: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self._tables.setdefault(name, {}))


def main() -> None:
    client = _FakeSupabaseClient()

    # The fake client means no real `supabase` package is needed for the demo.
    with patch.dict("sys.modules", {"supabase": MagicMock()}):
        # ── Deploy 1: remember facts, backend persists automatically ────────
        with tempfile.TemporaryDirectory() as scratch1:
            memory = KnowledgeMemory(
                directory=scratch1,
                backend=SupabaseKnowledgeBackend(client, key="user-123"),
            )
            memory.remember("User prefers dark mode", persistent=True, importance=0.9)
            memory.remember("Working on the Q3 report", category="context", importance=0.6)
            print("Deploy 1 context:")
            print(memory.build_context())

        # ── Deploy 2: fresh /tmp, same backend key — nothing was lost ──────
        with tempfile.TemporaryDirectory() as scratch2:
            memory = KnowledgeMemory(
                directory=scratch2,
                backend=SupabaseKnowledgeBackend(client, key="user-123"),
            )
            print("\nDeploy 2 context (restored from Supabase):")
            print(memory.build_context())
            assert "dark mode" in memory.build_context()

        # ── Keys isolate users ──────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as scratch3:
            other = KnowledgeMemory(
                directory=scratch3,
                backend=SupabaseKnowledgeBackend(client, key="user-456"),
            )
            print("\nDifferent key starts empty:", repr(other.build_context()))

    print("\nDone. In production, create the client with supabase.create_client(...).")


if __name__ == "__main__":
    main()
