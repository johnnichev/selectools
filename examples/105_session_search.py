#!/usr/bin/env python3
"""
Session History Search — find past conversations across stored sessions.

Demonstrates the @beta `SessionStore.search()` method: an agent can
"remember what we discussed last Tuesday" by querying message content
across every stored session, optionally scoped to a namespace.

Backends:
- SQLiteSessionStore: FTS5 full-text index with bm25 ranking (built on
  save, backfilled lazily for databases created before the feature).
- JsonFileSessionStore: linear scan with case-insensitive term-frequency
  scoring — fine at file-store scale.

No API key needed. Runs entirely offline.

Run: python examples/105_session_search.py
"""

from __future__ import annotations

import tempfile

from selectools import ConversationMemory, Message, Role
from selectools.sessions import JsonFileSessionStore, SQLiteSessionStore


def _conversation(*turns: str) -> ConversationMemory:
    """Build a memory alternating user/assistant turns."""
    memory = ConversationMemory(max_messages=50)
    for i, text in enumerate(turns):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        memory.add(Message(role=role, content=text))
    return memory


def seed(store: JsonFileSessionStore | SQLiteSessionStore) -> None:
    store.save(
        "support-001",
        _conversation(
            "I think there is a billing discrepancy on my invoice.",
            "I see a duplicate charge of $42 on the March invoice. Refund issued.",
        ),
        namespace="user:alice",
    )
    store.save(
        "support-002",
        _conversation(
            "My deployment keeps crashing on boot.",
            "The container was missing libgomp. Added it to the base image.",
        ),
        namespace="user:alice",
    )
    store.save(
        "support-003",
        _conversation(
            "Another billing question: was the discrepancy fixed?",
            "Yes — the billing discrepancy was resolved last week.",
        ),
        namespace="user:bob",
    )


def show(title: str, results: list) -> None:
    print(f"\n{title}")
    if not results:
        print("  (no matches)")
    for r in results:
        print(f"  {r.session_id}  score={r.score:.2f}")
        for snippet in r.matched_messages:
            print(f"    - {snippet}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # ── SQLite backend: FTS5 + bm25 ─────────────────────────────────
        sqlite_store = SQLiteSessionStore(db_path=f"{tmpdir}/sessions.db")
        seed(sqlite_store)

        show(
            "SQLite — search('billing discrepancy') across all namespaces:",
            sqlite_store.search("billing discrepancy"),
        )
        show(
            "SQLite — same query scoped to namespace='user:alice':",
            sqlite_store.search("billing discrepancy", namespace="user:alice"),
        )

        # ── JSON file backend: term-frequency scan ──────────────────────
        json_store = JsonFileSessionStore(directory=f"{tmpdir}/sessions")
        seed(json_store)

        show(
            "JsonFile — search('crashing', limit=1):",
            json_store.search("crashing", limit=1),
        )

        # Loading a result: pass the id back with the same namespace.
        top = sqlite_store.search("billing", namespace="user:bob")[0]
        memory = sqlite_store.load(top.session_id, namespace="user:bob")
        assert memory is not None
        print(f"\nReloaded session {top.session_id!r}: {len(memory)} messages")


if __name__ == "__main__":
    main()
