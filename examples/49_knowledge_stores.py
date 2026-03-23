#!/usr/bin/env python3
"""
Knowledge Memory Stores — persistent knowledge with importance and TTL.

Demonstrates:
- KnowledgeMemory with SQLiteKnowledgeStore backend
- Importance scoring (0.0-1.0)
- TTL-based expiry
- Importance-based eviction at max_entries

Prerequisites:
    pip install selectools
"""

import os
import tempfile

from selectools.knowledge import KnowledgeEntry, KnowledgeMemory, SQLiteKnowledgeStore


def main() -> None:
    print("=" * 70)
    print("  Knowledge Memory Stores Demo")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="selectools_knowledge_stores_")
    db_path = os.path.join(tmpdir, "knowledge.db")

    # --- Demo 1: SQLiteKnowledgeStore basics ---
    print("\n--- Demo 1: SQLiteKnowledgeStore basics ---\n")

    store = SQLiteKnowledgeStore(db_path=db_path)

    entries = [
        KnowledgeEntry(
            content="User prefers dark mode",
            category="preference",
            importance=0.8,
        ),
        KnowledgeEntry(
            content="Project deadline is March 30",
            category="fact",
            importance=1.0,
            persistent=True,
        ),
        KnowledgeEntry(
            content="Discussed Python 3.13 migration",
            category="context",
            importance=0.3,
        ),
        KnowledgeEntry(
            content="API key rotated on Monday",
            category="context",
            importance=0.2,
            ttl_days=7,
        ),
    ]

    for entry in entries:
        entry_id = store.save(entry)
        print(f"  Saved: {entry.content[:40]:40s}  imp={entry.importance}  id={entry_id[:8]}")

    print(f"\n  Total entries: {store.count()}")

    # --- Demo 2: Query with filters ---
    print("\n--- Demo 2: Query with filters ---\n")

    all_entries = store.query(limit=10)
    print(f"  All entries (sorted by importance):")
    for e in all_entries:
        print(f"    imp={e.importance:.1f}  cat={e.category:12s}  {e.content[:45]}")

    high_importance = store.query(min_importance=0.7)
    print(f"\n  High importance (>= 0.7): {len(high_importance)} entries")
    for e in high_importance:
        print(f"    imp={e.importance:.1f}  {e.content[:50]}")

    prefs = store.query(category="preference")
    print(f"\n  Preferences: {len(prefs)} entries")
    for e in prefs:
        print(f"    {e.content}")

    # --- Demo 3: KnowledgeMemory with store and eviction ---
    print("\n--- Demo 3: KnowledgeMemory with max_entries eviction ---\n")

    db_path2 = os.path.join(tmpdir, "knowledge2.db")
    store2 = SQLiteKnowledgeStore(db_path=db_path2)

    km = KnowledgeMemory(
        directory=tmpdir,
        store=store2,
        max_entries=5,
        max_context_chars=3000,
    )

    # Add entries with varying importance
    facts = [
        ("CEO's name is Jane Doe", "fact", 1.0),
        ("Office is in San Francisco", "fact", 0.9),
        ("Prefers concise responses", "preference", 0.8),
        ("Meeting at 3 PM today", "schedule", 0.4),
        ("Weather was nice yesterday", "context", 0.1),
    ]

    for content, category, importance in facts:
        km.remember(content, category=category, importance=importance)
        print(f"  Stored: imp={importance:.1f}  {content}")

    print(f"\n  Entries: {store2.count()} / max_entries=5")

    # Adding a 6th entry triggers eviction of lowest-importance non-persistent
    print("\n  Adding one more entry (triggers eviction)...")
    km.remember("New project codename: Phoenix", category="fact", importance=0.7)

    print(f"  Entries after eviction: {store2.count()}")
    remaining = store2.query(limit=10)
    print(f"  Remaining entries:")
    for e in remaining:
        print(f"    imp={e.importance:.1f}  {e.content[:50]}")

    # --- Demo 4: TTL-based pruning ---
    print("\n--- Demo 4: TTL-based pruning ---\n")

    db_path3 = os.path.join(tmpdir, "knowledge3.db")
    store3 = SQLiteKnowledgeStore(db_path=db_path3)

    store3.save(
        KnowledgeEntry(
            content="Temporary note: server IP 10.0.0.1",
            category="context",
            importance=0.5,
            ttl_days=0,  # Expires immediately
        )
    )
    store3.save(
        KnowledgeEntry(
            content="Permanent: always use HTTPS",
            category="instruction",
            importance=0.9,
            persistent=True,
        )
    )
    store3.save(
        KnowledgeEntry(
            content="Low priority trivia",
            category="context",
            importance=0.1,
        )
    )

    print(f"  Before prune: {store3.count()} entries")

    removed = store3.prune(max_age_days=0, min_importance=0.3)
    print(f"  Pruned: {removed} entries (expired TTL + low importance)")
    print(f"  After prune: {store3.count()} entries")

    remaining3 = store3.query(limit=10)
    for e in remaining3:
        flag = "[persistent]" if e.persistent else ""
        print(f"    imp={e.importance:.1f}  {e.content[:45]}  {flag}")

    # --- Demo 5: Context block for prompt injection ---
    print("\n--- Demo 5: Context block for system prompt ---\n")

    context = km.build_context()
    print(f"  Context block ({len(context)} chars):")
    for line in context.split("\n")[:8]:
        if line.strip():
            print(f"    {line}")

    # Cleanup
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("  Key takeaways:")
    print("    - SQLiteKnowledgeStore: durable, queryable, thread-safe")
    print("    - importance scoring (0.0-1.0) controls eviction priority")
    print("    - persistent=True protects entries from eviction")
    print("    - ttl_days auto-expires temporary knowledge")
    print("    - prune() cleans expired + low-importance entries")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
