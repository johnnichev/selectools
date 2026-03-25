#!/usr/bin/env python3
"""
Conversation Branching — fork conversation history for A/B exploration.

Demonstrates:
- ConversationMemory.branch() — snapshot memory for safe experimentation
- JsonFileSessionStore.branch(src, dst) — fork persisted sessions
- SQLiteSessionStore.branch(src, dst) — fork persisted sessions
- Independence: changes to a branch never affect the original
- Raises ValueError when source session is not found

Use cases:
- Try two different follow-up prompts from the same conversation state
- Checkpoint a conversation before entering a risky sub-task
- Parallelize agent explorations from a shared starting point

Prerequisites:
    pip install selectools
    # No API keys needed — all demo code is local.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from selectools import Message, Role
from selectools.memory import ConversationMemory
from selectools.sessions import JsonFileSessionStore, SQLiteSessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(n: int, max_messages: int = 20) -> ConversationMemory:
    """Return a ConversationMemory pre-loaded with n user messages."""
    mem = ConversationMemory(max_messages=max_messages)
    for i in range(n):
        mem.add(Message(role=Role.USER, content=f"Turn {i}: tell me about topic {i}."))
        mem.add(Message(role=Role.ASSISTANT, content=f"Topic {i}: here is what I know…"))
    return mem


def _separator(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_memory_branch() -> None:
    _separator("1. ConversationMemory.branch()")

    mem = _make_memory(3)
    print(f"  Original memory: {len(mem)} messages")

    branch = mem.branch()
    print(f"  Branch (snapshot): {len(branch)} messages — identical to original")

    # Modify branch — original must be unaffected
    branch.add(Message(role=Role.USER, content="Branch-only question."))
    branch.add(Message(role=Role.ASSISTANT, content="Branch-only answer."))

    print(f"\n  After adding 2 messages to branch:")
    print(f"    Branch:   {len(branch)} messages")
    print(f"    Original: {len(mem)} messages  ← unchanged")

    # Summary field is also copied independently
    mem.summary = "Context: user is learning Python."
    branch2 = mem.branch()
    branch2.summary = "Context: user switched to Rust."
    print(f"\n  Summary independence:")
    print(f"    Original: '{mem.summary}'")
    print(f"    Branch2:  '{branch2.summary}'")

    # Config is preserved
    mem3 = ConversationMemory(max_messages=10, max_tokens=2000)
    b3 = mem3.branch()
    print(f"\n  Config preserved: max_messages={b3.max_messages}, max_tokens={b3.max_tokens}")

    # Internal list is a new object
    assert branch._messages is not mem._messages
    print("\n  branch._messages is a new list object — deep independence confirmed.")


def demo_json_session_branch() -> None:
    _separator("2. JsonFileSessionStore.branch()")

    with tempfile.TemporaryDirectory() as tmp:
        store = JsonFileSessionStore(directory=tmp)

        mem = _make_memory(4)
        store.save("main", mem)
        print(f"  Saved 'main' session: {len(mem)} messages")

        # Fork 'main' into 'explore'
        store.branch("main", "explore")
        print(f"  Branched 'main' → 'explore'")

        # Modify 'explore' independently
        explore = store.load("explore")
        assert explore is not None
        explore.add(Message(role=Role.USER, content="Exploring a risky idea…"))
        store.save("explore", explore)

        # Reload both
        main_reloaded = store.load("main")
        explore_reloaded = store.load("explore")
        assert main_reloaded is not None
        assert explore_reloaded is not None

        print(f"\n  After modifying 'explore':")
        print(f"    'main'    session: {len(main_reloaded)} messages  ← unchanged")
        print(f"    'explore' session: {len(explore_reloaded)} messages")

        # Error on missing source
        try:
            store.branch("nonexistent", "ghost")
        except ValueError as exc:
            print(f"\n  ValueError for missing source: {exc}")

        sessions = store.list()
        print(f"\n  Sessions in store: {sorted(s.session_id for s in sessions)}")


def demo_sqlite_session_branch() -> None:
    _separator("3. SQLiteSessionStore.branch()")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "sessions.db")
        store = SQLiteSessionStore(db_path=db_path)

        mem = _make_memory(6)
        store.save("experiment_a", mem)
        print(f"  Saved 'experiment_a': {len(mem)} messages")

        store.branch("experiment_a", "experiment_b")
        print("  Branched 'experiment_a' → 'experiment_b'")

        b = store.load("experiment_b")
        assert b is not None
        b.clear()
        store.save("experiment_b", b)

        a_reloaded = store.load("experiment_a")
        b_reloaded = store.load("experiment_b")
        assert a_reloaded is not None
        assert b_reloaded is not None

        print(f"\n  After clearing 'experiment_b':")
        print(f"    'experiment_a': {len(a_reloaded)} messages  ← unchanged")
        print(f"    'experiment_b': {len(b_reloaded)} messages")

        # Error on missing source
        try:
            store.branch("does_not_exist", "dst")
        except ValueError as exc:
            print(f"\n  ValueError for missing source: {exc}")


def demo_usage_pattern() -> None:
    _separator("4. Idiomatic usage pattern")

    print(
        """
  # Checkpoint before risky sub-task
  mem = agent.memory
  checkpoint = mem.branch()

  result = agent.run("Try the dangerous approach")
  if is_bad_result(result):
      agent.memory = checkpoint          # restore safely
      result = agent.run("Safer approach")


  # A/B exploration from shared state
  store.branch("main_conversation", "variant_a")
  store.branch("main_conversation", "variant_b")

  agent_a = Agent(..., config=AgentConfig(session_id="variant_a", ...))
  agent_b = Agent(..., config=AgentConfig(session_id="variant_b", ...))
  # Both start from the same history but diverge independently
    """
    )


def main() -> None:
    print("=== Conversation Branching Demo ===")
    demo_memory_branch()
    demo_json_session_branch()
    demo_sqlite_session_branch()
    demo_usage_pattern()
    print("\nDone.")


if __name__ == "__main__":
    main()
