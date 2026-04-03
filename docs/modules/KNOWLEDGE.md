---
description: "KnowledgeMemory for storing and retrieving structured knowledge entries"
tags:
  - memory
  - knowledge
---

# Knowledge Module

**Added in:** v0.16.0 (enhanced in v0.17.4)
**File:** `src/selectools/knowledge.py`, `knowledge_store_redis.py`, `knowledge_store_supabase.py`
**Classes:** `KnowledgeMemory`, `KnowledgeEntry`, `KnowledgeStore`, `FileKnowledgeStore`, `SQLiteKnowledgeStore`, `RedisKnowledgeStore`, `SupabaseKnowledgeStore`

!!! tip "v0.17.4 Enhancements"
    Knowledge Memory now supports pluggable store backends (File, SQLite, Redis, Supabase),
    importance scoring (0.0–1.0), TTL per entry, category filtering, and importance-based eviction.
    The original file-based API is fully backward compatible.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [KnowledgeMemory Class](#knowledgememory-class)
5. [The remember() Method](#the-remember-method)
6. [Context Building](#context-building)
7. [Auto-Registered Tool](#auto-registered-tool)
8. [Agent Integration](#agent-integration)
9. [Log Pruning](#log-pruning)
10. [Best Practices](#best-practices)

---

## Overview

The **Knowledge** module provides cross-session, long-term memory for selectools agents. Unlike [Entity Memory](ENTITY_MEMORY.md) (which tracks entities within a conversation) or [Knowledge Graph](KNOWLEDGE_GRAPH.md) (which tracks relationships), Knowledge Memory is a simple, durable store where agents (and users) can explicitly save and recall facts, preferences, and instructions that persist indefinitely.

### Purpose

- **Long-Term Memory**: Facts that survive across sessions, restarts, and deployments
- **Daily Logs**: Time-stamped memory entries for recent context
- **Persistent MEMORY.md**: A durable file of important facts flagged as persistent
- **Auto-Registered Tool**: The agent can call `remember()` to save knowledge during conversations
- **Category Organization**: Memories are tagged with categories for structured recall

### When to Use Each Memory Type

| Memory Type | Scope | Lifetime | Use Case |
|---|---|---|---|
| `ConversationMemory` | Single session | Until cleared | Multi-turn dialogue context |
| `EntityMemory` | Entities mentioned | Session / persisted | "Who is Alice?" |
| `KnowledgeGraphMemory` | Relationships | Session / persisted | "How are X and Y related?" |
| **`KnowledgeMemory`** | **Explicit facts** | **Indefinite** | **"Remember that I prefer dark mode"** |

---

## Quick Start

```python
from selectools import Agent, AgentConfig, OpenAIProvider, ConversationMemory, Message, Role
from selectools.knowledge import KnowledgeMemory

knowledge = KnowledgeMemory(storage_dir="./agent_memory")

agent = Agent(
    tools=[],
    provider=OpenAIProvider(),
    memory=ConversationMemory(max_messages=50),
    config=AgentConfig(knowledge_memory=knowledge),
)

# The agent can now use the auto-registered "remember" tool
result = agent.run([
    Message(role=Role.USER, content="Remember that my preferred language is Python and I work at Acme Corp.")
])

# Later (even after restart):
knowledge2 = KnowledgeMemory(storage_dir="./agent_memory")
agent2 = Agent(
    tools=[],
    provider=OpenAIProvider(),
    memory=ConversationMemory(),
    config=AgentConfig(knowledge_memory=knowledge2),
)

result = agent2.run([
    Message(role=Role.USER, content="What programming language do I prefer?")
])
# Agent knows: Python (loaded from persistent memory)
```

---

## Architecture

KnowledgeMemory uses a two-tier storage model:

```
./agent_memory/
+-- MEMORY.md              # persistent facts (survives pruning)
+-- logs/
    +-- 2026-03-13.jsonl   # daily log entries
    +-- 2026-03-12.jsonl
    +-- 2026-03-11.jsonl
    +-- ...
```

### MEMORY.md (Long-Term)

A Markdown file containing facts explicitly flagged as persistent. These are always loaded and never pruned.

```markdown
# Agent Memory

## preferences
- Preferred language: Python
- Dark mode: enabled

## personal
- Works at Acme Corp
- Name: Alice

## technical
- Uses PostgreSQL for production databases
- Prefers pytest over unittest
```

### Daily Logs (Recent)

JSONL files containing timestamped memory entries. These provide recent context and are subject to pruning.

```json
{"timestamp": "2026-03-13T10:15:00Z", "category": "preferences", "content": "Preferred language: Python", "persistent": true}
{"timestamp": "2026-03-13T10:16:00Z", "category": "context", "content": "Working on Project Atlas this week", "persistent": false}
{"timestamp": "2026-03-13T14:30:00Z", "category": "technical", "content": "Uses PostgreSQL for production databases", "persistent": true}
```

---

## KnowledgeMemory Class

### Constructor

```python
class KnowledgeMemory:
    def __init__(
        self,
        storage_dir: str = "./agent_memory",
        max_log_days: int = 30,
        max_context_entries: int = 50,
    ):
        """
        Args:
            storage_dir: Directory for MEMORY.md and daily logs.
            max_log_days: Days to retain daily log files before pruning.
            max_context_entries: Max recent entries to include in context.
        """
```

### Core Methods

```python
def remember(
    self,
    content: str,
    category: str = "general",
    persistent: bool = False,
) -> str:
    """Store a knowledge entry.

    Args:
        content: The fact or information to remember.
        category: Organizational category (e.g., "preferences", "personal", "technical").
        persistent: If True, also write to MEMORY.md for indefinite retention.

    Returns:
        Confirmation message.
    """

def build_context(self) -> str:
    """Build context string for system prompt injection.

    Combines persistent facts from MEMORY.md with recent daily log entries.
    Returns a formatted block with [Long-term Memory] and [Recent Memory] sections.
    """

def get_persistent_facts(self) -> Dict[str, List[str]]:
    """Return all persistent facts, organized by category."""

def get_recent_entries(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent log entries from the last N days."""

def prune_logs(self, max_days: Optional[int] = None) -> int:
    """Delete daily log files older than max_days.

    Returns the number of log files deleted.
    """

def clear(self) -> None:
    """Remove all knowledge (MEMORY.md and all logs)."""

def clear_logs(self) -> None:
    """Remove daily logs only, preserving MEMORY.md."""
```

---

## The remember() Method

`remember()` is the primary interface for storing knowledge:

```python
knowledge = KnowledgeMemory(storage_dir="./agent_memory")

# Transient memory (daily log only)
knowledge.remember(
    content="Currently debugging a timeout issue in the API",
    category="context",
    persistent=False,
)

# Persistent memory (daily log + MEMORY.md)
knowledge.remember(
    content="Preferred editor: VS Code",
    category="preferences",
    persistent=True,
)
```

### Behavior

| `persistent` | Daily Log | MEMORY.md | Survives Pruning |
|---|---|---|---|
| `False` | Written | Not written | No (deleted after `max_log_days`) |
| `True` | Written | Appended | Yes (MEMORY.md is never pruned) |

### Categories

Categories organize memories in MEMORY.md under Markdown headers:

```python
knowledge.remember("Name: Alice", category="personal", persistent=True)
knowledge.remember("Prefers Python", category="preferences", persistent=True)
knowledge.remember("Uses VS Code", category="preferences", persistent=True)
```

Produces in MEMORY.md:

```markdown
# Agent Memory

## personal
- Name: Alice

## preferences
- Prefers Python
- Uses VS Code
```

---

## Context Building

`build_context()` assembles a prompt-ready context block from both storage tiers:

```python
context = knowledge.build_context()
```

Output:

```
[Long-term Memory]
## preferences
- Preferred language: Python
- Dark mode: enabled

## personal
- Works at Acme Corp
- Name: Alice

[Recent Memory]
- [2026-03-13 10:16] (context) Working on Project Atlas this week
- [2026-03-13 14:30] (technical) Investigating timeout in payment service
- [2026-03-13 15:00] (context) Meeting with Bob about Atlas milestone
```

### Section Details

| Section | Source | Content |
|---|---|---|
| `[Long-term Memory]` | `MEMORY.md` | All persistent facts, organized by category |
| `[Recent Memory]` | Daily log files | Last N entries (up to `max_context_entries`) |

If either section is empty, it is omitted from the output.

---

## Auto-Registered Tool

When `knowledge_memory` is set in `AgentConfig`, a `remember` tool is automatically registered on the agent. This allows the LLM to save knowledge during conversations without any additional configuration.

### Tool Definition

```python
@tool(description="Save important information to long-term memory for future reference")
def remember(content: str, category: str = "general", persistent: bool = False) -> str:
    """Remember a fact or piece of information.

    Args:
        content: The information to remember.
        category: Category for organization (e.g., "preferences", "personal", "technical").
        persistent: Whether this should be stored permanently.

    Returns:
        Confirmation message.
    """
    return knowledge_memory.remember(content, category, persistent)
```

### Usage in Conversation

```
User: "Remember that I prefer dark mode and my timezone is PST."

Agent calls: remember(
    content="User prefers dark mode",
    category="preferences",
    persistent=True,
)

Agent calls: remember(
    content="User timezone is PST",
    category="preferences",
    persistent=True,
)

Agent: "I've saved your preferences. I'll remember that you prefer dark mode
and your timezone is PST."
```

The agent decides when to call `remember()` based on the conversation context. Explicit requests like "remember that..." and "save this..." reliably trigger the tool.

---

## Agent Integration

### Configuration

```python
from selectools import Agent, AgentConfig, OpenAIProvider, ConversationMemory
from selectools.knowledge import KnowledgeMemory

knowledge = KnowledgeMemory(
    storage_dir="./agent_memory",
    max_log_days=30,
    max_context_entries=50,
)

agent = Agent(
    tools=[...],  # your tools -- "remember" is added automatically
    provider=OpenAIProvider(),
    memory=ConversationMemory(max_messages=50),
    config=AgentConfig(knowledge_memory=knowledge),
)
```

### Integration Flow

```
Agent.__init__()
    |
    +-- Register "remember" tool automatically
    |
    +-- knowledge_memory.build_context()
    |   +-- Load MEMORY.md and recent logs
    |   +-- Build [Long-term Memory] + [Recent Memory] block
    |
    +-- Inject context into system prompt

run() / arun() called
    |
    +-- System prompt includes knowledge context
    |
    +-- Execute agent loop
    |   +-- LLM may call remember() tool
    |   +-- Tool writes to daily log (+ MEMORY.md if persistent)
    |
    +-- Return AgentResult
```

### Combining with Session Storage

```python
from selectools.sessions import SQLiteSessionStore
from selectools.knowledge import KnowledgeMemory

agent = Agent(
    tools=[...],
    provider=OpenAIProvider(),
    memory=ConversationMemory(),
    config=AgentConfig(
        knowledge_memory=KnowledgeMemory(storage_dir="./memory"),
        session_store=SQLiteSessionStore(db_path="sessions.db"),
        session_id="user-42",
    ),
)
# Session storage handles conversation memory.
# Knowledge memory handles long-term facts (separate storage).
```

### Combining with Entity and Knowledge Graph Memory

```python
agent = Agent(
    tools=[...],
    provider=OpenAIProvider(),
    memory=ConversationMemory(),
    config=AgentConfig(
        entity_memory=EntityMemory(max_entities=100, provider=OpenAIProvider()),
        knowledge_graph=KnowledgeGraphMemory(
            store=SQLiteTripleStore(db_path="kg.db"),
            provider=OpenAIProvider(),
        ),
        knowledge_memory=KnowledgeMemory(storage_dir="./memory"),
    ),
)
# System prompt includes:
# [Known Entities] -- from entity memory
# [Known Relationships] -- from knowledge graph
# [Long-term Memory] -- from knowledge memory
# [Recent Memory] -- from knowledge memory
```

---

## Log Pruning

Daily log files older than `max_log_days` are pruned automatically or on demand:

```python
knowledge = KnowledgeMemory(
    storage_dir="./agent_memory",
    max_log_days=30,  # auto-prune logs older than 30 days
)

# Manual pruning
deleted = knowledge.prune_logs()
print(f"Pruned {deleted} old log files")

# Override max_days for a one-time cleanup
deleted = knowledge.prune_logs(max_days=7)
```

### Pruning Behavior

- Only daily log files (`.jsonl`) are deleted
- `MEMORY.md` is never pruned (persistent facts are permanent)
- Pruning runs at the start of `build_context()` if stale logs exist
- Returns the count of deleted files

### Storage Growth

```
Typical daily log: ~1-10 KB per day
30 days retention: ~30-300 KB total
MEMORY.md: ~1-50 KB (depends on usage)
```

---

## Best Practices

### 1. Choose Appropriate Retention

```python
# Short-lived assistant (customer support)
knowledge = KnowledgeMemory(max_log_days=7)

# Long-running personal assistant
knowledge = KnowledgeMemory(max_log_days=90)
```

### 2. Use Categories Consistently

```python
# Establish category conventions
knowledge.remember("Name: Alice", category="personal", persistent=True)
knowledge.remember("Prefers dark mode", category="preferences", persistent=True)
knowledge.remember("Uses PostgreSQL", category="technical", persistent=True)
knowledge.remember("Meeting at 3pm", category="context", persistent=False)
```

### 3. Flag Important Facts as Persistent

```python
# Transient -- will be pruned
knowledge.remember("Working on bug #1234 today", category="context")

# Persistent -- survives indefinitely
knowledge.remember("API key rotation policy: every 90 days", category="technical", persistent=True)
```

### 4. Inspect Stored Knowledge

```python
# View persistent facts
facts = knowledge.get_persistent_facts()
for category, entries in facts.items():
    print(f"\n{category}:")
    for entry in entries:
        print(f"  - {entry}")

# View recent entries
recent = knowledge.get_recent_entries(days=3, limit=20)
for entry in recent:
    print(f"[{entry['timestamp']}] ({entry['category']}) {entry['content']}")
```

### 5. Separate Storage Per User

```python
def create_agent_for_user(user_id: str) -> Agent:
    return Agent(
        tools=[...],
        provider=OpenAIProvider(),
        memory=ConversationMemory(),
        config=AgentConfig(
            knowledge_memory=KnowledgeMemory(
                storage_dir=f"./memory/{user_id}",
            ),
        ),
    )
```

---

## Testing

```python
import tempfile
import os

def test_remember_and_recall():
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KnowledgeMemory(storage_dir=tmpdir)

        km.remember("Prefers Python", category="preferences", persistent=True)
        km.remember("Meeting at 3pm", category="context", persistent=False)

        context = km.build_context()
        assert "[Long-term Memory]" in context
        assert "Prefers Python" in context
        assert "[Recent Memory]" in context
        assert "Meeting at 3pm" in context


def test_persistent_facts_survive_clear_logs():
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KnowledgeMemory(storage_dir=tmpdir)

        km.remember("Important fact", category="general", persistent=True)
        km.remember("Transient note", category="context", persistent=False)

        km.clear_logs()

        facts = km.get_persistent_facts()
        assert "Important fact" in facts.get("general", [])

        recent = km.get_recent_entries()
        assert len(recent) == 0


def test_memory_md_categories():
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KnowledgeMemory(storage_dir=tmpdir)

        km.remember("Name: Alice", category="personal", persistent=True)
        km.remember("Likes Python", category="preferences", persistent=True)
        km.remember("Uses VS Code", category="preferences", persistent=True)

        facts = km.get_persistent_facts()
        assert len(facts["personal"]) == 1
        assert len(facts["preferences"]) == 2


def test_log_pruning():
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KnowledgeMemory(storage_dir=tmpdir, max_log_days=0)

        km.remember("Old note", category="context")

        deleted = km.prune_logs()
        assert deleted >= 0  # may be 0 if same-day


def test_remember_tool_registration():
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KnowledgeMemory(storage_dir=tmpdir)

        agent = Agent(
            tools=[],
            provider=LocalProvider(),
            memory=ConversationMemory(),
            config=AgentConfig(knowledge_memory=km),
        )

        tool_names = [t.name for t in agent.tools]
        assert "remember" in tool_names
```

---

## API Reference

| Class | Description |
|---|---|
| `KnowledgeMemory(storage_dir, max_log_days, max_context_entries)` | Cross-session knowledge store with daily logs and persistent MEMORY.md |

| Method | Returns | Description |
|---|---|---|
| `remember(content, category, persistent)` | `str` | Store a knowledge entry |
| `build_context()` | `str` | Build `[Long-term Memory]` + `[Recent Memory]` context |
| `get_persistent_facts()` | `Dict[str, List[str]]` | All MEMORY.md facts by category |
| `get_recent_entries(days, limit)` | `List[Dict]` | Recent daily log entries |
| `prune_logs(max_days)` | `int` | Delete old log files, return count |
| `clear()` | `None` | Remove all knowledge |
| `clear_logs()` | `None` | Remove logs only, keep MEMORY.md |

| AgentConfig Field | Type | Description |
|---|---|---|
| `knowledge_memory` | `Optional[KnowledgeMemory]` | Knowledge memory instance; auto-registers `remember` tool |

---

## Further Reading

- [Memory Module](MEMORY.md) - Conversation memory (in-session)
- [Entity Memory Module](ENTITY_MEMORY.md) - Entity attribute tracking
- [Knowledge Graph Module](KNOWLEDGE_GRAPH.md) - Relationship tracking
- [Sessions Module](SESSIONS.md) - Session persistence for conversation state
- [Agent Module](AGENT.md) - How agents use knowledge context

---

**Next Steps:** See how all memory types work together in the [Architecture doc](../ARCHITECTURE.md).
