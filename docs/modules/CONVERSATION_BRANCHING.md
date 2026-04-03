---
description: "Fork conversation history for A/B exploration and checkpointing"
tags:
  - memory
  - branching
---

# Conversation Branching

**Added in:** v0.17.7
**Files:** `src/selectools/memory.py`, `src/selectools/sessions.py`

## Overview

Conversation branching lets you **fork a conversation at any point** to explore alternative paths without affecting the original.  Changes to a branch never propagate back to the source.

Two levels of branching are supported:

- **In-memory**: `ConversationMemory.branch()` — instant snapshot, no I/O
- **Persisted**: `SessionStore.branch(source_id, new_id)` — works with JSON, SQLite, and Redis backends

## Quick Start

```python
from selectools.memory import ConversationMemory
from selectools import Message, Role

# Build up some conversation history
memory = ConversationMemory(max_messages=50)
memory.add(Message(role=Role.USER, content="Let's discuss architecture options."))
memory.add(Message(role=Role.ASSISTANT, content="Sure! Option A is microservices…"))

# Fork at this point
branch = memory.branch()

# Explore a different direction in the branch
branch.add(Message(role=Role.USER, content="What about a monolith instead?"))

print(len(memory))  # 2 — unchanged
print(len(branch))  # 3 — branch has diverged
```

## `ConversationMemory.branch()`

Returns an **independent copy** of the current memory state.

```python
branch = memory.branch()
```

What is copied:

| Attribute | Copied? |
|-----------|---------|
| `_messages` | Yes — new list (mutations are independent) |
| `summary` | Yes — new string reference |
| `max_messages` | Yes |
| `max_tokens` | Yes |
| `_last_trimmed` | Reset to `[]` |

The branch and original share no mutable state — appending, clearing, or trimming one does not affect the other.

## `SessionStore.branch(source_id, new_id)`

All three session backends (`JsonFileSessionStore`, `SQLiteSessionStore`, `RedisSessionStore`) support branching persisted sessions:

```python
from selectools.sessions import JsonFileSessionStore

store = JsonFileSessionStore(directory="/tmp/sessions")

# Save the current state
store.save("main", agent.memory)

# Fork into a new session
store.branch("main", "experiment")

# Load and modify the branch independently
exp = store.load("experiment")
exp.add(Message(role=Role.USER, content="Risky idea…"))
store.save("experiment", exp)

# Original is untouched
original = store.load("main")
print(len(original))   # same as before
```

Raises `ValueError` if `source_id` does not exist.

## Use Cases

### Checkpoint before a risky sub-task

```python
checkpoint = agent.memory.branch()

result = agent.run("Delete all temporary files and rebuild the index")
if not result_looks_good(result):
    agent.memory = checkpoint  # roll back
    result = agent.run("Safer approach: archive first, then rebuild")
```

### A/B exploration

```python
store.branch("shared_context", "variant_a")
store.branch("shared_context", "variant_b")

agent_a = Agent(tools=[...], config=AgentConfig(session_id="variant_a", session_store=store))
agent_b = Agent(tools=[...], config=AgentConfig(session_id="variant_b", session_store=store))

result_a = agent_a.run("Prefer brevity over completeness")
result_b = agent_b.run("Be thorough and detailed")
```

### Parallel agent exploration

```python
import threading

branch_ids = []
for i in range(4):
    bid = f"branch_{i}"
    store.branch("root", bid)
    branch_ids.append(bid)

def explore(branch_id):
    agent = Agent(..., config=AgentConfig(session_id=branch_id, session_store=store))
    return agent.run(f"Explore strategy {branch_id}")

results = [None] * 4
threads = [threading.Thread(target=lambda i=i: results.__setitem__(i, explore(branch_ids[i]))) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Error Handling

`SessionStore.branch()` raises `ValueError` when the source session does not exist:

```python
try:
    store.branch("nonexistent", "dst")
except ValueError as exc:
    print(exc)  # "Session 'nonexistent' not found"
```

## Example

See [`examples/54_conversation_branching.py`](https://github.com/johnnichev/selectools/blob/main/examples/54_conversation_branching.py) for a runnable demo covering `ConversationMemory.branch()`, `JsonFileSessionStore.branch()`, and `SQLiteSessionStore.branch()`.
