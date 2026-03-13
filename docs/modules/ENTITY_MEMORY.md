# Entity Memory Module

**Added in:** v0.16.0
**File:** `src/selectools/entity_memory.py`
**Classes:** `Entity`, `EntityMemory`

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Entity Dataclass](#entity-dataclass)
4. [EntityMemory Class](#entitymemory-class)
5. [LLM-Powered Extraction](#llm-powered-extraction)
6. [Deduplication and Merging](#deduplication-and-merging)
7. [LRU Pruning](#lru-pruning)
8. [Agent Integration](#agent-integration)
9. [Observer Events](#observer-events)
10. [Best Practices](#best-practices)

---

## Overview

The **Entity Memory** module automatically extracts, tracks, and recalls named entities (people, organizations, locations, concepts) across conversation turns. It gives agents persistent awareness of who and what has been discussed, enabling more coherent multi-turn interactions.

### Purpose

- **Entity Extraction**: LLM-powered identification of entities from conversation text
- **Attribute Tracking**: Accumulate facts about entities across turns (e.g., "Alice works at Acme Corp")
- **Mention Counting**: Track how frequently each entity appears
- **Context Injection**: Automatically provide the agent with known entity context
- **LRU Pruning**: Evict least-recently-used entities when capacity is exceeded

---

## Quick Start

```python
from selectools import Agent, AgentConfig, OpenAIProvider, ConversationMemory, Message, Role
from selectools.entity_memory import EntityMemory

entity_memory = EntityMemory(
    max_entities=100,
    provider=OpenAIProvider(),  # used for LLM-based extraction
)

agent = Agent(
    tools=[],
    provider=OpenAIProvider(),
    memory=ConversationMemory(max_messages=50),
    config=AgentConfig(entity_memory=entity_memory),
)

# Turn 1 -- entities extracted automatically
result = agent.run([
    Message(role=Role.USER, content="Alice is a software engineer at Acme Corp in Seattle.")
])

# Turn 2 -- agent has entity context
result = agent.run([
    Message(role=Role.USER, content="What do you know about Alice?")
])
# Agent knows: Alice is a software engineer at Acme Corp, located in Seattle
```

---

## Entity Dataclass

Each tracked entity is represented as an `Entity` instance:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class Entity:
    name: str                                      # canonical name
    entity_type: str                               # "person", "organization", "location", etc.
    attributes: Dict[str, str] = field(default_factory=dict)
    mentions: int = 0                              # total mention count
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    aliases: List[str] = field(default_factory=list)  # alternative names
```

### Example Entity

```python
Entity(
    name="Alice",
    entity_type="person",
    attributes={
        "role": "software engineer",
        "company": "Acme Corp",
        "location": "Seattle",
    },
    mentions=3,
    first_seen=datetime(2026, 3, 13, 10, 0),
    last_seen=datetime(2026, 3, 13, 10, 15),
    aliases=["alice", "Alice Smith"],
)
```

---

## EntityMemory Class

### Constructor

```python
class EntityMemory:
    def __init__(
        self,
        max_entities: int = 100,
        provider: Optional[Provider] = None,
        extraction_model: Optional[str] = None,
    ):
        """
        Args:
            max_entities: Maximum entities to track. LRU eviction when exceeded.
            provider: LLM provider used for entity extraction. If None,
                      extraction is skipped and entities must be added manually.
            extraction_model: Override model for extraction calls.
                              Defaults to the provider's configured model.
        """
```

### Core Methods

```python
def extract_entities(self, text: str) -> List[Entity]:
    """Extract entities from text using the LLM provider.

    Sends a structured extraction prompt to the LLM and parses
    the response into Entity objects. Returns newly extracted entities.
    """

def update(self, entities: List[Entity]) -> None:
    """Merge extracted entities into the tracked set.

    - New entities are added.
    - Existing entities have their attributes merged and mention counts incremented.
    - LRU eviction is triggered if max_entities is exceeded.
    """

def build_context(self) -> str:
    """Build a context string for injection into the system prompt.

    Returns a formatted block listing all tracked entities with
    their types and attributes, suitable for prepending to messages.
    """

def get_entity(self, name: str) -> Optional[Entity]:
    """Look up a tracked entity by name (case-insensitive)."""

def get_all_entities(self) -> List[Entity]:
    """Return all tracked entities, ordered by last_seen (most recent first)."""

def clear(self) -> None:
    """Remove all tracked entities."""

def to_dict(self) -> Dict[str, Any]:
    """Serialize entity memory for persistence."""

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "EntityMemory":
    """Restore entity memory from serialized data."""
```

---

## LLM-Powered Extraction

When a provider is configured, `extract_entities()` sends the conversation text to the LLM with a structured extraction prompt:

```
Extract all named entities from the following text.
For each entity, provide:
- name: the canonical name
- entity_type: one of "person", "organization", "location", "product", "concept", "event", "other"
- attributes: key-value pairs of facts mentioned about the entity

Respond as a JSON array.

Text:
"""
Alice is a software engineer at Acme Corp in Seattle. She is working on Project Atlas.
"""
```

The LLM responds with structured JSON:

```json
[
    {"name": "Alice", "entity_type": "person", "attributes": {"role": "software engineer", "company": "Acme Corp"}},
    {"name": "Acme Corp", "entity_type": "organization", "attributes": {"location": "Seattle"}},
    {"name": "Seattle", "entity_type": "location", "attributes": {}},
    {"name": "Project Atlas", "entity_type": "product", "attributes": {"team_member": "Alice"}}
]
```

### Without a Provider

If no provider is given, automatic extraction is disabled. You can still manage entities manually:

```python
from selectools.entity_memory import EntityMemory, Entity

em = EntityMemory(max_entities=50)  # no provider

# Manual entity management
em.update([
    Entity(name="Alice", entity_type="person", attributes={"role": "engineer"}),
])

context = em.build_context()
```

---

## Deduplication and Merging

When `update()` encounters an entity whose name matches an existing tracked entity (case-insensitive), it merges rather than duplicates:

```python
# Turn 1: "Alice is an engineer"
em.update([Entity(name="Alice", entity_type="person", attributes={"role": "engineer"})])

# Turn 2: "Alice lives in Seattle and goes by Ali"
em.update([Entity(
    name="Alice",
    entity_type="person",
    attributes={"location": "Seattle"},
    aliases=["Ali"],
)])

# Result: single entity with merged attributes
alice = em.get_entity("Alice")
# alice.attributes == {"role": "engineer", "location": "Seattle"}
# alice.mentions == 2
# alice.aliases == ["Ali"]
```

### Merge Rules

| Field | Merge Strategy |
|---|---|
| `name` | Keep existing canonical name |
| `entity_type` | Keep existing (first wins) |
| `attributes` | Merge dicts; new values overwrite old for same key |
| `mentions` | Increment by 1 |
| `aliases` | Union of both alias lists |
| `last_seen` | Update to current time |

---

## LRU Pruning

When the number of tracked entities exceeds `max_entities`, the least-recently-used entities are evicted:

```python
em = EntityMemory(max_entities=3)

em.update([Entity(name="A", entity_type="person")])  # [A]
em.update([Entity(name="B", entity_type="person")])  # [A, B]
em.update([Entity(name="C", entity_type="person")])  # [A, B, C]

# Capacity full -- next update evicts LRU
em.update([Entity(name="D", entity_type="person")])  # [B, C, D]  -- A evicted
```

An entity's `last_seen` timestamp is updated on every mention, so frequently-discussed entities remain in memory.

---

## Agent Integration

### Configuration

```python
from selectools import Agent, AgentConfig, OpenAIProvider, ConversationMemory
from selectools.entity_memory import EntityMemory

entity_memory = EntityMemory(
    max_entities=200,
    provider=OpenAIProvider(),
)

agent = Agent(
    tools=[...],
    provider=OpenAIProvider(),
    memory=ConversationMemory(max_messages=50),
    config=AgentConfig(entity_memory=entity_memory),
)
```

### Context Injection Flow

When entity memory is configured, the agent automatically injects entity context into the system prompt:

```
run() / arun() called
    |
    +-- entity_memory.extract_entities(user_message)
    |   +-- LLM extracts entities from new messages
    |
    +-- entity_memory.update(extracted_entities)
    |   +-- Merge with existing entities, LRU prune
    |
    +-- entity_memory.build_context()
    |   +-- "[Known Entities]
    |   |    - Alice (person): role=software engineer, company=Acme Corp
    |   |    - Acme Corp (organization): location=Seattle
    |   |    - Seattle (location)"
    |
    +-- Prepend context to system message
    |
    +-- Execute agent loop (LLM sees entity context)
    |
    +-- Return AgentResult
```

### Context Format

The `build_context()` method produces a block like:

```
[Known Entities]
- Alice (person): role=software engineer, company=Acme Corp, location=Seattle
- Acme Corp (organization): location=Seattle, employee=Alice
- Project Atlas (product): team_member=Alice
```

This block is injected as part of the system message so the LLM can reference known entities without re-extraction.

---

## Observer Events

Entity extraction fires an observer event:

```python
from selectools import AgentObserver

class EntityWatcher(AgentObserver):
    def on_entity_extraction(
        self,
        run_id: str,
        entities_extracted: int,
        entities_total: int,
        entities: list,
    ) -> None:
        print(f"[{run_id}] Extracted {entities_extracted} entities, {entities_total} total tracked")
        for e in entities:
            print(f"  - {e.name} ({e.entity_type})")
```

| Event | When | Parameters |
|---|---|---|
| `on_entity_extraction` | After extracting and merging entities | `run_id`, `entities_extracted`, `entities_total`, `entities` |

---

## Best Practices

### 1. Set Appropriate Capacity

```python
# Short conversations -- fewer entities needed
em = EntityMemory(max_entities=50)

# Long-running assistants -- track more context
em = EntityMemory(max_entities=500)
```

### 2. Use a Cost-Effective Extraction Model

```python
# Use a smaller model for extraction to reduce cost
em = EntityMemory(
    max_entities=100,
    provider=OpenAIProvider(model="gpt-4o-mini"),
)
```

### 3. Persist Entity Memory with Sessions

Entity memory is serialized when used with session storage:

```python
from selectools.sessions import SQLiteSessionStore

store = SQLiteSessionStore(db_path="sessions.db")

agent = Agent(
    tools=[...],
    provider=OpenAIProvider(),
    memory=ConversationMemory(),
    config=AgentConfig(
        entity_memory=EntityMemory(max_entities=100, provider=OpenAIProvider()),
        session_store=store,
        session_id="user-42",
    ),
)
# Entity memory is saved/restored alongside conversation memory
```

### 4. Inspect Tracked Entities

```python
for entity in entity_memory.get_all_entities():
    print(f"{entity.name} ({entity.entity_type}): {entity.attributes}")
    print(f"  Mentions: {entity.mentions}, Last seen: {entity.last_seen}")
```

### 5. Manual Entity Seeding

Pre-populate entities for domain-specific contexts:

```python
em = EntityMemory(max_entities=100)

em.update([
    Entity(name="Selectools", entity_type="product", attributes={
        "type": "Python library",
        "purpose": "AI agent framework",
    }),
    Entity(name="OpenAI", entity_type="organization", attributes={
        "type": "AI company",
    }),
])
```

---

## Testing

```python
def test_entity_extraction_and_merge():
    em = EntityMemory(max_entities=50)

    em.update([
        Entity(name="Alice", entity_type="person", attributes={"role": "engineer"}),
    ])
    assert em.get_entity("Alice") is not None
    assert em.get_entity("Alice").mentions == 1

    # Merge new attributes
    em.update([
        Entity(name="Alice", entity_type="person", attributes={"location": "Seattle"}),
    ])
    alice = em.get_entity("Alice")
    assert alice.mentions == 2
    assert alice.attributes["role"] == "engineer"
    assert alice.attributes["location"] == "Seattle"


def test_lru_eviction():
    em = EntityMemory(max_entities=2)

    em.update([Entity(name="A", entity_type="person")])
    em.update([Entity(name="B", entity_type="person")])
    em.update([Entity(name="C", entity_type="person")])

    assert em.get_entity("A") is None  # evicted
    assert em.get_entity("B") is not None
    assert em.get_entity("C") is not None


def test_build_context():
    em = EntityMemory(max_entities=50)
    em.update([
        Entity(name="Alice", entity_type="person", attributes={"role": "engineer"}),
    ])

    context = em.build_context()
    assert "[Known Entities]" in context
    assert "Alice (person)" in context
    assert "role=engineer" in context


def test_serialization_roundtrip():
    em = EntityMemory(max_entities=50)
    em.update([
        Entity(name="Alice", entity_type="person", attributes={"role": "engineer"}),
    ])

    data = em.to_dict()
    em2 = EntityMemory.from_dict(data)

    assert em2.get_entity("Alice") is not None
    assert em2.get_entity("Alice").attributes["role"] == "engineer"
```

---

## API Reference

| Class | Description |
|---|---|
| `Entity(name, entity_type, attributes, mentions, aliases)` | Dataclass representing a tracked entity |
| `EntityMemory(max_entities, provider, extraction_model)` | LLM-powered entity tracker with LRU eviction |

| Method | Returns | Description |
|---|---|---|
| `extract_entities(text)` | `List[Entity]` | Extract entities from text via LLM |
| `update(entities)` | `None` | Merge entities into tracked set |
| `build_context()` | `str` | Build `[Known Entities]` context string |
| `get_entity(name)` | `Optional[Entity]` | Look up entity by name |
| `get_all_entities()` | `List[Entity]` | All tracked entities (most recent first) |
| `clear()` | `None` | Remove all entities |
| `to_dict()` | `Dict` | Serialize for persistence |
| `from_dict(data)` | `EntityMemory` | Restore from serialized data |

| AgentConfig Field | Type | Description |
|---|---|---|
| `entity_memory` | `Optional[EntityMemory]` | Entity memory instance for automatic extraction |

---

## Further Reading

- [Memory Module](MEMORY.md) - Conversation memory that entity memory extends
- [Sessions Module](SESSIONS.md) - Persist entity memory across restarts
- [Knowledge Graph Module](KNOWLEDGE_GRAPH.md) - Relationship tracking between entities
- [Agent Module](AGENT.md) - How agents use entity context

---

**Next Steps:** Learn about relationship tracking in the [Knowledge Graph Module](KNOWLEDGE_GRAPH.md).
