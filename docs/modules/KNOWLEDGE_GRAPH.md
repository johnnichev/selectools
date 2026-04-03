---
description: "Knowledge graph memory with triple extraction and relationship storage"
tags:
  - memory
  - knowledge-graph
---

# Knowledge Graph Module

**Import:** `from selectools.knowledge_graph import KnowledgeGraphMemory`
**Stability:** beta

```python title="knowledge_graph_quick.py"
from selectools.knowledge_graph import KnowledgeGraphMemory, InMemoryTripleStore, Triple

# Build a knowledge graph without an LLM provider (no API key needed)
store = InMemoryTripleStore()
kg = KnowledgeGraphMemory(store=store, max_context_triples=20)

kg.update([
    Triple(subject="Alice", relation="works_at", object="Acme Corp", confidence=0.95),
    Triple(subject="Acme Corp", relation="located_in", object="Seattle", confidence=0.90),
    Triple(subject="Alice", relation="manages", object="Project Atlas", confidence=0.85),
])

# Query by subject
for t in store.query(subject="Alice"):
    print(f"  {t.subject} --{t.relation}--> {t.object}")

# Build context for system prompt injection
context = kg.build_context()
print(context)
# [Known Relationships]
# - Alice works_at Acme Corp (0.95)
# - Acme Corp located_in Seattle (0.90)
# - Alice manages Project Atlas (0.85)
```

!!! tip "See Also"
    - [Entity Memory](ENTITY_MEMORY.md) - Entity attribute tracking (complements the graph)
    - [Knowledge](KNOWLEDGE.md) - Cross-session long-term knowledge memory

---

**Added in:** v0.16.0
**File:** `src/selectools/knowledge_graph.py`
**Classes:** `Triple`, `TripleStore`, `InMemoryTripleStore`, `SQLiteTripleStore`, `KnowledgeGraphMemory`

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Triple Dataclass](#triple-dataclass)
4. [TripleStore Protocol](#triplestore-protocol)
5. [Store Implementations](#store-implementations)
6. [KnowledgeGraphMemory](#knowledgegraphmemory)
7. [LLM-Powered Extraction](#llm-powered-extraction)
8. [Agent Integration](#agent-integration)
9. [Observer Events](#observer-events)
10. [Querying the Graph](#querying-the-graph)
11. [Best Practices](#best-practices)

---

## Overview

The **Knowledge Graph** module builds a graph of relationships between entities extracted from conversations. While [Entity Memory](ENTITY_MEMORY.md) tracks individual entities and their attributes, the Knowledge Graph tracks how entities relate to each other -- forming a structured, queryable web of knowledge.

### Purpose

- **Relationship Tracking**: Capture subject-relation-object triples from conversation
- **LLM Extraction**: Automatically extract relationships using an LLM provider
- **Keyword Query**: Retrieve relevant triples by keyword or entity name
- **Context Injection**: Feed relationship context into the system prompt
- **Persistence**: Store triples in memory or SQLite for durability

### How It Differs from Entity Memory

| Feature | Entity Memory | Knowledge Graph |
|---|---|---|
| **Tracks** | Individual entities + attributes | Relationships between entities |
| **Structure** | Key-value (entity -> attributes) | Graph (subject -> relation -> object) |
| **Example** | Alice: {role: engineer} | Alice --works_at--> Acme Corp |
| **Query** | By entity name | By keyword, subject, or object |
| **Best for** | "What do I know about X?" | "How are X and Y related?" |

---

## Quick Start

```python
from selectools import Agent, AgentConfig, OpenAIProvider, ConversationMemory, Message, Role
from selectools.knowledge_graph import KnowledgeGraphMemory, InMemoryTripleStore

kg = KnowledgeGraphMemory(
    store=InMemoryTripleStore(),
    provider=OpenAIProvider(),  # used for LLM-based extraction
)

agent = Agent(
    tools=[],
    provider=OpenAIProvider(),
    memory=ConversationMemory(max_messages=50),
    config=AgentConfig(knowledge_graph=kg),
)

# Turn 1 -- relationships extracted automatically
result = agent.run([
    Message(role=Role.USER, content="Alice works at Acme Corp. Acme Corp is based in Seattle.")
])

# Turn 2 -- agent has relationship context
result = agent.run([
    Message(role=Role.USER, content="Where does Alice's company operate?")
])
# Agent knows: Alice works_at Acme Corp, Acme Corp located_in Seattle
```

---

## Triple Dataclass

Each relationship is represented as a `Triple`:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Triple:
    subject: str                       # source entity
    relation: str                      # relationship type (e.g., "works_at")
    object: str                        # target entity
    confidence: float = 1.0            # extraction confidence (0.0 - 1.0)
    source_turn: Optional[int] = None  # conversation turn where extracted
    created_at: Optional[datetime] = None
```

### Example Triples

```python
Triple(subject="Alice", relation="works_at", object="Acme Corp", confidence=0.95)
Triple(subject="Acme Corp", relation="located_in", object="Seattle", confidence=0.90)
Triple(subject="Alice", relation="manages", object="Project Atlas", confidence=0.85)
Triple(subject="Bob", relation="reports_to", object="Alice", confidence=0.80)
```

---

## TripleStore Protocol

All backends implement the `TripleStore` protocol:

```python
from typing import Protocol, List, Optional

class TripleStore(Protocol):
    def add(self, triples: List[Triple]) -> None:
        """Add triples to the store. Duplicates are ignored."""
        ...

    def query(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Triple]:
        """Query triples by any combination of subject, relation, object.
        None fields act as wildcards.
        """
        ...

    def search(self, keywords: List[str], top_k: int = 20) -> List[Triple]:
        """Search triples matching any of the given keywords.
        Matches against subject, relation, and object fields.
        """
        ...

    def delete(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        object: Optional[str] = None,
    ) -> int:
        """Delete matching triples. Returns the number of triples deleted."""
        ...

    def all(self) -> List[Triple]:
        """Return all triples in the store."""
        ...

    def clear(self) -> None:
        """Remove all triples."""
        ...

    def count(self) -> int:
        """Return the total number of triples."""
        ...
```

---

## Store Implementations

### 1. InMemoryTripleStore

**Best for:** Prototyping, testing, short-lived sessions

```python
from selectools.knowledge_graph import InMemoryTripleStore

store = InMemoryTripleStore()

store.add([
    Triple(subject="Alice", relation="works_at", object="Acme Corp"),
    Triple(subject="Acme Corp", relation="located_in", object="Seattle"),
])

# Query by subject
results = store.query(subject="Alice")
# [Triple(subject="Alice", relation="works_at", object="Acme Corp")]

# Keyword search
results = store.search(keywords=["Alice", "Seattle"], top_k=10)
# Returns triples mentioning Alice or Seattle
```

**Features:**

- No dependencies
- Fast in-memory lookup
- No persistence (lost on restart)
- Suitable for up to ~10k triples

### 2. SQLiteTripleStore

**Best for:** Production single-instance, persistent knowledge graphs

```python
from selectools.knowledge_graph import SQLiteTripleStore

store = SQLiteTripleStore(db_path="knowledge_graph.db")
```

**Schema:**

```sql
CREATE TABLE triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source_turn INTEGER,
    created_at TEXT,
    UNIQUE(subject, relation, object)
);

CREATE INDEX idx_subject ON triples(subject);
CREATE INDEX idx_object ON triples(object);
CREATE INDEX idx_relation ON triples(relation);
```

**Features:**

- Persistent storage
- Indexed queries
- Duplicate-safe (UNIQUE constraint)
- ACID transactions
- Suitable for up to ~100k triples

### Choosing a Store

| Feature | InMemory | SQLite |
|---|---|---|
| **Persistence** | No | Yes |
| **Dependencies** | None | None |
| **Max Triples** | ~10k | ~100k |
| **Query Speed** | Fast | Fast (indexed) |
| **Setup** | None | DB path |

---

## KnowledgeGraphMemory

`KnowledgeGraphMemory` wraps a `TripleStore` with LLM-powered extraction and context building:

### Constructor

```python
class KnowledgeGraphMemory:
    def __init__(
        self,
        store: TripleStore,
        provider: Optional[Provider] = None,
        extraction_model: Optional[str] = None,
        max_context_triples: int = 30,
        min_confidence: float = 0.5,
    ):
        """
        Args:
            store: Backend triple store.
            provider: LLM provider for relationship extraction.
                      If None, extraction is disabled (manual-only).
            extraction_model: Override model for extraction calls.
            max_context_triples: Max triples to include in context injection.
            min_confidence: Minimum confidence threshold for context inclusion.
        """
```

### Core Methods

```python
def extract_triples(self, text: str) -> List[Triple]:
    """Extract relationship triples from text using the LLM provider.

    Returns a list of Triple objects parsed from the LLM response.
    """

def update(self, triples: List[Triple]) -> None:
    """Add triples to the underlying store."""

def query(self, keywords: List[str], top_k: int = 20) -> List[Triple]:
    """Search the triple store by keywords.

    Filters results by min_confidence threshold.
    """

def build_context(self, keywords: Optional[List[str]] = None) -> str:
    """Build a context string for system prompt injection.

    If keywords are provided, only relevant triples are included.
    Otherwise, the most recent triples (up to max_context_triples) are used.
    """

def clear(self) -> None:
    """Clear all triples from the store."""

def to_dict(self) -> Dict[str, Any]:
    """Serialize for persistence (used by session storage)."""

@classmethod
def from_dict(cls, data: Dict[str, Any], store: TripleStore) -> "KnowledgeGraphMemory":
    """Restore from serialized data."""
```

---

## LLM-Powered Extraction

When a provider is configured, `extract_triples()` sends text to the LLM with a structured prompt:

```
Extract all relationships from the following text as subject-relation-object triples.

For each triple, provide:
- subject: the source entity
- relation: the relationship (use snake_case, e.g., "works_at", "located_in", "manages")
- object: the target entity
- confidence: how confident you are (0.0 to 1.0)

Respond as a JSON array.

Text:
"""
Alice is a senior engineer at Acme Corp. She manages the Atlas project
and reports to Bob, the VP of Engineering. Acme Corp is headquartered in Seattle.
"""
```

The LLM responds:

```json
[
    {"subject": "Alice", "relation": "works_at", "object": "Acme Corp", "confidence": 0.95},
    {"subject": "Alice", "relation": "has_role", "object": "senior engineer", "confidence": 0.95},
    {"subject": "Alice", "relation": "manages", "object": "Atlas project", "confidence": 0.90},
    {"subject": "Alice", "relation": "reports_to", "object": "Bob", "confidence": 0.90},
    {"subject": "Bob", "relation": "has_role", "object": "VP of Engineering", "confidence": 0.90},
    {"subject": "Acme Corp", "relation": "headquartered_in", "object": "Seattle", "confidence": 0.95}
]
```

---

## Agent Integration

### Configuration

```python
from selectools import Agent, AgentConfig, OpenAIProvider, ConversationMemory
from selectools.knowledge_graph import KnowledgeGraphMemory, SQLiteTripleStore

kg = KnowledgeGraphMemory(
    store=SQLiteTripleStore(db_path="kg.db"),
    provider=OpenAIProvider(model="gpt-4o-mini"),
    max_context_triples=30,
    min_confidence=0.6,
)

agent = Agent(
    tools=[...],
    provider=OpenAIProvider(),
    memory=ConversationMemory(max_messages=50),
    config=AgentConfig(knowledge_graph=kg),
)
```

### Context Injection Flow

```
run() / arun() called
    |
    +-- knowledge_graph.extract_triples(user_message)
    |   +-- LLM extracts relationship triples
    |
    +-- knowledge_graph.update(extracted_triples)
    |   +-- Store triples in backend
    |
    +-- Extract keywords from user message
    |
    +-- knowledge_graph.build_context(keywords)
    |   +-- "[Known Relationships]
    |   |    - Alice works_at Acme Corp (confidence: 0.95)
    |   |    - Acme Corp headquartered_in Seattle (confidence: 0.95)
    |   |    - Alice manages Atlas project (confidence: 0.90)"
    |
    +-- Prepend context to system message
    |
    +-- Execute agent loop
    |
    +-- Return AgentResult
```

### Context Format

The `build_context()` method produces:

```
[Known Relationships]
- Alice works_at Acme Corp (0.95)
- Alice manages Atlas project (0.90)
- Alice reports_to Bob (0.90)
- Acme Corp headquartered_in Seattle (0.95)
- Bob has_role VP of Engineering (0.90)
```

---

## Observer Events

Knowledge graph extraction fires an observer event:

```python
from selectools import AgentObserver

class KGWatcher(AgentObserver):
    def on_kg_extraction(
        self,
        run_id: str,
        triples_extracted: int,
        triples_total: int,
        triples: list,
    ) -> None:
        print(f"[{run_id}] Extracted {triples_extracted} triples, {triples_total} total in store")
        for t in triples:
            print(f"  {t.subject} --{t.relation}--> {t.object} ({t.confidence:.2f})")
```

| Event | When | Parameters |
|---|---|---|
| `on_kg_extraction` | After extracting and storing triples | `run_id`, `triples_extracted`, `triples_total`, `triples` |

---

## Querying the Graph

### By Subject

```python
# All relationships where Alice is the subject
triples = kg.store.query(subject="Alice")
# Alice works_at Acme Corp
# Alice manages Atlas project
# Alice reports_to Bob
```

### By Object

```python
# All relationships pointing to Acme Corp
triples = kg.store.query(object="Acme Corp")
# Alice works_at Acme Corp
```

### By Relation Type

```python
# All "manages" relationships
triples = kg.store.query(relation="manages")
# Alice manages Atlas project
```

### By Keywords

```python
# Free-text keyword search
triples = kg.query(keywords=["Alice", "engineering"], top_k=10)
# Returns triples mentioning Alice or engineering
```

### Combined Queries

```python
# Alice's role at Acme Corp specifically
triples = kg.store.query(subject="Alice", object="Acme Corp")
# Alice works_at Acme Corp
```

---

## Best Practices

### 1. Use SQLite for Persistent Graphs

```python
# Prototyping
kg = KnowledgeGraphMemory(store=InMemoryTripleStore(), provider=provider)

# Production
kg = KnowledgeGraphMemory(
    store=SQLiteTripleStore(db_path="knowledge.db"),
    provider=provider,
)
```

### 2. Filter by Confidence

```python
# Only high-confidence triples in context
kg = KnowledgeGraphMemory(
    store=store,
    provider=provider,
    min_confidence=0.8,  # ignore uncertain extractions
)
```

### 3. Use a Cost-Effective Extraction Model

```python
# Use a smaller model for extraction
kg = KnowledgeGraphMemory(
    store=store,
    provider=OpenAIProvider(model="gpt-4o-mini"),
)
```

### 4. Limit Context Size

```python
# Prevent context from growing too large
kg = KnowledgeGraphMemory(
    store=store,
    provider=provider,
    max_context_triples=20,  # cap at 20 triples in prompt
)
```

### 5. Combine with Entity Memory

```python
from selectools.entity_memory import EntityMemory
from selectools.knowledge_graph import KnowledgeGraphMemory, SQLiteTripleStore

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
    ),
)
# Agent gets both [Known Entities] and [Known Relationships] context
```

### 6. Seed Domain Knowledge

```python
from selectools.knowledge_graph import Triple

kg.update([
    Triple(subject="Python", relation="is_a", object="programming language", confidence=1.0),
    Triple(subject="selectools", relation="written_in", object="Python", confidence=1.0),
    Triple(subject="selectools", relation="supports", object="OpenAI", confidence=1.0),
    Triple(subject="selectools", relation="supports", object="Anthropic", confidence=1.0),
])
```

---

## Testing

```python
def test_triple_store_add_and_query():
    store = InMemoryTripleStore()

    store.add([
        Triple(subject="Alice", relation="works_at", object="Acme"),
        Triple(subject="Bob", relation="works_at", object="Acme"),
    ])

    results = store.query(subject="Alice")
    assert len(results) == 1
    assert results[0].object == "Acme"

    results = store.query(object="Acme")
    assert len(results) == 2


def test_triple_store_keyword_search():
    store = InMemoryTripleStore()

    store.add([
        Triple(subject="Alice", relation="works_at", object="Acme Corp"),
        Triple(subject="Bob", relation="lives_in", object="Seattle"),
    ])

    results = store.search(keywords=["Alice"], top_k=10)
    assert len(results) == 1
    assert results[0].subject == "Alice"


def test_duplicate_triples_ignored():
    store = InMemoryTripleStore()

    store.add([
        Triple(subject="A", relation="r", object="B"),
        Triple(subject="A", relation="r", object="B"),  # duplicate
    ])

    assert store.count() == 1


def test_build_context():
    store = InMemoryTripleStore()
    store.add([
        Triple(subject="Alice", relation="works_at", object="Acme", confidence=0.9),
    ])

    kg = KnowledgeGraphMemory(store=store, max_context_triples=10)
    context = kg.build_context()

    assert "[Known Relationships]" in context
    assert "Alice" in context
    assert "works_at" in context
    assert "Acme" in context


def test_confidence_filtering():
    store = InMemoryTripleStore()
    store.add([
        Triple(subject="A", relation="r1", object="B", confidence=0.9),
        Triple(subject="C", relation="r2", object="D", confidence=0.3),
    ])

    kg = KnowledgeGraphMemory(store=store, min_confidence=0.5)
    results = kg.query(keywords=["A", "C"], top_k=10)

    assert len(results) == 1
    assert results[0].subject == "A"
```

---

## API Reference

| Class | Description |
|---|---|
| `Triple(subject, relation, object, confidence)` | Dataclass for a subject-relation-object relationship |
| `TripleStore` | Protocol defining add/query/search/delete/clear interface |
| `InMemoryTripleStore()` | In-memory triple store for prototyping |
| `SQLiteTripleStore(db_path)` | SQLite-backed persistent triple store |
| `KnowledgeGraphMemory(store, provider, max_context_triples, min_confidence)` | LLM-powered knowledge graph with context injection |

| Method | Returns | Description |
|---|---|---|
| `extract_triples(text)` | `List[Triple]` | Extract triples from text via LLM |
| `update(triples)` | `None` | Add triples to the store |
| `query(keywords, top_k)` | `List[Triple]` | Search triples by keywords |
| `build_context(keywords)` | `str` | Build `[Known Relationships]` context string |
| `clear()` | `None` | Remove all triples |
| `to_dict()` | `Dict` | Serialize for persistence |
| `from_dict(data, store)` | `KnowledgeGraphMemory` | Restore from serialized data |

| AgentConfig Field | Type | Description |
|---|---|---|
| `knowledge_graph` | `Optional[KnowledgeGraphMemory]` | Knowledge graph instance for relationship tracking |

---

## Further Reading

- [Entity Memory Module](ENTITY_MEMORY.md) - Entity attribute tracking (complements the knowledge graph)
- [Memory Module](MEMORY.md) - Conversation memory
- [Sessions Module](SESSIONS.md) - Persist graph state across restarts
- [Knowledge Module](KNOWLEDGE.md) - Cross-session long-term knowledge

---

**Next Steps:** Learn about cross-session knowledge in the [Knowledge Module](KNOWLEDGE.md).

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 36 | [`36_knowledge_graph.py`](https://github.com/johnnichev/selectools/blob/main/examples/36_knowledge_graph.py) | Knowledge graph with triple extraction |
| 20 | [`20_customer_support_bot.py`](https://github.com/johnnichev/selectools/blob/main/examples/20_customer_support_bot.py) | Production bot with knowledge context |
| 35 | [`35_entity_memory.py`](https://github.com/johnnichev/selectools/blob/main/examples/35_entity_memory.py) | Entity memory (graph complement) |
| 37 | [`37_knowledge_memory.py`](https://github.com/johnnichev/selectools/blob/main/examples/37_knowledge_memory.py) | Long-term knowledge memory |
