---
description: "UnifiedMemory — tiered memory with auto-promotion across short-term, long-term, entity, and episodic tiers"
tags:
  - memory
  - knowledge
---

# Unified Memory Module

**Import:** `from selectools import UnifiedMemory`
**Stability:** beta

`UnifiedMemory` orchestrates the existing memory systems into one lifecycle:
`ConversationMemory` (short-term), `KnowledgeMemory` (long-term),
`EntityMemory` (entities, optional), and a new `EpisodicMemory`
(date-keyed history). Items flow between tiers automatically based on
importance scoring.

```text
add_turn() ──> Short-term (ConversationMemory, rolling window)
    │                │ ages out (auto_promote) / consolidate()
    │                ▼  importance >= threshold
    │          Long-term (KnowledgeMemory)
    ├────────> Episodic (date-keyed, retention-pruned)
    └────────> Entity (EntityMemory, when provided)
```

```python title="unified_memory_quick.py"
from selectools import UnifiedMemory

# Zero-arg default builds in-memory tiers — no API key, no filesystem state.
memory = UnifiedMemory(
    importance_threshold=0.7,
    short_term_limit=100,       # rolling window, in messages (1 turn = 2)
    long_term_limit=1000,       # importance-based eviction above this
    episodic_retention_days=30, # older episodes are pruned
    auto_promote=True,          # promote items as they age out of STM
)

memory.add_turn("My name is John", "Nice to meet you, John!")
memory.add_turn("I prefer dark roast coffee", "Noted.")

context = memory.assemble_context(max_tokens=4000)
results = memory.recall("user's coffee preference")
```

## Tier lifecycle

1. **`add_turn(user, assistant)`** appends both messages to short-term
   memory, records an episode (and prunes anything past the retention
   window), and — when an `EntityMemory` is configured — runs entity
   extraction on the turn.
2. **Promotion.** When a message ages out of the short-term rolling window
   (with `auto_promote=True`) or when `consolidate()` is called, the message
   text is scored. Scores at or above `importance_threshold` are written to
   long-term memory via `KnowledgeMemory.remember(...)` with the matched
   rule name as the entry category. Promotion is idempotent: a SHA-256
   content hash prevents the same text from being promoted twice.
3. **`consolidate()`** scores everything still in short-term memory and
   promotes eligible items immediately — useful at session end or with
   `auto_promote=False` for full manual control. Returns the number of
   items promoted.

## Importance scoring

Rule-based by default — no LLM required. Each rule maps a case-insensitive
regex to a score; the highest matching score wins, and unmatched text gets
the base score `0.3` (below the default threshold, so unremarkable turns
stay short-term).

| Rule | Score | Rationale |
|------|-------|-----------|
| `identity` | 0.9 | "my name is", "call me" — names are near-permanent facts |
| `relationship` | 0.85 | "my wife/daughter/boss…" — stable, high recall value |
| `preference` | 0.75 | "I prefer/like/love/hate", "my favorite" — durable but may evolve |
| `goal` | 0.7 | "my goal", "I decided/plan to" — important while active |
| `location` | 0.6 | "I live in", "based in", "I work at" — changes over time |
| `date_fact` | 0.6 | "birthday", "anniversary", "deadline" — follow-up anchors |

Override the table with `importance_rules=[ImportanceRule(...)]` (replaces
the defaults), or plug in an LLM-based scorer:

```python
def llm_scorer(text: str) -> float:
    ...  # call your provider; return 0.0-1.0

memory = UnifiedMemory(scorer=llm_scorer)
```

The `scorer` overrides the rule *score* (rules still supply the promotion
category); scorer failures fall back to the rule table.

## Context compaction

`assemble_context(max_tokens=...)` joins four sections: long-term knowledge,
known entities, recent episodes, and the short-term conversation. When the
result exceeds **70%** of `max_tokens` (configurable via
`compaction_threshold`), older short-term content is compacted:

- **With `summarizer=`** (any `str -> str` callable, e.g. an LLM call): the
  older message segment is replaced by `[Earlier conversation summary]`
  plus the summary.
- **Without one**: the segment is replaced by a
  `[... N earlier messages compacted ...]` marker (truncation mode).

The number of recent messages kept verbatim is progressively halved (never
below the latest turn). If the budget is still exceeded, the episodic and
then the entity sections are dropped, and as a last resort the output is
hard-truncated with a `[... context truncated ...]` marker.

## Federated recall

`recall(query, limit=10, days=None)` searches long-term entries, entities,
and episodes (date-filtered to the last `days` days, defaulting to the
retention window) and merges with a documented score rule:

| Source | Score | Max |
|--------|-------|-----|
| `long_term` | `importance * (0.5 + 0.5 * overlap)` | 1.0 |
| `entity` | `0.55 + 0.3 * overlap` | 0.85 |
| `episodic` | `0.35 + 0.2 * overlap` | 0.55 |

`overlap` is the fraction of query terms (>2 chars, lowercased) found in
the item. Zero-overlap items are excluded; results sort by score descending,
ties newest-first. Distilled long-term knowledge outranks raw history by
design.

## Dependency injection

Constructor accepts pre-built sub-memories so you pick the backends:

```python
from selectools import (
    ConversationMemory, EntityMemory, EpisodicMemory,
    KnowledgeMemory, SQLiteKnowledgeStore, UnifiedMemory,
)

memory = UnifiedMemory(
    short_term=ConversationMemory(max_messages=50),
    long_term=KnowledgeMemory(directory="./memory", store=SQLiteKnowledgeStore("kb.db")),
    entity_memory=EntityMemory(provider=provider),  # enables the entity tier
    episodic=EpisodicMemory(retention_days=90),
)
```

Without injection, the long-term tier uses the new `InMemoryKnowledgeStore`
(a dict-backed `KnowledgeStore`) with a temp scratch directory for the
legacy daily-log files. The entity tier is disabled unless an
`EntityMemory` is provided, since extraction needs an LLM provider.

`EpisodicMemory` is JSON-serializable via `to_dict()` / `from_dict()`.

## Thread safety

All `UnifiedMemory` and `EpisodicMemory` operations are protected by an
`RLock`, matching the `ConversationMemory` convention.

!!! tip "See Also"
    - [Memory](MEMORY.md) — the short-term tier
    - [Knowledge Memory](KNOWLEDGE.md) — the long-term tier
    - [Entity Memory](ENTITY_MEMORY.md) — the entity tier
    - Example: [`examples/106_unified_memory.py`](https://github.com/johnnichev/selectools/blob/main/examples/106_unified_memory.py)
