---
description: "Embedding-based response caching for semantically similar queries"
tags:
  - caching
  - semantic
---

# Semantic Cache

**Added in:** v0.17.7
**File:** `src/selectools/cache_semantic.py`

## Overview

`SemanticCache` is a drop-in replacement for `InMemoryCache` that serves cached LLM responses for **semantically equivalent** queries — even when the exact wording differs.

Instead of exact-string key matching, it embeds each cache key using any [`EmbeddingProvider`](EMBEDDINGS.md) and compares incoming queries via **cosine similarity**.  A hit is returned when the best match exceeds a configurable `similarity_threshold`.

```python
from selectools.cache_semantic import SemanticCache
from selectools.embeddings.openai import OpenAIEmbeddingProvider

cache = SemanticCache(
    embedding_provider=OpenAIEmbeddingProvider(),
    similarity_threshold=0.92,
    max_size=500,
    default_ttl=3600,
)
config = AgentConfig(cache=cache)

# "What's the weather in NYC?" hits cache for "Weather in New York City?"
```

## Quick Start

```python
from selectools import Agent, AgentConfig
from selectools.cache_semantic import SemanticCache
from selectools.embeddings.openai import OpenAIEmbeddingProvider

cache = SemanticCache(
    embedding_provider=OpenAIEmbeddingProvider(),
    similarity_threshold=0.92,
)

agent = Agent(
    tools=[...],
    config=AgentConfig(cache=cache),
)

r1 = agent.run("What's the weather in NYC?")
r2 = agent.run("Weather in New York City?")  # cache hit — no LLM call

cache_hit = any(s.type.value == "cache_hit" for s in r2.trace.steps)
print(cache_hit)  # True
print(cache.stats)  # CacheStats(hits=1, misses=1, evictions=0, hit_rate=50.00%)
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_provider` | `EmbeddingProvider` | — | Provides `embed_text()` and `embed_query()`. Required. |
| `similarity_threshold` | `float` | `0.92` | Minimum cosine similarity for a cache hit. Range: `[0.0, 1.0]`. |
| `max_size` | `int` | `1000` | Maximum entries before LRU eviction. `0` = unbounded. |
| `default_ttl` | `Optional[int]` | `None` | Default TTL in seconds. `None` = no expiry. |

## Methods

### `get(key: str) → Optional[Tuple[Any, Any]]`

Embeds `key` and scans stored entries for the best cosine similarity match.

- Returns the cached value if best score ≥ `similarity_threshold` and entry has not expired.
- Moves the matched entry to the end of the LRU list on hit.
- Returns `None` on miss.

### `set(key: str, value: Tuple[Any, Any], ttl: Optional[int] = None) → None`

Embeds `key` and stores the entry.

- Replaces an existing entry if the exact `key` already exists.
- Evicts the LRU entry when `max_size` is reached.
- TTL overrides `default_ttl` if provided.

### `delete(key: str) → bool`

Removes an entry by **exact** original key.  Returns `True` if found and removed.

### `clear() → None`

Removes all entries and resets `CacheStats`.

### `stats → CacheStats`

Read-only snapshot of hit/miss/eviction counters and `hit_rate`.

### `size → int`

Number of entries currently stored (includes expired entries not yet pruned).

## How Similarity Works

Cosine similarity is computed in pure Python (no NumPy):

```
similarity(a, b) = dot(a, b) / (‖a‖ · ‖b‖)
```

Embeddings are normalised to unit vectors by most providers, so this reduces to a dot product.  **No external dependencies are required** beyond the embedding provider itself.

## Threshold Guide

| Threshold | Behaviour |
|-----------|-----------|
| `0.99–1.0` | Near-exact matches only (minor typo tolerance) |
| `0.92–0.98` | Paraphrases and synonyms (recommended for general use) |
| `0.80–0.92` | Loose topic-level similarity |
| `< 0.80` | Very permissive; may cause false hits |

## TTL and Expiry

```python
cache = SemanticCache(
    embedding_provider=ep,
    default_ttl=3600,  # 1-hour default TTL
)

# Override TTL per entry
cache.set("weather nyc", response, ttl=300)  # 5-minute TTL for this entry
```

Expired entries are **not eagerly pruned** — they are skipped during `get()` scans.  They are evicted naturally when max_size is reached or `clear()` is called.

## LRU Eviction

When `size == max_size`, the least-recently-used entry (front of the list) is evicted on the next `set()`.  Accessing an entry via `get()` moves it to the end (most-recently-used position).

## Thread Safety

All public methods acquire an internal `threading.Lock`.  `SemanticCache` is safe to share across threads.

## Trace Integration

When a cache hit occurs through the agent, a `CACHE_HIT` step is appended to `AgentTrace`:

```python
result = agent.run("What's the weather?")
for step in result.trace.steps:
    if step.type == StepType.CACHE_HIT:
        print(f"Cache hit: {step.summary}")
```

## Comparison with InMemoryCache

| Feature | `InMemoryCache` | `SemanticCache` |
|---------|-----------------|-----------------|
| Key matching | Exact string | Cosine similarity |
| Extra dependency | None | EmbeddingProvider |
| Hit on paraphrase | No | Yes |
| LRU eviction | Yes | Yes |
| TTL support | Yes | Yes |
| Thread-safe | Yes | Yes |

## Example

See [`examples/52_semantic_cache.py`](https://github.com/johnnichev/selectools/blob/main/examples/52_semantic_cache.py) for a runnable demo using a mock embedding provider.
