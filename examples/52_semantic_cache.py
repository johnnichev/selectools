#!/usr/bin/env python3
"""
Semantic Cache — serve LLM responses for similar (not just identical) queries.

Demonstrates:
- SemanticCache as a drop-in replacement for InMemoryCache
- Configuring similarity_threshold to control match sensitivity
- Cache hits for paraphrased questions
- LRU eviction when max_size is reached
- CacheStats tracking hits, misses, and evictions

How it works:
  SemanticCache embeds each cache key using an EmbeddingProvider and compares
  incoming queries via cosine similarity.  A hit is returned when the best
  match exceeds similarity_threshold — even if the wording differs.

Prerequisites:
    pip install selectools
    # A real embedding provider (e.g. OpenAI) is needed for live use.
    # This example uses a mock provider for demonstration.
"""

from __future__ import annotations

import math
from typing import List

from selectools.cache_semantic import SemanticCache
from selectools.embeddings.provider import EmbeddingProvider

# ---------------------------------------------------------------------------
# Mock embedding provider (avoids real API calls in this demo)
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(EmbeddingProvider):
    """Returns deterministic unit-vector embeddings for demo purposes."""

    # Pre-defined 4-d unit vectors for a small vocabulary
    _VOCAB = {
        "weather nyc": [1.0, 0.0, 0.0, 0.0],
        "weather new york": [0.98, 0.2, 0.0, 0.0],  # very similar to above
        "capital france": [0.0, 1.0, 0.0, 0.0],
        "paris france": [0.0, 0.97, 0.25, 0.0],  # similar to above
        "recipe pasta": [0.0, 0.0, 1.0, 0.0],  # unrelated to weather/capitals
    }

    @property
    def dimension(self) -> int:
        return 4

    def _vec(self, text: str) -> List[float]:
        key = text.lower().strip()
        vec = self._VOCAB.get(key, [0.5, 0.5, 0.5, 0.5])
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    def embed_text(self, text: str) -> List[float]:
        return self._vec(text)

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _separator(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print("=" * 55)


def main() -> None:
    print("=== Semantic Cache Demo ===")

    ep = MockEmbeddingProvider()

    # ------------------------------------------------------------------ #
    # 1. Basic usage: exact and near-miss hits
    # ------------------------------------------------------------------ #
    _separator("1. Basic hit / miss")

    cache = SemanticCache(
        embedding_provider=ep,
        similarity_threshold=0.92,  # require > 92 % cosine similarity
        max_size=100,
        default_ttl=None,  # no expiry
    )

    # Populate the cache manually (in real use the agent does this)
    CACHED_RESPONSE = ("The weather in NYC is sunny, 22 °C.", None)
    cache.set("weather nyc", CACHED_RESPONSE)

    # Exact match
    hit = cache.get("weather nyc")
    print(f"  Exact query → {'HIT' if hit else 'MISS'}: {hit}")

    # Paraphrase — should still hit (cosine ≈ 0.98)
    hit2 = cache.get("weather new york")
    print(f"  Paraphrase  → {'HIT' if hit2 else 'MISS'}: {hit2}")

    # Unrelated query — should miss
    miss = cache.get("recipe pasta")
    print(f"  Unrelated   → {'HIT' if miss else 'MISS'}")

    print(f"\n  Stats: {cache.stats}")

    # ------------------------------------------------------------------ #
    # 2. TTL expiry
    # ------------------------------------------------------------------ #
    _separator("2. TTL expiry")

    import time

    cache_ttl = SemanticCache(embedding_provider=ep, similarity_threshold=0.9)
    cache_ttl.set("capital france", ("Paris", None), ttl=1)  # expires in 1 s

    before = cache_ttl.get("capital france")
    print(f"  Before TTL expiry: {'HIT' if before else 'MISS'}")

    time.sleep(1.1)
    after = cache_ttl.get("capital france")
    print(f"  After TTL expiry:  {'HIT' if after else 'MISS'} (expected MISS)")

    # ------------------------------------------------------------------ #
    # 3. LRU eviction
    # ------------------------------------------------------------------ #
    _separator("3. LRU eviction (max_size=2)")

    small_cache = SemanticCache(embedding_provider=ep, similarity_threshold=0.9, max_size=2)
    small_cache.set("weather nyc", ("sunny", None))
    small_cache.set("capital france", ("Paris", None))
    # Third insert evicts the LRU entry (weather nyc)
    small_cache.set("recipe pasta", ("carbonara", None))

    print(f"  Size after 3 inserts into max_size=2 cache: {small_cache.size}")
    evicted = small_cache.get("weather nyc")
    print(f"  First entry after eviction: {'HIT' if evicted else 'MISS (evicted)'}")
    still_there = small_cache.get("recipe pasta")
    print(f"  Last entry still present:   {'HIT' if still_there else 'MISS'}")
    print(f"  Stats: {small_cache.stats}")

    # ------------------------------------------------------------------ #
    # 4. delete() and clear()
    # ------------------------------------------------------------------ #
    _separator("4. delete() and clear()")

    d_cache = SemanticCache(embedding_provider=ep, similarity_threshold=0.9)
    d_cache.set("weather nyc", ("sunny", None))
    d_cache.set("capital france", ("Paris", None))
    print(f"  Size before delete: {d_cache.size}")

    d_cache.delete("weather nyc")
    print(f"  Size after delete('weather nyc'): {d_cache.size}")

    d_cache.clear()
    print(f"  Size after clear(): {d_cache.size}")
    print(f"  Stats reset:        {d_cache.stats}")

    # ------------------------------------------------------------------ #
    # 5. Drop-in for AgentConfig.cache
    # ------------------------------------------------------------------ #
    _separator("5. Usage with Agent")

    print(
        """
  from selectools import Agent, AgentConfig
  from selectools.cache_semantic import SemanticCache
  from selectools.embeddings.openai import OpenAIEmbeddingProvider

  cache = SemanticCache(
      embedding_provider=OpenAIEmbeddingProvider(),
      similarity_threshold=0.92,
      max_size=500,
      default_ttl=3600,   # 1-hour TTL
  )

  agent = Agent(
      tools=[...],
      config=AgentConfig(cache=cache),
  )

  # First call — LLM is invoked, response cached
  r1 = agent.run("What's the weather in NYC?")

  # Second call with paraphrase — served from cache, no LLM call
  r2 = agent.run("Weather in New York City?")
  cache_hit = any(s.type.value == "cache_hit" for s in r2.trace.steps)
  print(f"Cache hit: {cache_hit}")   # True
    """
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
