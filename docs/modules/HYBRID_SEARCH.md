# Hybrid Search Module

**Directory:** `src/selectools/rag/`
**Files:** `bm25.py`, `hybrid.py`, `tools.py`, `reranker.py`

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [BM25 Keyword Search](#bm25-keyword-search)
5. [HybridSearcher](#hybridsearcher)
6. [HybridSearchTool](#hybridsearchtool)
7. [Reranking](#reranking)
8. [Configuration Guide](#configuration-guide)
9. [RAGTool vs HybridSearchTool](#ragtool-vs-hybridsearchtool)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)
12. [Further Reading](#further-reading)

---

## Overview

**Hybrid search** combines two complementary retrieval strategies to improve recall and precision:

| Strategy | Captures | Strengths |
|----------|----------|-----------|
| **Vector (Semantic)** | Meaning, context, synonyms | Finds conceptually similar content even with different wording |
| **BM25 (Keyword)** | Exact terms, names, acronyms | Matches specific words and phrases precisely |

### Why Hybrid Search Matters

Semantic search alone can miss important matches:

```
Query: "GDPR compliance requirements"
Document: "European Union General Data Protection Regulation mandates..."
```

- **Vector search**: May find it (similar meaning) ✓
- **Keyword search**: "GDPR" exact match → high relevance ✓

Keyword search alone fails on paraphrasing:

```
Query: "How to get started quickly"
Document: "Quick start guide and setup instructions"
```

- **Keyword search**: No overlap ✗
- **Vector search**: High semantic similarity ✓

**Hybrid search** runs both and fuses results, catching cases that either approach might miss.

### Import Paths

```python
from selectools.rag import (
    BM25,
    HybridSearcher,
    FusionMethod,
    HybridSearchTool,
    Reranker,
    CohereReranker,
    JinaReranker,
)
```

---

## Architecture

### Flow Diagram

```
                                    ┌─────────────────────────────────────────┐
                                    │              USER QUERY                   │
                                    │         "GDPR compliance requirements"   │
                                    └──────────────────┬───────────────────────┘
                                                       │
                          ┌────────────────────────────┼────────────────────────────┐
                          │                            │                            │
                          ▼                            │                            ▼
              ┌───────────────────────┐                │                ┌───────────────────────┐
              │   Vector Store        │                │                │   BM25 Index          │
              │   (Semantic Search)   │                │                │   (Keyword Search)    │
              │   • embed_query()     │                │                │   • tokenize()        │
              │   • cosine similarity │                │                │   • Okapi BM25 score  │
              └───────────┬───────────┘                │                └───────────┬───────────┘
                          │                            │                            │
                          │   vector_top_k             │             keyword_top_k   │
                          ▼                            │                            ▼
              ┌───────────────────────┐                │                ┌───────────────────────┐
              │ Vector Results        │                │                │ Keyword Results       │
              │ [SearchResult, ...]   │                │                │ [SearchResult, ...]   │
              └───────────┬───────────┘                │                └───────────┬───────────┘
                          │                            │                            │
                          └────────────────────────────┼────────────────────────────┘
                                                       │
                                                       ▼
                                    ┌─────────────────────────────────────────┐
                                    │           FUSION                        │
                                    │   RRF or Weighted Linear Combination    │
                                    │   • Deduplicate documents                │
                                    │   • Combine scores                       │
                                    └──────────────────┬──────────────────────┘
                                                       │
                                                       ▼
                                    ┌─────────────────────────────────────────┐
                                    │   RERANKER (Optional)                   │
                                    │   Cohere / Jina cross-encoder           │
                                    │   Re-score fused candidates             │
                                    └──────────────────┬──────────────────────┘
                                                       │
                                                       ▼
                                    ┌─────────────────────────────────────────┐
                                    │   FINAL RESULTS                         │
                                    │   [SearchResult, ...] sorted by score   │
                                    └─────────────────────────────────────────┘
```

---

## Quick Start

### Minimal Working Example

```python
from selectools.rag import Document, VectorStore, HybridSearcher, HybridSearchTool
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools import Agent, OpenAIProvider

# 1. Set up vector store with embeddings
embedder = OpenAIEmbeddingProvider()
store = VectorStore.create("memory", embedder=embedder)

# 2. Create hybrid searcher
searcher = HybridSearcher(vector_store=store)

# 3. Add documents (indexes in both vector store and BM25)
docs = [
    Document(text="GDPR requires data protection by design.", metadata={"source": "legal.md"}),
    Document(text="Python is a programming language.", metadata={"source": "intro.md"}),
    Document(text="European Union data regulation compliance guide.", metadata={"source": "eu.md"}),
]
searcher.add_documents(docs)

# 4. Search
results = searcher.search("GDPR compliance", top_k=3)
for r in results:
    print(f"Score: {r.score:.4f} | {r.document.text[:50]}...")

# 5. Use with agent (optional)
hybrid_tool = HybridSearchTool(searcher=searcher, top_k=5)
agent = Agent(tools=[hybrid_tool.search_knowledge_base], provider=OpenAIProvider())
response = agent.run("What does GDPR require?")
```

---

## BM25 Keyword Search

### Class: BM25

Pure-Python Okapi BM25 keyword search with zero external dependencies. Uses only the Python standard library.

### Constructor

```python
BM25(k1=1.5, b=0.75, remove_stopwords=True)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k1` | 1.5 | Term frequency saturation. Higher = more weight on repeated terms |
| `b` | 0.75 | Length normalisation. 0 = none, 1 = full normalisation |
| `remove_stopwords` | True | Filter English stop words (a, the, is, etc.) |

### Tokenization

- **Regex-based**: Splits on `[^a-z0-9]+` (non-alphanumeric)
- **Lowercase**: All text normalised to lowercase
- **Stop words**: Optional removal of common English words

```python
from selectools.rag import BM25, Document

bm25 = BM25(remove_stopwords=True)
tokens = bm25.tokenize("The quick brown fox jumps over the lazy dog")
# ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']

bm25_no_stop = BM25(remove_stopwords=False)
tokens = bm25_no_stop.tokenize("The quick brown fox")
# ['the', 'quick', 'brown', 'fox']
```

### Indexing

```python
from selectools.rag import BM25, Document

bm25 = BM25(k1=1.5, b=0.75)

# Build or rebuild index (replaces existing)
docs = [
    Document(text="Python programming language"),
    Document(text="Java programming language"),
    Document(text="Machine learning with Python"),
]
bm25.index_documents(docs)

# Incrementally add documents
bm25.add_documents([Document(text="Rust systems programming")])

print(bm25.document_count)  # 4
```

### Search

```python
# Returns List[SearchResult] (same format as VectorStore.search)
results = bm25.search("Python programming", top_k=2)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.document.text}")
    print(f"Metadata: {result.document.metadata}")

# With metadata filter
results = bm25.search(
    "programming",
    top_k=5,
    filter={"source": "intro.md"},
)
```

### Standalone BM25 Example

```python
from selectools.rag import BM25, Document

bm25 = BM25()
docs = [
    Document(text="Selectools is an AI agent framework"),
    Document(text="Framework for building agents with tools"),
    Document(text="AI and ML tool integration"),
]
bm25.index_documents(docs)

results = bm25.search("AI framework", top_k=2)
assert "Selectools" in results[0].document.text or "Framework" in results[0].document.text
```

---

## HybridSearcher

Combines vector (semantic) and BM25 (keyword) retrieval with score fusion.

### Constructor

```python
HybridSearcher(
    vector_store,           # VectorStore instance (required)
    bm25=None,              # Pre-built BM25 or None (creates internal BM25)
    vector_weight=0.5,      # Weight for semantic results
    keyword_weight=0.5,     # Weight for keyword results
    fusion="rrf",           # "rrf" or "weighted"
    rrf_k=60,               # RRF constant (fusion="rrf" only)
    reranker=None,          # Optional Reranker for post-fusion re-scoring
)
```

### Fusion Methods

```python
from selectools.rag import FusionMethod

# RRF - Reciprocal Rank Fusion (default)
# Rank-based, no score normalisation needed, robust
searcher = HybridSearcher(vector_store=store, fusion=FusionMethod.RRF)
# or: fusion="rrf"

# WEIGHTED - Weighted Linear Combination
# Min-max normalised scores, then weighted sum
searcher = HybridSearcher(vector_store=store, fusion=FusionMethod.WEIGHTED)
# or: fusion="weighted"
```

| Fusion | Formula | Use When |
|--------|---------|----------|
| **RRF** | `score = w_v/(k+rank_v) + w_k/(k+rank_k)` | Default; handles diverse score scales well |
| **WEIGHTED** | `score = w_v * norm(v) + w_k * norm(k)` | You want explicit score contribution control |

### Methods

#### add_documents

Add documents to both vector store and BM25 index:

```python
docs = DocumentLoader.from_directory("./docs")
ids = searcher.add_documents(docs)

# Optional: pre-computed embeddings
embeddings = embedder.embed_texts([d.text for d in docs])
ids = searcher.add_documents(docs, embeddings=embeddings)
```

#### index_existing_documents

Build BM25 index from documents already in the vector store (e.g. pre-populated store):

```python
# Vector store was filled before HybridSearcher was created
store.add_documents(existing_docs)

searcher = HybridSearcher(vector_store=store)
searcher.index_existing_documents(existing_docs)
```

#### search

```python
results = searcher.search(
    query="GDPR data protection",
    top_k=5,
    filter={"category": "legal"},
    vector_top_k=10,   # Candidates from vector search (default: top_k * 2)
    keyword_top_k=10,  # Candidates from BM25 (default: top_k * 2)
)
```

### Deduplication

Documents appearing in both vector and keyword result sets are automatically deduplicated before fusion. Matching is by `document.text` equality.

### Complete Example

```python
from selectools.rag import Document, VectorStore, HybridSearcher, FusionMethod
from selectools.embeddings import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider()
store = VectorStore.create("memory", embedder=embedder)

searcher = HybridSearcher(
    vector_store=store,
    vector_weight=0.6,
    keyword_weight=0.4,
    fusion=FusionMethod.RRF,
    rrf_k=60,
)

docs = [
    Document(text="GDPR Article 32: Security of processing", metadata={"source": "gdpr.pdf"}),
    Document(text="Data protection by design and default", metadata={"source": "gdpr.pdf"}),
]
searcher.add_documents(docs)

results = searcher.search("Article 32 security measures", top_k=3)
```

---

## HybridSearchTool

Pre-built `@tool`-decorated search for agent integration. Drop-in replacement for `RAGTool` with better recall for exact terms, names, and acronyms.

### Constructor

```python
HybridSearchTool(
    searcher,           # HybridSearcher instance (required)
    top_k=5,
    score_threshold=0.0,
    include_scores=True,
)
```

### Tool Method: search_knowledge_base

The tool the agent calls:

```python
from selectools.rag import HybridSearchTool
from selectools import Agent, OpenAIProvider

hybrid_tool = HybridSearchTool(searcher=searcher, top_k=5)
agent = Agent(
    tools=[hybrid_tool.search_knowledge_base],
    provider=OpenAIProvider(),
)
response = agent.run("What are the GDPR security requirements?")
```

### Programmatic Method: search

Direct search without going through the agent:

```python
results = hybrid_tool.search(
    query="installation steps",
    filter={"source": "README.md"},
)
# Returns List[SearchResult]
```

### Output Format

Same as RAGTool:

```
[Source 1: gdpr.pdf (page 5), Relevance: 0.8234]
GDPR Article 32 requires appropriate technical and organizational
measures to ensure a level of security...

[Source 2: compliance.md, Relevance: 0.7102]
Security of processing includes encryption and pseudonymization...
```

---

## Reranking

Rerankers use cross-encoder models to re-score candidates from initial retrieval, improving precision over bi-encoder similarity alone.

### Reranker ABC

```python
from abc import ABC, abstractmethod

class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Re-score and re-order results. top_k=None returns all, re-ordered."""
        pass
```

### CohereReranker

Uses Cohere Rerank API v2:

```python
from selectools.rag import CohereReranker

# pip install cohere  # or pip install selectools[rag]
reranker = CohereReranker(
    model="rerank-v3.5",
    api_key=None,  # Defaults to COHERE_API_KEY env var
)

# Standalone
reranked = reranker.rerank("best programming language", candidates, top_k=3)

# With HybridSearcher
searcher = HybridSearcher(
    vector_store=store,
    reranker=CohereReranker(model="rerank-v3.5"),
)
results = searcher.search("Python vs Java", top_k=5)  # Reranked after fusion
```

### JinaReranker

Uses Jina AI Rerank API via HTTP:

```python
from selectools.rag import JinaReranker

# pip install requests
# JINA_API_KEY env var or api_key="..."
reranker = JinaReranker(
    model="jina-reranker-v2-base-multilingual",
    api_key=None,
    api_url="https://api.jina.ai/v1/rerank",
)

# With HybridSearcher
searcher = HybridSearcher(
    vector_store=store,
    reranker=JinaReranker(),
)
results = searcher.search("multilingual document search", top_k=5)
```

### Integration Flow

```
Vector + BM25 → Fusion → [Fused Candidates] → Reranker → [Final top_k]
```

Reranking is applied **after** fusion. The reranker receives the fused candidate list and returns the final `top_k` by relevance score.

---

## Configuration Guide

### Tuning Weights

```python
# Semantic-heavy (conceptual queries, paraphrasing)
searcher = HybridSearcher(
    vector_store=store,
    vector_weight=0.7,
    keyword_weight=0.3,
)

# Keyword-heavy (exact terms, IDs, acronyms)
searcher = HybridSearcher(
    vector_store=store,
    vector_weight=0.3,
    keyword_weight=0.7,
)

# Balanced (default)
searcher = HybridSearcher(
    vector_store=store,
    vector_weight=0.5,
    keyword_weight=0.5,
)
```

### Fusion Method Selection

| Scenario | Recommended |
|----------|-------------|
| Default, diverse score scales | RRF |
| Need interpretable score contributions | WEIGHTED |
| Many candidate sources | RRF |

### Candidate Sizes

```python
# Retrieve more candidates for better fusion (default: top_k * 2)
results = searcher.search(
    "query",
    top_k=5,
    vector_top_k=20,
    keyword_top_k=20,
)
```

### Reranker Selection

| Provider | Model | Best For |
|----------|-------|----------|
| Cohere | rerank-v3.5 | High precision, English-heavy |
| Jina | jina-reranker-v2-base-multilingual | Multilingual content |

---

## RAGTool vs HybridSearchTool

| Aspect | RAGTool | HybridSearchTool |
|--------|---------|------------------|
| **Retrieval** | Vector only | Vector + BM25 |
| **Constructor** | `RAGTool(vector_store=...)` | `HybridSearchTool(searcher=...)` |
| **Exact terms** | May miss | Better recall |
| **Names, acronyms** | May miss | Better recall |
| **Paraphrasing** | Strong | Strong |
| **Setup** | Simpler | Requires HybridSearcher |
| **Dependencies** | Embedding provider | Embedding provider + BM25 (no extra deps) |

### When to Use RAGTool

- Simple RAG, fast setup
- Purely conceptual/semantic queries
- Minimal infrastructure

### When to Use HybridSearchTool

- Queries with exact terms, names, IDs, acronyms (e.g. "GDPR Article 32", "API v2")
- Technical documentation with code identifiers
- When recall matters more than simplicity

### Migration Example

```python
# Before (RAGTool)
rag_tool = RAGTool(vector_store=store, top_k=5)
agent = Agent(tools=[rag_tool.search_knowledge_base], provider=provider)

# After (HybridSearchTool) - drop-in replacement
searcher = HybridSearcher(vector_store=store)
searcher.add_documents(docs)  # or index_existing_documents if store already full
hybrid_tool = HybridSearchTool(searcher=searcher, top_k=5)
agent = Agent(tools=[hybrid_tool.search_knowledge_base], provider=provider)
```

---

## Best Practices

### 1. Use Hybrid Search for Mixed Queries

```python
# Queries with both concepts and exact terms benefit most
searcher.search("OpenAI GPT-4 API rate limits")  # "GPT-4" exact, "rate limits" semantic
```

### 2. Index Once, Search Many

```python
# Add documents once
searcher.add_documents(docs)

# Search repeatedly
for query in user_queries:
    results = searcher.search(query, top_k=5)
```

### 3. Add Reranking for Quality-Critical Use Cases

```python
searcher = HybridSearcher(
    vector_store=store,
    reranker=CohereReranker(),
)
# Better precision at cost of latency and API usage
```

### 4. Tune BM25 for Your Corpus

```python
# Longer documents: increase b for length normalisation
bm25 = BM25(k1=1.5, b=0.8)

# Shorter chunks: reduce b
bm25 = BM25(k1=1.5, b=0.5)
```

### 5. Use Metadata Filters Consistently

```python
# Both vector and BM25 receive the same filter
results = searcher.search(
    "query",
    filter={"category": "api", "version": "v2"},
)
```

---

## Troubleshooting

### No Results from BM25

```python
# Issue: All terms filtered as stop words
bm25 = BM25(remove_stopwords=True)
results = bm25.search("a the is", top_k=5)  # Empty

# Fix: Disable stop word removal for query-heavy content
bm25 = BM25(remove_stopwords=False)
```

### Vector Store Has No Embedder

```python
# Error: "Vector store does not have an embedding provider configured."

# Fix: Pass embedder when creating store
store = VectorStore.create("memory", embedder=embedder)
searcher = HybridSearcher(vector_store=store)
```

### Reranker API Errors

```python
# Cohere: Set COHERE_API_KEY
# Jina: Set JINA_API_KEY or pass api_key to constructor

import os
os.environ["COHERE_API_KEY"] = "your-key"
# or
reranker = JinaReranker(api_key="your-jina-key")
```

### Duplicate Documents in Results

HybridSearcher deduplicates by `document.text`. Ensure documents with identical text are intended (e.g. same chunk from different sources). If you see duplicates, check that `Document` instances are not being recreated with different object identities but same text.

### Slow Search

```python
# Issue: Large vector_top_k and keyword_top_k

# Fix: Reduce candidate sizes
results = searcher.search(
    query,
    top_k=5,
    vector_top_k=10,
    keyword_top_k=10,
)
```

---

## Further Reading

- [RAG Module](RAG.md) - Complete RAG pipeline, document loading, chunking
- [Embeddings Module](EMBEDDINGS.md) - Embedding providers for vector search
- [Vector Stores Module](VECTOR_STORES.md) - VectorStore implementations
- [Tools Module](TOOLS.md) - Tool decorator and agent integration

---

**Next Steps:** Integrate hybrid search into your RAG pipeline by following the [RAG Module](RAG.md) and swapping `RAGTool` for `HybridSearchTool` when exact term recall matters.
