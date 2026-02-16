# Advanced Chunking

**Directory:** `src/selectools/rag/`
**Source:** `chunking.py`

## Table of Contents

1. [Overview](#overview)
2. [Chunking Progression](#chunking-progression)
3. [Quick Start](#quick-start)
4. [SemanticChunker](#semanticchunker)
5. [ContextualChunker](#contextualchunker)
6. [Composability](#composability)
7. [Full Pipeline](#full-pipeline)
8. [Comparison Table](#comparison-table)
9. [Best Practices](#best-practices)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)
12. [Further Reading](#further-reading)

---

## Overview

**Advanced Chunking** provides semantic and contextual text splitting strategies that go beyond fixed-size or recursive splitting. These chunkers produce higher-quality chunks for RAG systems by:

- **SemanticChunker**: Splitting at topic boundaries using embedding similarity (not character count)
- **ContextualChunker**: Enriching each chunk with LLM-generated context that situates it within the full document

Both chunkers integrate with the selectools RAG pipeline: load → chunk → embed → store → search.

---

## Chunking Progression

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CHUNKING EVOLUTION                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Fixed-size (TextSplitter)                                               │
│     └─ Split at every N characters                                          │
│     └─ Fast, predictable, but may cut mid-sentence or mid-topic            │
│                                                                             │
│  2. Recursive (RecursiveTextSplitter)                                       │
│     └─ Prefer natural boundaries: \n\n → \n → . → space → char             │
│     └─ Better coherence, still character-driven                             │
│                                                                             │
│  3. Semantic (SemanticChunker)                                               │
│     └─ Split when embedding similarity between consecutive sentences drops   │
│     └─ Groups sentences by topic; chunks follow meaning boundaries          │
│                                                                             │
│  4. Contextual (ContextualChunker)                                           │
│     └─ Wraps ANY chunker; adds LLM-generated context per chunk               │
│     └─ Each chunk prefixed with situating description (Anthropic technique) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### SemanticChunker

```python
from selectools.rag import SemanticChunker, Document
from selectools.embeddings import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider()
chunker = SemanticChunker(embedder, similarity_threshold=0.75)

# Split text
text = "Topic A sentence 1. Topic A sentence 2. Topic B sentence 1. Topic B sentence 2."
chunks = chunker.split_text(text)

# Split documents
docs = [Document(text=text, metadata={"source": "doc.txt"})]
chunked_docs = chunker.split_documents(docs)
```

### ContextualChunker

```python
from selectools.rag import ContextualChunker, RecursiveTextSplitter, Document
from selectools import OpenAIProvider

base = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=200)
provider = OpenAIProvider()
chunker = ContextualChunker(base_chunker=base, provider=provider)

docs = [Document(text="Full document about product features and pricing...")]
enriched_docs = chunker.split_documents(docs)

# Each chunk now has: [Context] <description>\n\n<original chunk>
for doc in enriched_docs:
    print(doc.metadata.get("context"))
```

---

## SemanticChunker

### How It Works

SemanticChunker splits documents at **topic boundaries** by comparing consecutive sentence embeddings. Consecutive sentences with high cosine similarity stay together; when similarity drops below the threshold, a new chunk starts.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SEMANTIC CHUNKER FLOW                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Text Input                                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────┐                                                      │
│  │ Sentence Split   │  _split_into_sentences(text)                          │
│  │ (regex heuristic)│  Handles abbreviations, decimals                     │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ▼                                                                 │
│  [Sent1, Sent2, Sent3, ...]                                                 │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                      │
│  │ Embed            │  embedder.embed_texts(sentences)                      │
│  │ (EmbeddingProvider)                                                      │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ▼                                                                 │
│  [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]                                   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                      │
│  │ Similarity Check  │  _cosine_similarity(emb[i-1], emb[i])                 │
│  │ (pure Python)     │  No numpy required                                    │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ▼                                                                 │
│  sim >= threshold? → Keep in group                                          │
│  sim <  threshold? → Split (if >= min_chunk_sentences)                        │
│  len(group) >= max? → Force split                                            │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                      │
│  │ Group            │  Join sentences per chunk                              │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ▼                                                                 │
│  [Chunk1, Chunk2, Chunk3, ...]                                              │
│  metadata: chunker="semantic", chunk, total_chunks                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Constructor

```python
SemanticChunker(
    embedder: EmbeddingProvider,
    similarity_threshold: float = 0.75,
    min_chunk_sentences: int = 1,
    max_chunk_sentences: int = 50,
)
```

| Parameter              | Type               | Default | Description                                                |
| ---------------------- | ------------------ | ------- | ---------------------------------------------------------- |
| `embedder`             | EmbeddingProvider  | —       | Provider for sentence embeddings (OpenAI, Gemini, etc.)    |
| `similarity_threshold` | float             | 0.75    | Below this cosine similarity, start a new chunk (0.0–1.0)   |
| `min_chunk_sentences`  | int                | 1       | Minimum sentences before a split is allowed                |
| `max_chunk_sentences`  | int                | 50      | Maximum sentences per chunk (forces split if exceeded)      |

### Methods

| Method                     | Returns         | Description                                      |
| -------------------------- | --------------- | ------------------------------------------------ |
| `split_text(text: str)`    | `List[str]`     | Split raw text into semantic chunks              |
| `split_documents(documents)`| `List[Document]`| Split documents, preserve and add metadata        |

### Metadata Added

```python
{
    "chunker": "semantic",
    "chunk": 0,           # Index of this chunk
    "total_chunks": 3,    # Total chunks from this document
    # ... original metadata (source, etc.)
}
```

### Supported Embedding Providers

Any provider implementing `EmbeddingProvider`:

```python
from selectools.embeddings import (
    OpenAIEmbeddingProvider,
    GeminiEmbeddingProvider,
    AnthropicEmbeddingProvider,
    CohereEmbeddingProvider,
)

# All work with SemanticChunker
chunker = SemanticChunker(GeminiEmbeddingProvider(), similarity_threshold=0.75)  # Free
chunker = SemanticChunker(OpenAIEmbeddingProvider(), similarity_threshold=0.80)
```

### Threshold Tuning Guide

| Threshold | Effect                            | Use Case                                      |
| --------- | --------------------------------- | --------------------------------------------- |
| **0.85+** | Fewer, larger chunks              | Dense technical docs, single-topic content     |
| **0.75**  | Balanced (default)                | General documents, mixed topics              |
| **0.65**  | More, smaller chunks              | Multi-topic docs, news articles               |
| **0.55**  | Very fine-grained                 | Each sentence often its own chunk             |

```python
# Dense technical manual
chunker = SemanticChunker(embedder, similarity_threshold=0.85)

# News article with many unrelated sections
chunker = SemanticChunker(embedder, similarity_threshold=0.65)
```

### Implementation Notes

- **Pure Python cosine similarity**: Uses `math.sqrt` and built-in `sum`; no numpy dependency.
- **Sentence splitting**: Regex `(?<=[.!?])\s+(?=[A-Z"\'(\[])` to avoid splitting on abbreviations (e.g. "Dr.", "U.S.") or decimals.
- **Helper functions** (internal): `_split_into_sentences(text)`, `_cosine_similarity(a, b)`.

---

## ContextualChunker

### How It Works

ContextualChunker wraps **any** chunker and enriches each chunk with an LLM-generated description. The LLM sees the **full document** and the **chunk**, then produces a 1–2 sentence description that situates the chunk within the document. This description is prepended to the chunk text so that embeddings capture the chunk's role in context.

Inspired by [Anthropic's Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval).

### Format

Each enriched chunk:

```
[Context] <LLM-generated 1-2 sentence description>

<original chunk text>
```

### Constructor

```python
ContextualChunker(
    base_chunker: Any,              # Object with split_documents()
    provider: Provider,
    model: str = "gpt-4o-mini",
    prompt_template: Optional[str] = None,
    max_document_chars: int = 50_000,
    context_prefix: str = "[Context] ",
)
```

| Parameter            | Type     | Default     | Description                                   |
| -------------------- | -------- | ----------- | --------------------------------------------- |
| `base_chunker`       | Any      | —           | Chunker with `split_documents()` method       |
| `provider`           | Provider | —           | LLM provider for context generation           |
| `model`              | str      | "gpt-4o-mini" | Model for context generation                |
| `prompt_template`    | str or None | None     | Custom prompt with `{document}` and `{chunk}` |
| `max_document_chars` | int      | 50_000      | Truncate document to this before prompting    |
| `context_prefix`     | str      | "[Context] "| Prefix before generated description          |

### Methods

| Method                      | Returns         | Description                                      |
| --------------------------- | --------------- | ------------------------------------------------ |
| `split_documents(documents)`| `List[Document]`| Split and enrich each chunk with context         |
| `split_text(text: str)`     | `List[str]`     | Convenience: wrap in Document, split, return text|

### Metadata Added

```python
{
    "chunker": "contextual",
    "context": "LLM-generated description string",
    # ... base chunker metadata (chunk, total_chunks, etc.)
}
```

### Default Prompt Template

```
Document:
<document>
{document}
</document>

Chunk:
<chunk>
{chunk}
</chunk>

Give a short (1-2 sentence) description that situates this chunk
within the overall document for search and retrieval purposes.
Respond ONLY with the description, nothing else.
```

### Custom Prompt Examples

**Technical documentation:**

```python
custom_prompt = (
    "Document:\n<document>\n{document}\n</document>\n\n"
    "Chunk:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Describe this chunk's role in the API documentation in 1-2 sentences. "
    "Include the section or feature it covers. Response only."
)
chunker = ContextualChunker(base, provider, prompt_template=custom_prompt)
```

**Legal / contracts:**

```python
legal_prompt = (
    "Document:\n<document>\n{document}\n</document>\n\n"
    "Chunk:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Summarize what legal clause or provision this chunk covers in 1-2 sentences. "
    "Be precise. Respond only with the description."
)
chunker = ContextualChunker(base, provider, prompt_template=legal_prompt)
```

**Non-English:**

```python
spanish_prompt = (
    "Documento:\n<document>\n{document}\n</document>\n\n"
    "Fragmento:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Da una descripción breve (1-2 oraciones) que sitúe este fragmento "
    "en el documento completo para búsqueda. Responde SOLO con la descripción."
)
chunker = ContextualChunker(base, provider, prompt_template=spanish_prompt)
```

### Compatible Base Chunkers

- `TextSplitter`
- `RecursiveTextSplitter`
- `SemanticChunker`
- Any object with `split_documents(documents) -> List[Document]`

---

## Composability

ContextualChunker wraps any chunker. A common pattern is **SemanticChunker** as the base for **ContextualChunker**:

```python
from selectools.rag import SemanticChunker, ContextualChunker, Document
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools import OpenAIProvider

# Semantic splitting for topic boundaries
embedder = OpenAIEmbeddingProvider()
semantic = SemanticChunker(embedder, similarity_threshold=0.75)

# Add LLM-generated context to each chunk
provider = OpenAIProvider()
chunker = ContextualChunker(base_chunker=semantic, provider=provider)

docs = [Document(text="Multi-topic document...", metadata={"source": "doc.txt"})]
enriched = chunker.split_documents(docs)

# Result: semantic boundaries + contextual enrichment
```

```
┌────────────────────────────────────────────────────────────┐
│ COMPOSABILITY                                                │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Document                                                   │
│       │                                                     │
│       ▼                                                     │
│  SemanticChunker.split_documents()                          │
│       │                                                     │
│       ▼                                                     │
│  [Chunk1, Chunk2, Chunk3]  (topic boundaries)               │
│       │                                                     │
│       ▼                                                     │
│  ContextualChunker (for each chunk)                          │
│       │                                                     │
│       ├─ LLM: "Chunk1 in context of full doc"                │
│       ├─ LLM: "Chunk2 in context of full doc"                │
│       └─ LLM: "Chunk3 in context of full doc"                │
│       │                                                     │
│       ▼                                                     │
│  [Context] <desc1>\n\nChunk1                                 │
│  [Context] <desc2>\n\nChunk2                                 │
│  [Context] <desc3>\n\nChunk3                                 │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Full Pipeline

End-to-end flow: Semantic + Contextual → VectorStore → Search

```python
from selectools.rag import (
    DocumentLoader,
    SemanticChunker,
    ContextualChunker,
    RecursiveTextSplitter,
    VectorStore,
    RAGTool,
)
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools import OpenAIProvider, Agent

# 1. Load
docs = DocumentLoader.from_directory("./docs", glob_pattern="**/*.md")

# 2. Chunk (Semantic + Contextual)
embedder = OpenAIEmbeddingProvider()
semantic = SemanticChunker(embedder, similarity_threshold=0.75)
provider = OpenAIProvider()
chunker = ContextualChunker(base_chunker=semantic, provider=provider)
chunked = chunker.split_documents(docs)

# 3. Store
store = VectorStore.create("sqlite", embedder=embedder, db_path="kb.db")
store.add_documents(chunked)

# 4. Search
rag_tool = RAGTool(vector_store=store, top_k=3)
agent = Agent(tools=[rag_tool.search_knowledge_base], provider=provider)
response = agent.run("What are the installation steps?")
```

---

## Comparison Table

| Feature              | TextSplitter | RecursiveTextSplitter | SemanticChunker | ContextualChunker  |
| -------------------- | ------------ | --------------------- | --------------- | ------------------ |
| **Split criterion**  | Character count | Character + separators | Embedding similarity | Any (wraps base) |
| **Topic boundaries** | ❌           | Partial               | ✅              | ✅ (via context)   |
| **External API**     | None         | None                  | Embeddings      | LLM completion     |
| **Cost**             | Free         | Free                  | Embedding cost  | Embedding + LLM    |
| **Speed**            | Fast         | Fast                  | Slower (embeds) | Slowest (LLM/embed)|
| **Chunk quality**    | Basic        | Good                  | High            | Highest            |
| **Dependencies**     | None         | None                  | EmbeddingProvider | Provider + base  |
| **Use case**         | Simple RAG   | General RAG           | Topic-aware RAG | High-precision RAG |

---

## Best Practices

### 1. Use SemanticChunker for Multi-Topic Documents

```python
# ✅ Good - topic boundaries
chunker = SemanticChunker(embedder, similarity_threshold=0.75)

# ❌ Less ideal - may cut mid-topic
splitter = RecursiveTextSplitter(chunk_size=1000)
```

### 2. Use Gemini Embeddings for Cost Savings

```python
from selectools.embeddings import GeminiEmbeddingProvider

# SemanticChunker with free embeddings
embedder = GeminiEmbeddingProvider()
chunker = SemanticChunker(embedder)
```

### 3. Reserve ContextualChunker for High-Value Content

```python
# Use for critical knowledge bases (legal, medical, support)
# Skip for large, low-stakes corpuses
chunker = ContextualChunker(semantic, provider)
```

### 4. Tune `similarity_threshold` per Domain

```python
# Dense technical docs
chunker = SemanticChunker(embedder, similarity_threshold=0.85)

# Mixed articles
chunker = SemanticChunker(embedder, similarity_threshold=0.70)
```

### 5. Limit Context Generation Cost with `max_document_chars`

```python
# Truncate very long documents to control prompt size
chunker = ContextualChunker(
    base,
    provider,
    max_document_chars=30_000  # ~7.5k tokens
)
```

---

## Performance Considerations

### SemanticChunker

- **Embedding API calls**: One batch per document (all sentences).
- **Batch size**: Typically 10–200 sentences per document.
- **Cost**: Embedding cost only (e.g. OpenAI ~$0.02/1M tokens).

### ContextualChunker

- **LLM API calls**: One completion **per chunk**.
- **Cost**: Embedding (for chunks) + LLM (for context generation).
- **Example**: 50 chunks × gpt-4o-mini ≈ 50 completion calls; cost scales with chunk count.

```python
# Estimate: 50 chunks, ~200 tokens per completion
# gpt-4o-mini: ~$0.15/1M input, $0.60/1M output
# 50 × 200 ≈ 10k output tokens ≈ $0.006 per document
```

### Recommendations

| Scenario                  | Suggested approach                          |
| ------------------------- | ------------------------------------------- |
| Large corpus, budget     | RecursiveTextSplitter or SemanticChunker    |
| Small, high-value docs   | ContextualChunker (Semantic base)           |
| Prototyping              | RecursiveTextSplitter                       |
| Production, mixed usage  | SemanticChunker + optional ContextualChunker|

---

## Troubleshooting

### SemanticChunker: Chunks Too Large

```python
# Issue: Single-topic docs produce very large chunks

# Fix: Lower similarity_threshold or reduce max_chunk_sentences
chunker = SemanticChunker(
    embedder,
    similarity_threshold=0.65,
    max_chunk_sentences=25
)
```

### SemanticChunker: Chunks Too Small

```python
# Issue: Every sentence becomes its own chunk

# Fix: Raise similarity_threshold
chunker = SemanticChunker(embedder, similarity_threshold=0.85)
```

### ContextualChunker: Slow or Expensive

```python
# Issue: Too many chunks → many LLM calls

# Fix: Use a chunker that produces fewer, larger chunks
base = RecursiveTextSplitter(chunk_size=1500, chunk_overlap=300)
chunker = ContextualChunker(base_chunker=base, provider=provider)
```

### ContextualChunker: Truncated Document Context

```python
# Issue: Document truncated before relevant section

# Fix: Increase max_document_chars (watch token limits)
chunker = ContextualChunker(
    base,
    provider,
    max_document_chars=80_000  # ~20k tokens
)
```

### Embedding Rate Limits

```python
# Issue: SemanticChunker hits embedding API rate limits on large docs

# Fix: Process documents in smaller batches, or use Gemini (higher limits)
embedder = GeminiEmbeddingProvider()
chunker = SemanticChunker(embedder)
```

---

## Further Reading

- [RAG Module](RAG.md) - Full RAG pipeline and RAGAgent
- [Embeddings Module](EMBEDDINGS.md) - Embedding providers
- [Vector Stores Module](VECTOR_STORES.md) - Storage and search

---

**Next Steps:** Integrate advanced chunking into your pipeline via [RAG Module](RAG.md).
