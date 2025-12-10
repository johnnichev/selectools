# Embeddings Module

**Directory:** `src/selectools/embeddings/`
**Files:** `provider.py`, `openai.py`, `anthropic.py`, `gemini.py`, `cohere.py`

## Table of Contents

1. [Overview](#overview)
2. [Embedding Provider Protocol](#embedding-provider-protocol)
3. [Provider Implementations](#provider-implementations)
4. [Usage Patterns](#usage-patterns)
5. [Cost Comparison](#cost-comparison)
6. [Implementation Details](#implementation-details)

---

## Overview

**Embedding Providers** convert text into dense vector representations (embeddings) that capture semantic meaning. These vectors enable:

- Semantic search (finding similar content)
- Clustering (grouping related content)
- Classification (categorizing content)
- Recommendation (suggesting similar items)

### Why Embeddings?

Traditional keyword search fails on:

```
Query: "How do I install the library?"
Document: "Setup instructions for getting started"
```

Keyword match: **0% overlap**
Semantic similarity: **85% similar**

Embeddings capture meaning, not just words.

---

## Embedding Provider Protocol

### Interface

```python
from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (batch operation)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a query (may differ from document embedding)."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding vector dimension."""
        pass
```

### Key Methods

- **`embed_text()`**: Single text → embedding
- **`embed_texts()`**: Multiple texts → embeddings (batched, more efficient)
- **`embed_query()`**: Query text → embedding (some models differentiate)
- **`dimension`**: Vector size (e.g., 1536 for OpenAI text-embedding-3-small)

---

## Provider Implementations

### OpenAI Embeddings

```python
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.models import OpenAI

embedder = OpenAIEmbeddingProvider(
    api_key="sk-...",  # Or set OPENAI_API_KEY
    model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id
)

# Embed single text
embedding = embedder.embed_text("Hello world")
print(f"Dimension: {embedder.dimension}")  # 1536

# Embed multiple texts (batched)
embeddings = embedder.embed_texts([
    "First document",
    "Second document",
    "Third document"
])
print(f"Got {len(embeddings)} embeddings")
```

**Available Models:**

- `text-embedding-3-small` (1536d, $0.02/1M tokens) - **Recommended**
- `text-embedding-3-large` (3072d, $0.13/1M tokens) - Higher quality
- `text-embedding-ada-002` (1536d, $0.10/1M tokens) - Legacy

### Anthropic/Voyage Embeddings

```python
from selectools.embeddings import AnthropicEmbeddingProvider
from selectools.models import Anthropic

embedder = AnthropicEmbeddingProvider(
    api_key="...",  # Or set VOYAGE_API_KEY
    model=Anthropic.Embeddings.VOYAGE_3_LITE.id
)

embedding = embedder.embed_text("Hello world")
print(f"Dimension: {embedder.dimension}")  # 512
```

**Available Models:**

- `voyage-3` (1024d, $0.06/1M tokens) - High quality
- `voyage-3-lite` (512d, $0.02/1M tokens) - **Recommended**

**Note:** Uses Voyage AI API (Anthropic's embedding partner)

### Gemini Embeddings

```python
from selectools.embeddings import GeminiEmbeddingProvider
from selectools.models import Gemini

embedder = GeminiEmbeddingProvider(
    api_key="...",  # Or set GEMINI_API_KEY / GOOGLE_API_KEY
    model=Gemini.Embeddings.EMBEDDING_004.id
)

embedding = embedder.embed_text("Hello world")
print(f"Dimension: {embedder.dimension}")  # 768
```

**Available Models:**

- `text-embedding-004` (768d, **FREE**) - **Recommended for cost**
- `text-embedding-001` (768d, **FREE**) - Legacy

**Best Feature:** Completely free (with rate limits)

### Cohere Embeddings

```python
from selectools.embeddings import CohereEmbeddingProvider
from selectools.models import Cohere

embedder = CohereEmbeddingProvider(
    api_key="...",  # Or set COHERE_API_KEY
    model=Cohere.Embeddings.EMBED_V3.id
)

embedding = embedder.embed_text("Hello world")
print(f"Dimension: {embedder.dimension}")  # 1024
```

**Available Models:**

- `embed-english-v3.0` (1024d, $0.10/1M tokens)
- `embed-multilingual-v3.0` (1024d, $0.10/1M tokens) - 100+ languages
- `embed-english-light-v3.0` (384d, $0.10/1M tokens) - Smaller

---

## Usage Patterns

### With RAG System

```python
from selectools.rag import VectorStore, RAGAgent
from selectools.embeddings import OpenAIEmbeddingProvider

# 1. Create embedding provider
embedder = OpenAIEmbeddingProvider()

# 2. Create vector store
store = VectorStore.create("sqlite", embedder=embedder, db_path="docs.db")

# 3. Create RAG agent
agent = RAGAgent.from_directory(
    directory="./docs",
    provider=provider,
    vector_store=store
)
```

### Manual Embedding

```python
# Embed documents
docs = ["Document 1", "Document 2", "Document 3"]
doc_embeddings = embedder.embed_texts(docs)

# Embed query
query = "What is Document 1 about?"
query_embedding = embedder.embed_query(query)

# Compute similarity (cosine similarity)
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

for i, doc_emb in enumerate(doc_embeddings):
    similarity = cosine_similarity(query_embedding, doc_emb)
    print(f"Document {i+1} similarity: {similarity:.3f}")
```

### Batch Processing

```python
# ✅ Good - Batch processing
texts = ["Text 1", "Text 2", ..., "Text 1000"]
embeddings = embedder.embed_texts(texts)  # Single API call (batched)

# ❌ Bad - Individual calls
embeddings = [embedder.embed_text(text) for text in texts]  # 1000 API calls!
```

### Document vs Query Embedding

Some models optimize differently for documents and queries:

```python
# For indexing documents
doc_embeddings = embedder.embed_texts(documents)

# For searching
query_embedding = embedder.embed_query("search query")
```

Most providers (OpenAI, Gemini) use the same method internally, but Cohere differentiates.

---

## Cost Comparison

### Per 1M Tokens

| Provider             | Model                  | Dimension | Cost      | Notes              |
| -------------------- | ---------------------- | --------- | --------- | ------------------ |
| **Gemini**           | text-embedding-004     | 768       | **$0.00** | FREE (with limits) |
| **OpenAI**           | text-embedding-3-small | 1536      | $0.02     | Best value         |
| **Anthropic/Voyage** | voyage-3-lite          | 512       | $0.02     | Compact            |
| **Anthropic/Voyage** | voyage-3               | 1024      | $0.06     | High quality       |
| **OpenAI**           | text-embedding-ada-002 | 1536      | $0.10     | Legacy             |
| **Cohere**           | embed-v3               | 1024      | $0.10     | Multilingual       |
| **OpenAI**           | text-embedding-3-large | 3072      | $0.13     | Premium            |

### Cost Example

Embedding 1,000 documents (avg 500 tokens each):

```
Total tokens: 1,000 × 500 = 500,000 tokens

Gemini:    $0.00 (FREE)
OpenAI-small:  500k / 1M × $0.02 = $0.01
Voyage-lite:   500k / 1M × $0.02 = $0.01
Voyage-3:      500k / 1M × $0.06 = $0.03
OpenAI-large:  500k / 1M × $0.13 = $0.065
```

### Recommendation

- **Budget/Hobby Projects**: Gemini (free)
- **Production (balanced)**: OpenAI text-embedding-3-small
- **Production (premium)**: OpenAI text-embedding-3-large or Voyage-3
- **Multilingual**: Cohere embed-multilingual-v3.0

---

## Implementation Details

### OpenAI Provider

```python
class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=self.api_key)
        self.model = model
        self._dimension = self._get_dimension()

    def embed_text(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self.model,
            input=texts  # Batch API call
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)  # Same for OpenAI

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_dimension(self) -> int:
        # Model-specific dimensions
        if "text-embedding-3-small" in self.model:
            return 1536
        elif "text-embedding-3-large" in self.model:
            return 3072
        elif "ada-002" in self.model:
            return 1536
        return 1536  # default
```

### Gemini Provider

```python
class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-004"):
        import google.generativeai as genai

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=f"models/{self.model}",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Batch embedding
        results = []
        for text in texts:
            results.append(self.embed_text(text))
        return results

    def embed_query(self, query: str) -> List[float]:
        result = genai.embed_content(
            model=f"models/{self.model}",
            content=query,
            task_type="retrieval_query"  # Different task type
        )
        return result["embedding"]

    @property
    def dimension(self) -> int:
        return 768  # Gemini embeddings are 768-dimensional
```

---

## Best Practices

### 1. Batch When Possible

```python
# ✅ Good
embeddings = embedder.embed_texts(all_docs)

# ❌ Bad
embeddings = [embedder.embed_text(doc) for doc in all_docs]
```

### 2. Cache Embeddings

```python
# Embed once, store in vector DB
store.add_documents(docs)  # Embeddings stored

# Reuse for all queries
results = store.search(query_embedding)
```

### 3. Choose Appropriate Dimension

Higher dimensions ≠ always better:

- More dimensions → more storage, slower search
- Consider 512-768d for most applications
- Use 1536-3072d for high-precision requirements

### 4. Monitor Costs

```python
from selectools.rag import RAGAgent

agent = RAGAgent.from_directory("./docs", provider, store)

# Check embedding costs
print(agent.usage)
# Shows LLM + embedding costs separately
```

### 5. Use Free Tier for Development

```python
# Development
embedder = GeminiEmbeddingProvider()  # FREE

# Production
embedder = OpenAIEmbeddingProvider()  # Paid but reliable
```

---

## Testing

```python
def test_embedding_provider():
    embedder = OpenAIEmbeddingProvider()

    # Test single embedding
    embedding = embedder.embed_text("Hello world")
    assert len(embedding) == embedder.dimension
    assert all(isinstance(x, float) for x in embedding)

    # Test batch embedding
    embeddings = embedder.embed_texts(["Text 1", "Text 2"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == embedder.dimension

    # Test query embedding
    query_emb = embedder.embed_query("search query")
    assert len(query_emb) == embedder.dimension

def test_embeddings_are_different():
    embedder = OpenAIEmbeddingProvider()

    emb1 = embedder.embed_text("Python programming")
    emb2 = embedder.embed_text("JavaScript coding")
    emb3 = embedder.embed_text("Cooking recipes")

    # Similar texts should have similar embeddings
    sim_prog = cosine_similarity(emb1, emb2)
    sim_unrelated = cosine_similarity(emb1, emb3)

    assert sim_prog > sim_unrelated  # Programming more similar than cooking
```

---

## Troubleshooting

### Rate Limits

```python
# Issue: Hit rate limits with free tier

# Fix: Add retry logic or use paid tier
import time

def embed_with_retry(embedder, texts, max_retries=3):
    for attempt in range(max_retries):
        try:
            return embedder.embed_texts(texts)
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

### High Memory Usage

```python
# Issue: Embedding millions of documents at once

# Fix: Batch processing
def embed_in_batches(embedder, texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embedder.embed_texts(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings
```

---

## Further Reading

- [RAG Module](RAG.md) - Using embeddings with RAG
- [Vector Stores Module](VECTOR_STORES.md) - Storing embeddings
- [Models Module](MODELS.md) - Embedding model metadata

---

**Next Steps:** Learn about vector storage in the [Vector Stores Module](VECTOR_STORES.md).
