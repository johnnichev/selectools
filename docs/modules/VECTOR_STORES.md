# Vector Stores Module

**Directory:** `src/selectools/rag/stores/`
**Files:** `memory.py`, `sqlite.py`, `chroma.py`, `pinecone.py`

## Table of Contents

1. [Overview](#overview)
2. [VectorStore Interface](#vectorstore-interface)
3. [Store Implementations](#store-implementations)
4. [Choosing a Store](#choosing-a-store)
5. [Implementation Details](#implementation-details)
6. [Best Practices](#best-practices)

---

## Overview

**Vector Stores** persist and search document embeddings. They provide:

- **Storage**: Save embeddings with metadata
- **Similarity Search**: Find most similar vectors (cosine similarity)
- **Filtering**: Search with metadata constraints
- **Scaling**: Handle thousands to billions of vectors

### Why Vector Stores?

Regular databases don't efficiently search by semantic similarity:

```
Query: "Find documents about machine learning"
Traditional DB: Text search for "machine learning"
Vector Store: Semantic search (finds "AI", "neural networks", "deep learning", etc.)
```

---

## VectorStore Interface

### Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    embedder: Optional[EmbeddingProvider] = None

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """Add documents, return IDs."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass
```

### Data Structures

```python
@dataclass
class Document:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    document: Document
    score: float  # Cosine similarity (0-1, higher is better)
```

---

## Store Implementations

### 1. In-Memory Store

**Best for:** Prototyping, testing, small datasets

```python
from selectools.rag import VectorStore
from selectools.embeddings import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider()
store = VectorStore.create("memory", embedder=embedder)

# Add documents
docs = [
    Document(text="Python is great", metadata={"topic": "programming"}),
    Document(text="JavaScript is popular", metadata={"topic": "programming"}),
]
ids = store.add_documents(docs)

# Search
query_emb = embedder.embed_query("programming languages")
results = store.search(query_emb, top_k=2)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.document.text}")
```

**Features:**

- ✅ Fast (in-memory)
- ✅ No dependencies
- ✅ NumPy-based similarity
- ❌ Not persistent
- ❌ Limited scalability

**Use Cases:**

- Quick prototypes
- Unit tests
- Small datasets (<10k docs)

### 2. SQLite Store

**Best for:** Production (small-medium), local storage

```python
store = VectorStore.create(
    "sqlite",
    embedder=embedder,
    db_path="knowledge.db"  # Persistent file
)

# Same API as memory store
ids = store.add_documents(docs)
results = store.search(query_emb, top_k=5)
```

**Features:**

- ✅ Persistent (file-based)
- ✅ No external dependencies
- ✅ ACID transactions
- ✅ Good for <100k documents
- ❌ Slower than in-memory
- ❌ Limited concurrent writes

**Use Cases:**

- Production apps (single instance)
- Local knowledge bases
- Desktop applications
- CI/CD pipelines

**Schema:**

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    metadata TEXT,  -- JSON
    embedding BLOB  -- Serialized vector
);
```

### 3. Chroma Store

**Best for:** Production (medium-large), advanced features

```python
# Requires: pip install chromadb
store = VectorStore.create(
    "chroma",
    embedder=embedder,
    persist_directory="./chroma_db",  # Persistent storage
    collection_name="my_knowledge"    # Optional
)

ids = store.add_documents(docs)
results = store.search(query_emb, top_k=5)
```

**Features:**

- ✅ Built for embeddings
- ✅ Persistent storage
- ✅ Efficient similarity search
- ✅ Metadata filtering
- ✅ Good for 10k-1M documents
- ✅ Local or server mode
- ❌ External dependency

**Use Cases:**

- Production applications
- Research projects
- Multi-user applications
- Large knowledge bases

**Installation:**

```bash
pip install selectools[rag]  # Includes chromadb
```

### 4. Pinecone Store

**Best for:** Production (scale), managed service

```python
# Requires: pip install pinecone-client
# Set environment: PINECONE_API_KEY, PINECONE_ENVIRONMENT

store = VectorStore.create(
    "pinecone",
    embedder=embedder,
    index_name="my-knowledge-base",
    namespace="production"  # Optional logical partition
)

ids = store.add_documents(docs)
results = store.search(query_emb, top_k=5)
```

**Features:**

- ✅ Fully managed (cloud)
- ✅ Auto-scaling
- ✅ Low latency at scale
- ✅ Handles billions of vectors
- ✅ High availability
- ❌ External service (cost)
- ❌ Network latency

**Use Cases:**

- Large-scale production
- Multi-region deployments
- High-traffic applications
- Billions of documents

**Setup:**

```bash
pip install selectools[rag]

# Set credentials
export PINECONE_API_KEY="your-key"
export PINECONE_ENVIRONMENT="your-env"
```

---

## Choosing a Store

### Decision Matrix

```
Documents:       <1k      <10k     <100k    <1M      1M+
                  │         │         │        │        │
                  ▼         ▼         ▼        ▼        ▼
┌─────────────────────────────────────────────────────────┐
│ Memory        ✓✓✓      ✓✓       ✓        ✗        ✗    │
│ SQLite        ✓✓✓      ✓✓✓      ✓✓       ✓        ✗    │
│ Chroma        ✓✓       ✓✓✓      ✓✓✓      ✓✓✓      ✓    │
│ Pinecone      ✓        ✓✓       ✓✓       ✓✓✓      ✓✓✓  │
└─────────────────────────────────────────────────────────┘

✓✓✓ = Excellent
✓✓  = Good
✓   = Acceptable
✗   = Not recommended
```

### Feature Comparison

| Feature         | Memory  | SQLite  | Chroma   | Pinecone  |
| --------------- | ------- | ------- | -------- | --------- |
| **Persistence** | ❌      | ✅      | ✅       | ✅        |
| **Setup**       | None    | None    | Minimal  | Account   |
| **Cost**        | Free    | Free    | Free     | $$$       |
| **Max Docs**    | 10k     | 100k    | 1M       | Billions  |
| **Speed**       | Fastest | Fast    | Fast     | Fast\*    |
| **Filtering**   | Basic   | Basic   | Advanced | Advanced  |
| **Concurrency** | Single  | Limited | Good     | Excellent |
| **Maintenance** | None    | None    | Low      | None      |

\* Network latency applies

### Recommendation Flow

```
Are you prototyping?
└─ Yes → Memory Store

Do you need persistence?
└─ No → Memory Store
└─ Yes ↓

How many documents?
├─ <100k → SQLite Store
├─ 100k-1M → Chroma Store
└─ >1M → Pinecone Store

Is this production?
└─ Yes, high-traffic → Pinecone Store
└─ Yes, low-traffic → SQLite or Chroma
└─ No → Any
```

---

## Implementation Details

### In-Memory Store

```python
class InMemoryVectorStore(VectorStore):
    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, List[float]] = {}

    def add_documents(self, documents, embeddings=None):
        if embeddings is None:
            # Generate embeddings
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        ids = []
        for doc, emb in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())
            self._documents[doc_id] = doc
            self._embeddings[doc_id] = emb
            ids.append(doc_id)

        return ids

    def search(self, query_embedding, top_k=5, filter=None):
        import numpy as np

        # Compute cosine similarity
        results = []
        for doc_id, doc_emb in self._embeddings.items():
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )

            doc = self._documents[doc_id]

            # Apply filter
            if filter and not self._matches_filter(doc.metadata, filter):
                continue

            results.append(SearchResult(document=doc, score=similarity))

        # Sort by score (descending) and take top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
```

### SQLite Store

```python
class SQLiteVectorStore(VectorStore):
    def __init__(self, embedder: EmbeddingProvider, db_path: str = "vector_store.db"):
        self.embedder = embedder
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()

    def add_documents(self, documents, embeddings=None):
        if embeddings is None:
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        conn = sqlite3.connect(self.db_path)
        ids = []

        for doc, emb in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())

            # Serialize embedding
            emb_bytes = pickle.dumps(emb)
            meta_json = json.dumps(doc.metadata)

            conn.execute(
                "INSERT INTO documents (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
                (doc_id, doc.text, meta_json, emb_bytes)
            )
            ids.append(doc_id)

        conn.commit()
        conn.close()
        return ids

    def search(self, query_embedding, top_k=5, filter=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT id, text, metadata, embedding FROM documents")

        results = []
        for row in cursor:
            doc_id, text, meta_json, emb_bytes = row
            doc_emb = pickle.loads(emb_bytes)

            # Compute similarity
            similarity = cosine_similarity(query_embedding, doc_emb)

            metadata = json.loads(meta_json)
            if filter and not matches_filter(metadata, filter):
                continue

            doc = Document(text=text, metadata=metadata)
            results.append(SearchResult(document=doc, score=similarity))

        conn.close()

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
```

---

## Best Practices

### 1. Choose Appropriate Backend

```python
# ✅ Good - Match use case
if ENV == "development":
    store = VectorStore.create("memory", embedder)
elif DOCUMENT_COUNT < 100000:
    store = VectorStore.create("sqlite", embedder, db_path="prod.db")
else:
    store = VectorStore.create("pinecone", embedder, index_name="prod")

# ❌ Bad - Wrong tool for job
store = VectorStore.create("memory", embedder)  # Production with millions of docs
```

### 2. Use Metadata Filtering

```python
# Add documents with metadata
docs = [
    Document(text="...", metadata={"category": "api", "version": "v2"}),
    Document(text="...", metadata={"category": "guide", "version": "v2"}),
]
store.add_documents(docs)

# Search with filter
results = store.search(
    query_emb,
    top_k=5,
    filter={"category": "api"}  # Only API docs
)
```

### 3. Batch Document Addition

```python
# ✅ Good - Batch
all_docs = load_all_documents()
store.add_documents(all_docs)

# ❌ Bad - One at a time
for doc in all_docs:
    store.add_documents([doc])  # Inefficient
```

### 4. Clean Up Periodically

```python
# Remove old or irrelevant documents
old_ids = get_outdated_document_ids()
store.delete(old_ids)
```

### 5. Monitor Performance

```python
import time

start = time.time()
results = store.search(query_emb, top_k=10)
duration = time.time() - start

if duration > 1.0:
    logger.warning(f"Slow search: {duration:.2f}s")
```

---

## Testing

```python
def test_vector_store():
    from selectools.rag import VectorStore, Document
    from selectools.embeddings import OpenAIEmbeddingProvider

    embedder = OpenAIEmbeddingProvider()
    store = VectorStore.create("memory", embedder=embedder)

    # Add documents
    docs = [
        Document(text="Python programming", metadata={"lang": "python"}),
        Document(text="JavaScript coding", metadata={"lang": "js"}),
    ]
    ids = store.add_documents(docs)
    assert len(ids) == 2

    # Search
    query_emb = embedder.embed_query("Python")
    results = store.search(query_emb, top_k=1)

    assert len(results) == 1
    assert "Python" in results[0].document.text
    assert results[0].score > 0.5

    # Clean up
    store.delete(ids)
```

---

## Troubleshooting

### Slow Search

```python
# Issue: Linear scan for large datasets

# Fix: Use appropriate backend
store = VectorStore.create("chroma", embedder)  # Indexed search
```

### Out of Memory

```python
# Issue: Too many documents in memory

# Fix: Use persistent storage
store = VectorStore.create("sqlite", embedder)  # or Chroma/Pinecone
```

### Connection Errors (Pinecone)

```python
# Issue: Missing credentials

# Fix: Set environment variables
import os
os.environ["PINECONE_API_KEY"] = "your-key"
os.environ["PINECONE_ENVIRONMENT"] = "your-env"
```

---

## Further Reading

- [RAG Module](RAG.md) - Complete RAG system
- [Embeddings Module](EMBEDDINGS.md) - Generating embeddings
- [Architecture](../ARCHITECTURE.md) - System overview

---

**Next Steps:** Understand the model registry in the [Models Module](MODELS.md).
