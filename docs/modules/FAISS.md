---
description: "In-process FAISS vector index for fast local similarity search with disk persistence"
tags:
  - rag
  - vector-stores
  - faiss
---

# FAISS Vector Store

**Import:** `from selectools.rag.stores import FAISSVectorStore`
**Stability:** beta
**Added in:** v0.21.0

`FAISSVectorStore` wraps Facebook AI's FAISS library to provide a fast, in-process
vector index that lives entirely in memory but can be persisted to disk. It's ideal
when you want zero-server RAG with millions of vectors and have plenty of RAM.

```python title="faiss_quick.py"
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.rag import Document
from selectools.rag.stores import FAISSVectorStore

embedder = OpenAIEmbeddingProvider()
store = FAISSVectorStore(embedder=embedder)
store.add_documents([
    Document(text="Selectools is a Python AI agent framework."),
    Document(text="FAISS does fast similarity search."),
])

# search() takes a query embedding, not a string — embed the query first
query_vec = embedder.embed_query("agent framework")
results = store.search(query_vec, top_k=2)
for r in results:
    print(r.score, r.document.text)

store.save("faiss_index")  # writes index + documents
```

!!! tip "See Also"
    - [Qdrant](QDRANT.md) - Self-hosted vector store with REST + gRPC
    - [pgvector](PGVECTOR.md) - PostgreSQL-backed vector store
    - [RAG](RAG.md) - High-level retrieval pipeline

---

## Install

```bash
pip install "selectools[rag]"
```

`faiss-cpu>=1.7.0` is part of the `[rag]` optional extras. If you want GPU acceleration,
install `faiss-gpu` separately.

---

## Constructor

```python
FAISSVectorStore(
    embedder: EmbeddingProvider | None = None,
    dimension: int | None = None,
)
```

| Parameter | Description |
|---|---|
| `embedder` | Any `selectools.embeddings.EmbeddingProvider`. May be `None` when loading a persisted index that already contains pre-computed vectors. |
| `dimension` | Vector dimension. If `None`, inferred from the first batch of `add_documents()`. |

---

## Persistence

```python
store.save("path/to/index")   # writes index file + sidecar JSON for documents
loaded = FAISSVectorStore.load("path/to/index", embedder=OpenAIEmbedder())
```

`save()` persists both the FAISS index and the parallel `Document` list so search
results can return original text/metadata after reload.

---

## Thread Safety

FAISS itself is not thread-safe for writes. `FAISSVectorStore` wraps every mutation
in a `threading.Lock`, so concurrent `add_documents()` and `search()` calls from
multiple agent threads are safe.

---

## API Reference

| Method | Description |
|---|---|
| `add_documents(docs)` | Embed and add documents to the index |
| `search(query, top_k)` | Cosine similarity search; returns `List[SearchResult]` |
| `delete(ids)` | Remove documents by ID |
| `clear()` | Wipe the index |
| `save(path)` | Persist index + documents to disk |
| `load(path, embedder)` | Class method: rehydrate a persisted store |

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 77 | [`77_faiss_vector_store.py`](https://github.com/johnnichev/selectools/blob/main/examples/77_faiss_vector_store.py) | FAISS quickstart with embeddings + persistence |
