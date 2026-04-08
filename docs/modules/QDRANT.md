---
description: "Connector for the Qdrant vector database with REST + gRPC support and payload filtering"
tags:
  - rag
  - vector-stores
  - qdrant
---

# Qdrant Vector Store

**Import:** `from selectools.rag.stores import QdrantVectorStore`
**Stability:** beta
**Added in:** v0.21.0

`QdrantVectorStore` wraps the official `qdrant-client` to give you a self-hosted or
Qdrant Cloud-backed vector store. It auto-creates collections, supports cosine
similarity by default, and lets you filter searches on metadata via Qdrant's payload
indexing.

```python title="qdrant_quick.py"
from selectools.embeddings import OpenAIEmbedder
from selectools.rag import Document
from selectools.rag.stores import QdrantVectorStore

store = QdrantVectorStore(
    embedder=OpenAIEmbedder(),
    collection_name="my_docs",
    url="http://localhost:6333",
)

store.add_documents([
    Document(text="Qdrant is a vector search engine.", metadata={"category": "infra"}),
    Document(text="It supports REST and gRPC.", metadata={"category": "infra"}),
])

results = store.search("vector search", top_k=2)
```

!!! tip "See Also"
    - [FAISS](FAISS.md) - In-process vector index, no server required
    - [pgvector](PGVECTOR.md) - PostgreSQL-backed vector store
    - [RAG](RAG.md) - Higher-level retrieval pipeline

---

## Install

```bash
pip install "selectools[rag]"
```

`qdrant-client>=1.7.0` is part of the `[rag]` extras.

You also need a running Qdrant instance. The simplest way:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Or sign up for [Qdrant Cloud](https://cloud.qdrant.io/) and get a managed instance.

---

## Constructor

```python
QdrantVectorStore(
    embedder: EmbeddingProvider,
    collection_name: str = "selectools",
    url: str = "http://localhost:6333",
    api_key: str | None = None,
    prefer_grpc: bool = True,
    **qdrant_kwargs,
)
```

| Parameter | Description |
|---|---|
| `embedder` | Any `EmbeddingProvider`. Used to compute vectors for both `add_documents()` and `search()`. |
| `collection_name` | Qdrant collection. Auto-created on first `add_documents()` if it doesn't exist. |
| `url` | Qdrant server URL. Use `https://...` for cloud. |
| `api_key` | Optional API key for Qdrant Cloud or authenticated servers. |
| `prefer_grpc` | When `True` (default) the client uses gRPC for lower-latency vector ops. |
| `**qdrant_kwargs` | Additional arguments forwarded to `qdrant_client.QdrantClient`. |

---

## Cloud Configuration

```python
import os

store = QdrantVectorStore(
    embedder=OpenAIEmbedder(),
    collection_name="prod_docs",
    url="https://my-cluster.qdrant.io",
    api_key=os.environ["QDRANT_API_KEY"],
)
```

---

## Metadata Filtering

Document metadata is stored as Qdrant payload, so you can filter searches at the
database level. Use `qdrant_client.models.Filter` constructs and pass them via
`**search_kwargs` (the store forwards them to the underlying client).

---

## API Reference

| Method | Description |
|---|---|
| `add_documents(docs)` | Embed documents and upsert into the collection |
| `search(query, top_k)` | Cosine similarity search |
| `delete(ids)` | Delete documents by ID |
| `clear()` | Delete the entire collection |

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 78 | [`78_qdrant_vector_store.py`](https://github.com/johnnichev/selectools/blob/main/examples/78_qdrant_vector_store.py) | Qdrant quickstart with metadata filtering |
