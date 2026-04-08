---
description: "PostgreSQL-backed vector store using the pgvector extension"
tags:
  - rag
  - vector-stores
  - postgres
  - pgvector
---

# pgvector Store

**Import:** `from selectools.rag.stores import PgVectorStore`
**Stability:** beta
**Added in:** v0.21.0

`PgVectorStore` lets you store and search document embeddings inside a PostgreSQL
database using the [pgvector](https://github.com/pgvector/pgvector) extension. It's
the right choice when you already run Postgres and want vectors next to the rest of
your application data without standing up a separate vector service.

```python title="pgvector_quick.py"
from selectools.embeddings import OpenAIEmbedder
from selectools.rag import Document
from selectools.rag.stores import PgVectorStore

store = PgVectorStore(
    embedder=OpenAIEmbedder(),
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    table_name="selectools_documents",
)

store.add_documents([
    Document(text="pgvector adds vector types to Postgres."),
    Document(text="It supports cosine, L2, and inner-product distance."),
])

results = store.search("postgres vector search", top_k=2)
```

!!! tip "See Also"
    - [Qdrant](QDRANT.md) - Self-hosted vector database with REST + gRPC
    - [FAISS](FAISS.md) - In-process vector index, no server required
    - [Sessions](SESSIONS.md) - Postgres-backed agent sessions

---

## Install

```bash
pip install "selectools[postgres]"
```

The `[postgres]` extras already include `psycopg2-binary>=2.9.0`. You also need
the pgvector extension installed in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Constructor

```python
PgVectorStore(
    embedder: EmbeddingProvider,
    connection_string: str,
    table_name: str = "selectools_documents",
    dimensions: int | None = None,
)
```

| Parameter | Description |
|---|---|
| `embedder` | Embedding provider used to compute vectors. |
| `connection_string` | Standard libpq connection string. |
| `table_name` | Table to store documents in. Validated as a SQL identifier (letters, digits, underscores) to prevent injection. |
| `dimensions` | Vector dimensions. Auto-detected from `embedder.embed_query("test")` on first use if not specified. |

---

## Schema

`PgVectorStore` creates the following table on first use (idempotent):

```sql
CREATE TABLE IF NOT EXISTS selectools_documents (
    id        TEXT PRIMARY KEY,
    text      TEXT NOT NULL,
    metadata  JSONB,
    embedding vector(N)
);
```

The `N` is the embedding dimension. An index on the `embedding` column accelerates
cosine similarity queries.

---

## Search

`search()` runs a parameterized query using pgvector's `<=>` cosine distance
operator:

```sql
SELECT id, text, metadata, embedding <=> %s AS distance
FROM selectools_documents
ORDER BY distance ASC
LIMIT %s;
```

All queries are parameterized — there's no SQL injection risk from user input.

---

## Connection Pooling

`PgVectorStore` opens a single `psycopg2.connect()` per instance. If you need
pooling for high concurrency, manage it externally (e.g. PgBouncer) and pass the
pooler URL as the connection string.

---

## API Reference

| Method | Description |
|---|---|
| `add_documents(docs)` | Embed and upsert documents (`INSERT ... ON CONFLICT DO UPDATE`) |
| `search(query, top_k)` | Cosine similarity search |
| `delete(ids)` | Delete documents by ID |
| `clear()` | `TRUNCATE` the table |

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 79 | [`79_pgvector_store.py`](https://github.com/johnnichev/selectools/blob/main/examples/79_pgvector_store.py) | pgvector quickstart with auto-table creation |
