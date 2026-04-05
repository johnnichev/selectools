#!/usr/bin/env python3
"""
pgvector Store -- PostgreSQL-native vector search.

Use your existing PostgreSQL database for vector similarity search.
No additional database infrastructure needed.

Prerequisites: pip install psycopg2-binary, PostgreSQL with pgvector extension
Run: python examples/79_pgvector_store.py
"""

print("=== pgvector Store Example ===\n")

print(
    """
from selectools.rag.stores.pgvector import PgVectorStore
from selectools.embeddings import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider()
store = PgVectorStore(
    embedder=embedder,
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    table_name="document_embeddings",
)

# Add documents (auto-creates table + HNSW index on first use)
docs = [Document(text="...", metadata={"source": "api"})]
store.add_documents(docs)

# Search with cosine similarity
results = store.search(
    embedder.embed_query("search query"),
    top_k=5,
)

# All queries are parameterized (SQL injection safe)
# Table schema: id TEXT PK, text TEXT, metadata JSONB, embedding vector(N)
"""
)

print("Install pgvector: CREATE EXTENSION vector;")
print("Install driver: pip install psycopg2-binary")
