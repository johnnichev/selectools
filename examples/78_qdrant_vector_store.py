#!/usr/bin/env python3
"""
Qdrant Vector Store -- production vector search with metadata filtering.

Qdrant is a high-performance vector database with advanced filtering.
This example shows the API pattern (requires a running Qdrant server).

Prerequisites: pip install qdrant-client
Run: python examples/78_qdrant_vector_store.py
"""

print("=== Qdrant Vector Store Example ===\n")

# Usage pattern (requires qdrant-client + running Qdrant server):
print(
    """
from selectools.rag.stores.qdrant import QdrantVectorStore
from selectools.embeddings import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider()
store = QdrantVectorStore(
    embedder=embedder,
    collection_name="my_docs",
    url="http://localhost:6333",  # Qdrant server
)

# Add documents
docs = [Document(text="...", metadata={"category": "tech"})]
store.add_documents(docs)

# Search with metadata filtering
results = store.search(
    embedder.embed_query("search query"),
    top_k=5,
    metadata_filter={"category": "tech"},  # Filter by metadata
)

# Results
for r in results:
    print(f"  {r.score:.2f}: {r.document.text[:50]}")
"""
)

print("Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
print("Install: pip install qdrant-client")
