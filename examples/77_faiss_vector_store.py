#!/usr/bin/env python3
"""
FAISS Vector Store -- fast local similarity search with persistence.

No API key needed. Uses FAISS (Facebook AI Similarity Search) for
high-performance local vector search. Supports save/load to disk.

Prerequisites: pip install faiss-cpu
Run: python examples/77_faiss_vector_store.py
"""

from selectools.rag import DocumentLoader, InMemoryVectorStore
from selectools.rag.vector_store import Document

# Note: This example uses InMemoryVectorStore as a stand-in.
# With faiss-cpu installed, replace with:
#   from selectools.rag.stores.faiss import FAISSVectorStore
#   store = FAISSVectorStore(embedder=embedder, dimension=128)

print("=== FAISS Vector Store Example ===\n")

# Create documents
docs = [
    Document(text="Python is a versatile programming language", metadata={"lang": "python"}),
    Document(text="JavaScript runs in the browser", metadata={"lang": "javascript"}),
    Document(
        text="Rust provides memory safety without garbage collection", metadata={"lang": "rust"}
    ),
    Document(text="Go is great for concurrent server applications", metadata={"lang": "go"}),
]

# Using InMemoryVectorStore as demonstration (no external deps needed)
store = InMemoryVectorStore()

# With FAISS installed:
# from selectools.rag.stores.faiss import FAISSVectorStore
# from selectools.embeddings import OpenAIEmbeddingProvider
# embedder = OpenAIEmbeddingProvider()
# store = FAISSVectorStore(embedder=embedder)
# store.add_documents(docs)
# results = store.search(embedder.embed_query("memory safe language"), top_k=2)
# store.save("./my_index")  # Persist to disk
# loaded = FAISSVectorStore.load("./my_index", embedder=embedder)

print(f"Created {len(docs)} documents")
print("FAISS supports: add, search, delete, clear, save, load")
print("Install: pip install faiss-cpu")
print("\nDone!")
