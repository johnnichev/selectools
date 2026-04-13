"""
Multi-Tenant RAG with Permission Filters — safe metadata filtering.

Since v0.22.0 (BUG-25), in-memory and BM25 stores raise NotImplementedError
when you pass operator-syntax filters ({$in: [...]}) instead of silently
returning wrong results. Use backend stores (Chroma, Qdrant, Pinecone) for
operator support, or use equality filters for in-memory/BM25.

Also demonstrates citation-preserving dedup (BUG-24): documents with
identical text but different sources are preserved as distinct citations.

Prerequisites: No API key needed (uses numpy embeddings)
Run: python examples/93_multi_tenant_rag.py
"""

from unittest.mock import MagicMock

import numpy as np

from selectools.rag.bm25 import BM25
from selectools.rag.stores.memory import InMemoryVectorStore
from selectools.rag.vector_store import Document


def _mock_embedder():
    """Create a mock embedder for demonstration."""
    embedder = MagicMock()
    rng = np.random.RandomState(42)
    embedder.embed_query.return_value = rng.randn(8).astype(np.float32)
    embedder.embed_texts.side_effect = lambda texts: rng.randn(len(texts), 8).astype(np.float32)
    return embedder


def main() -> None:
    embedder = _mock_embedder()
    store = InMemoryVectorStore(embedder=embedder)
    bm25 = BM25()

    # Add multi-tenant documents
    docs = [
        Document(text="Q4 revenue was $10M", metadata={"tenant": "acme", "source": "10-K.pdf"}),
        Document(text="Q4 revenue was $10M", metadata={"tenant": "globex", "source": "annual.pdf"}),
        Document(text="Hiring plan for 2025", metadata={"tenant": "acme", "source": "hr.pdf"}),
    ]
    store.add_documents(docs)
    bm25.add_documents(docs)

    # 1. Equality filter works everywhere
    query_emb = embedder.embed_query("revenue")
    results = store.search(query_emb, top_k=10, filter={"tenant": "acme"})
    print(f"Equality filter (tenant=acme): {len(results)} results")
    for r in results:
        print(f"  {r.document.metadata['source']}: {r.document.text[:40]}")

    # 2. Operator-syntax filters raise NotImplementedError (not silently wrong)
    print("\nOperator-syntax filter on in-memory store:")
    try:
        store.search(query_emb, filter={"tenant": {"$in": ["acme", "globex"]}})
    except NotImplementedError as e:
        print(f"  Caught: {e}")

    # 3. BM25 same behavior
    print("\nOperator-syntax filter on BM25:")
    try:
        bm25.search("revenue", filter={"tenant": {"$in": ["acme", "globex"]}})
    except NotImplementedError as e:
        print(f"  Caught: {e}")

    # 4. Citation-preserving dedup
    results = store.search(query_emb, top_k=10, dedup=True)
    print(f"\nDedup search: {len(results)} results (same text, different sources preserved)")
    for r in results:
        print(f"  {r.document.metadata.get('source', 'unknown')}: {r.document.text[:40]}")

    print("\n✓ Filters are safe — no silent permission bypass")
    print("✓ Dedup preserves citations from different sources")


if __name__ == "__main__":
    main()
