"""
Hybrid Search — BM25 keyword + vector semantic search with RRF/weighted fusion and reranking.

Prerequisites: OPENAI_API_KEY (examples 14-16)
    pip install selectools[rag]
Run: python examples/18_hybrid_search.py
"""

import os
from typing import List, Optional

from selectools import Agent, AgentConfig, Message, OpenAIProvider, Role
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.models import OpenAI
from selectools.rag import (
    BM25,
    Document,
    FusionMethod,
    HybridSearcher,
    HybridSearchTool,
    RAGTool,
    VectorStore,
)

# Sample documents about different topics (tech, science, cooking, etc.)
SAMPLE_DOCUMENTS = [
    Document(
        text="Python is a high-level programming language known for its readability. "
        "It supports multiple programming paradigms including procedural and object-oriented styles.",
        metadata={"topic": "tech", "subject": "programming"},
    ),
    Document(
        text="Machine learning uses statistical techniques to enable computers to learn from data. "
        "Neural networks and deep learning are subsets of machine learning.",
        metadata={"topic": "tech", "subject": "ai"},
    ),
    Document(
        text="Photosynthesis is the process by which plants convert sunlight into chemical energy. "
        "Chlorophyll in chloroplasts absorbs light for this process.",
        metadata={"topic": "science", "subject": "biology"},
    ),
    Document(
        text="Quantum mechanics describes the behavior of matter at atomic scales. "
        "Superposition and entanglement are key quantum phenomena.",
        metadata={"topic": "science", "subject": "physics"},
    ),
    Document(
        text="Italian pasta carbonara traditionally uses eggs, pecorino cheese, guanciale, and black pepper. "
        "Never add cream to an authentic carbonara recipe.",
        metadata={"topic": "cooking", "subject": "italian"},
    ),
    Document(
        text="Sourdough bread requires a fermented starter of flour and water. "
        "The natural yeasts in the starter give sourdough its distinctive tangy flavor.",
        metadata={"topic": "cooking", "subject": "baking"},
    ),
    Document(
        text="The GDPR (General Data Protection Regulation) is an EU law for data privacy. "
        "It requires consent for personal data processing and grants rights to data subjects.",
        metadata={"topic": "legal", "subject": "privacy"},
    ),
    Document(
        text="REST APIs use HTTP methods: GET for retrieval, POST for creation, PUT for updates, DELETE for removal. "
        "Stateless design and resource-based URLs are core REST principles.",
        metadata={"topic": "tech", "subject": "apis"},
    ),
]


def main() -> None:
    """Run the hybrid search demonstration."""

    print("=" * 80)
    print("🔍 Hybrid Search Demo: BM25 + Vector + Reranking")
    print("=" * 80)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return

    # -------------------------------------------------------------------------
    # Step 1: Create sample documents
    # -------------------------------------------------------------------------
    print("\n📄 Step 1: Sample documents")
    print("-" * 80)
    print(
        f"   Created {len(SAMPLE_DOCUMENTS)} documents across topics: tech, science, cooking, legal"
    )
    for i, doc in enumerate(SAMPLE_DOCUMENTS[:3], 1):
        print(f"   [{i}] {doc.text[:60]}...")

    # -------------------------------------------------------------------------
    # Step 2: Set up OpenAI embeddings + in-memory vector store
    # -------------------------------------------------------------------------
    print("\n🔧 Step 2: Setting up embeddings and vector store")
    print("-" * 80)
    embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
    vector_store = VectorStore.create("memory", embedder=embedder)
    vector_store.add_documents(SAMPLE_DOCUMENTS)
    print(f"   ✓ Embedder: {embedder.model} (dimension: {embedder.dimension})")
    print(f"   ✓ Vector store: in-memory with {len(SAMPLE_DOCUMENTS)} documents")

    # -------------------------------------------------------------------------
    # Step 3: Pure semantic search (RAGTool)
    # -------------------------------------------------------------------------
    print("\n🔮 Step 3: Pure semantic search (RAGTool)")
    print("-" * 80)
    rag_tool = RAGTool(vector_store=vector_store, top_k=3, score_threshold=0.0)
    query = "European data privacy law"
    print(f'   Query: "{query}"')
    semantic_results = vector_store.search(embedder.embed_query(query), top_k=3)
    for i, r in enumerate(semantic_results, 1):
        print(f"   [{i}] Score: {r.score:.4f} | {r.document.text[:50]}...")

    # -------------------------------------------------------------------------
    # Step 4: Pure BM25 keyword search
    # -------------------------------------------------------------------------
    print("\n📝 Step 4: Pure BM25 keyword search")
    print("-" * 80)
    bm25 = BM25(remove_stopwords=True)
    bm25.index_documents(SAMPLE_DOCUMENTS)
    bm25_results = bm25.search(query, top_k=3)
    print(f'   Query: "{query}"')
    for i, r in enumerate(bm25_results, 1):
        print(f"   [{i}] Score: {r.score:.4f} | {r.document.text[:50]}...")
    if not bm25_results:
        print("   (No BM25 matches for this conceptual query - keyword search misses paraphrasing)")

    # Query that BM25 excels at: exact terms
    exact_query = "GDPR consent data"
    print(f'\n   Query: "{exact_query}" (exact terms)')
    bm25_exact = bm25.search(exact_query, top_k=3)
    for i, r in enumerate(bm25_exact, 1):
        print(f"   [{i}] Score: {r.score:.4f} | {r.document.text[:50]}...")

    # -------------------------------------------------------------------------
    # Step 5: Hybrid search with RRF fusion
    # -------------------------------------------------------------------------
    print("\n🔄 Step 5: Hybrid search with RRF fusion")
    print("-" * 80)
    searcher_rrf = HybridSearcher(
        vector_store=vector_store,
        fusion=FusionMethod.RRF,
        rrf_k=60,
    )
    searcher_rrf.index_existing_documents(SAMPLE_DOCUMENTS)
    rrf_results = searcher_rrf.search(query, top_k=3)
    print(f'   Query: "{query}"')
    for i, r in enumerate(rrf_results, 1):
        print(f"   [{i}] Score: {r.score:.4f} | {r.document.text[:50]}...")

    # -------------------------------------------------------------------------
    # Step 6: Hybrid search with weighted fusion
    # -------------------------------------------------------------------------
    print("\n⚖️  Step 6: Hybrid search with weighted fusion")
    print("-" * 80)
    store_weighted = VectorStore.create("memory", embedder=embedder)
    store_weighted.add_documents(SAMPLE_DOCUMENTS)
    searcher_weighted = HybridSearcher(
        vector_store=store_weighted,
        fusion=FusionMethod.WEIGHTED,
        vector_weight=0.6,
        keyword_weight=0.4,
    )
    searcher_weighted.index_existing_documents(SAMPLE_DOCUMENTS)
    weighted_results = searcher_weighted.search(query, top_k=3)
    print(f'   Query: "{query}"')
    for i, r in enumerate(weighted_results, 1):
        print(f"   [{i}] Score: {r.score:.4f} | {r.document.text[:50]}...")

    # -------------------------------------------------------------------------
    # Step 7: Hybrid search with reranking (mock CohereReranker)
    # -------------------------------------------------------------------------
    print("\n🏆 Step 7: Hybrid search with reranking")
    print("-" * 80)

    # Use a mock reranker (no Cohere API key needed) - re-orders by keyword boost
    try:
        from selectools.rag import SearchResult
        from selectools.rag.reranker import Reranker

        class MockReranker(Reranker):
            """Mock reranker that boosts documents containing query terms."""

            def rerank(
                self,
                query: str,
                results: List[SearchResult],
                top_k: Optional[int] = None,
            ) -> List[SearchResult]:
                keywords = set(query.lower().split())
                scored = []
                for r in results:
                    text_lower = r.document.text.lower()
                    matches = sum(1 for k in keywords if k in text_lower)
                    boost = 0.2 * matches if matches else 0.0
                    scored.append(SearchResult(document=r.document, score=r.score + boost))
                scored.sort(key=lambda x: x.score, reverse=True)
                return scored[:top_k] if top_k else scored

        mock_reranker = MockReranker()
        store_rerank = VectorStore.create("memory", embedder=embedder)
        store_rerank.add_documents(SAMPLE_DOCUMENTS)
        searcher_rerank = HybridSearcher(
            vector_store=store_rerank,
            fusion=FusionMethod.RRF,
            reranker=mock_reranker,
        )
        searcher_rerank.index_existing_documents(SAMPLE_DOCUMENTS)
        rerank_results = searcher_rerank.search(query, top_k=3)
        print(f'   Query: "{query}" (with mock reranker)')
        for i, r in enumerate(rerank_results, 1):
            print(f"   [{i}] Score: {r.score:.4f} | {r.document.text[:50]}...")
    except Exception as e:
        print(f"   ⚠️  Reranker step skipped: {e}")

    # -------------------------------------------------------------------------
    # Step 8: Compare results side by side
    # -------------------------------------------------------------------------
    print("\n📊 Step 8: Side-by-side comparison")
    print("-" * 80)
    print("   | Semantic (vector) | BM25 (keyword) | Hybrid (RRF) |")
    print("   |-------------------|----------------|--------------|")
    for i in range(3):
        sem = semantic_results[i] if i < len(semantic_results) else None
        kw = bm25_results[i] if i < len(bm25_results) else None
        hy = rrf_results[i] if i < len(rrf_results) else None
        sem_t = (sem.document.text[:25] + "...") if sem else "—"
        kw_t = (kw.document.text[:25] + "...") if kw else "—"
        hy_t = (hy.document.text[:25] + "...") if hy else "—"
        print(f"   | {sem_t:17} | {kw_t:14} | {hy_t:12} |")

    # -------------------------------------------------------------------------
    # Step 9: HybridSearchTool with agent integration
    # -------------------------------------------------------------------------
    print("\n🤖 Step 9: HybridSearchTool with agent integration")
    print("-" * 80)
    hybrid_tool = HybridSearchTool(searcher=searcher_rrf, top_k=3)
    provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)
    agent = Agent(
        tools=[hybrid_tool.search_knowledge_base],
        provider=provider,
        config=AgentConfig(max_iterations=5),
    )
    print("   Agent created with HybridSearchTool (vector + BM25 fusion)")
    test_query = "What does GDPR require for personal data?"
    print(f'   Running agent with: "{test_query}"')
    try:
        response = agent.run([Message(role=Role.USER, content=test_query)])
        preview = (
            response.content[:300] + "..." if len(response.content) > 300 else response.content
        )
        print(f"   Response: {preview}")
    except Exception as e:
        print(f"   ❌ Agent run failed: {e}")

    print("\n" + "=" * 80)
    print("✅ Hybrid Search Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
