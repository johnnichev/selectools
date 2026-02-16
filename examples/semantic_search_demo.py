"""
Semantic Search Demo: Pure Embedding-Based Search

This example demonstrates:
- Using SemanticSearchTool for pure semantic search (no LLM)
- Comparing different embedding providers
- Analyzing similarity scores
- Building a simple search interface
- Performance and cost comparison
"""

import os
import time
from typing import Any, Dict, List

from selectools.embeddings import GeminiEmbeddingProvider, OpenAIEmbeddingProvider
from selectools.models import Gemini, OpenAI
from selectools.rag import Document, SemanticSearchTool, VectorStore

# Sample knowledge base: Programming concepts
SAMPLE_DOCUMENTS = [
    Document(
        text="Python is a high-level, interpreted programming language with dynamic semantics. "
        "Its high-level built-in data structures, combined with dynamic typing and binding, "
        "make it very attractive for Rapid Application Development.",
        metadata={"category": "languages", "topic": "python", "difficulty": "beginner"},
    ),
    Document(
        text="JavaScript is a lightweight, interpreted programming language with first-class functions. "
        "It is most well-known as the scripting language for Web pages, but it's also used in "
        "many non-browser environments like Node.js.",
        metadata={"category": "languages", "topic": "javascript", "difficulty": "beginner"},
    ),
    Document(
        text="Machine Learning is a subset of artificial intelligence that provides systems the ability "
        "to automatically learn and improve from experience without being explicitly programmed. "
        "It focuses on the development of computer programs that can access data and use it to learn.",
        metadata={"category": "ai", "topic": "machine-learning", "difficulty": "intermediate"},
    ),
    Document(
        text="Neural Networks are computing systems inspired by biological neural networks that "
        "constitute animal brains. They consist of interconnected nodes (neurons) organized in layers. "
        "Deep learning uses neural networks with multiple layers.",
        metadata={"category": "ai", "topic": "neural-networks", "difficulty": "advanced"},
    ),
    Document(
        text="Docker is a platform that uses OS-level virtualization to deliver software in containers. "
        "Containers are lightweight, standalone packages that include everything needed to run "
        "an application: code, runtime, system tools, libraries, and settings.",
        metadata={"category": "devops", "topic": "docker", "difficulty": "intermediate"},
    ),
    Document(
        text="Kubernetes is an open-source container orchestration platform that automates deployment, "
        "scaling, and management of containerized applications. It groups containers into logical "
        "units for easy management and discovery.",
        metadata={"category": "devops", "topic": "kubernetes", "difficulty": "advanced"},
    ),
    Document(
        text="SQL (Structured Query Language) is a domain-specific language used for managing and "
        "manipulating relational databases. It's used for tasks like querying data, updating records, "
        "creating and modifying tables, and setting permissions.",
        metadata={"category": "databases", "topic": "sql", "difficulty": "beginner"},
    ),
    Document(
        text="NoSQL databases provide a mechanism for storage and retrieval of data that is modeled "
        "differently than tabular relations in relational databases. They're often used for big data "
        "and real-time web applications due to their flexibility and scalability.",
        metadata={"category": "databases", "topic": "nosql", "difficulty": "intermediate"},
    ),
    Document(
        text="RESTful APIs are application programming interfaces that conform to REST architectural "
        "constraints. REST stands for Representational State Transfer and uses HTTP requests to "
        "GET, PUT, POST and DELETE data.",
        metadata={"category": "web", "topic": "rest-api", "difficulty": "intermediate"},
    ),
    Document(
        text="GraphQL is a query language for APIs and a runtime for fulfilling those queries with "
        "existing data. It provides a complete and understandable description of the data in your API "
        "and gives clients the power to ask for exactly what they need.",
        metadata={"category": "web", "topic": "graphql", "difficulty": "intermediate"},
    ),
]


def setup_vector_store(embedder: Any, documents: List[Document]) -> VectorStore:
    """Set up a vector store with the given embedder and documents."""
    vector_store = VectorStore.create("memory", embedder=embedder)
    vector_store.add_documents(documents)
    return vector_store


def run_search_comparison(queries: List[str]) -> None:
    """Compare semantic search across different embedding providers."""

    print("=" * 100)
    print("Semantic Search Demo: Comparing Embedding Providers")
    print("=" * 100)

    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    if not has_openai and not has_gemini:
        print("\n❌ Please set at least one API key:")
        print("   export OPENAI_API_KEY='your-key'  (paid)")
        print("   export GEMINI_API_KEY='your-key'   (free)")
        return

    print(f"\n📊 Knowledge Base: {len(SAMPLE_DOCUMENTS)} documents")
    print(f"   Categories: {len(set(d.metadata['category'] for d in SAMPLE_DOCUMENTS))}")
    print(f"   Topics: {len(set(d.metadata['topic'] for d in SAMPLE_DOCUMENTS))}")

    # -------------------------------------------------------------------------
    # Set up embedding providers
    # -------------------------------------------------------------------------
    providers_info = []

    if has_openai:
        print("\n🔧 Setting up OpenAI embeddings...")
        openai_embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
        openai_store = setup_vector_store(openai_embedder, SAMPLE_DOCUMENTS)
        openai_tool = SemanticSearchTool(vector_store=openai_store, top_k=3, score_threshold=0.5)
        providers_info.append(
            {
                "name": "OpenAI (text-embedding-3-small)",
                "embedder": openai_embedder,
                "tool": openai_tool,
                "cost_per_1m": "$0.02",
                "dimension": openai_embedder.dimension,
            }
        )
        print(f"   ✓ Model: {openai_embedder.model}")
        print(f"   ✓ Dimension: {openai_embedder.dimension}")

    if has_gemini:
        print("\n🔧 Setting up Gemini embeddings...")
        gemini_embedder = GeminiEmbeddingProvider(model=Gemini.Embeddings.EMBEDDING_001.id)
        gemini_store = setup_vector_store(gemini_embedder, SAMPLE_DOCUMENTS)
        gemini_tool = SemanticSearchTool(vector_store=gemini_store, top_k=3, score_threshold=0.5)
        providers_info.append(
            {
                "name": "Gemini (text-embedding-001)",
                "embedder": gemini_embedder,
                "tool": gemini_tool,
                "cost_per_1m": "$0.00 (FREE)",
                "dimension": gemini_embedder.dimension,
            }
        )
        print(f"   ✓ Model: {gemini_embedder.model}")
        print(f"   ✓ Dimension: {gemini_embedder.dimension}")

    # -------------------------------------------------------------------------
    # Run semantic searches
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("🔍 Running Semantic Searches")
    print("=" * 100)

    for query in queries:
        print(f"\n{'=' * 100}")
        print(f'Query: "{query}"')
        print("=" * 100)

        for provider in providers_info:
            print(f"\n[{provider['name']}]")
            print(
                f"Cost: {provider['cost_per_1m']} per 1M tokens | Dimension: {provider['dimension']}"
            )
            print("-" * 100)

            # Measure search time
            start_time = time.time()
            results = provider["tool"].search(query)
            elapsed_time = time.time() - start_time

            if not results:
                print("   ❌ No results found above threshold (0.5)")
            else:
                for i, result in enumerate(results, 1):
                    score = result.score
                    text = result.document.text
                    metadata = result.document.metadata

                    # Determine score quality
                    if score >= 0.8:
                        score_indicator = "🟢 Excellent"
                    elif score >= 0.7:
                        score_indicator = "🟡 Good"
                    elif score >= 0.6:
                        score_indicator = "🟠 Fair"
                    else:
                        score_indicator = "🔴 Weak"

                    print(f"\n   [{i}] Score: {score:.3f} {score_indicator}")
                    print(f"       Category: {metadata.get('category', 'N/A')}")
                    print(f"       Topic: {metadata.get('topic', 'N/A')}")
                    print(f"       Difficulty: {metadata.get('difficulty', 'N/A')}")
                    print(
                        f"       Text: {text[:150]}..."
                        if len(text) > 150
                        else f"       Text: {text}"
                    )

            print(f"\n   ⏱️  Search time: {elapsed_time*1000:.2f}ms")

    # -------------------------------------------------------------------------
    # Performance comparison
    # -------------------------------------------------------------------------
    if len(providers_info) > 1:
        print("\n" + "=" * 100)
        print("📈 Performance Comparison")
        print("=" * 100)

        print("\n| Provider | Cost/1M tokens | Dimension | Best For |")
        print("|----------|----------------|-----------|----------|")
        for provider in providers_info:
            if "OpenAI" in provider["name"]:
                best_for = "Production, high accuracy"
            elif "Gemini" in provider["name"]:
                best_for = "Development, cost-sensitive"
            else:
                best_for = "Various use cases"

            print(
                f"| {provider['name']:30} | {provider['cost_per_1m']:14} | {provider['dimension']:9} | {best_for} |"
            )

    # -------------------------------------------------------------------------
    # Tips and recommendations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("💡 Recommendations")
    print("=" * 100)

    print(
        """
1. **Choosing an Embedding Model:**
   - OpenAI (text-embedding-3-small): Great balance of quality and cost
   - OpenAI (text-embedding-3-large): Highest quality, higher cost
   - Gemini: Free tier, good for development and testing
   - Cohere: Specialized models for search vs classification

2. **Score Thresholds:**
   - 0.8+: Highly relevant, safe to use
   - 0.7-0.8: Good relevance, usually accurate
   - 0.6-0.7: Moderate relevance, may need verification
   - <0.6: Weak relevance, consider excluding

3. **Performance Tips:**
   - Cache embeddings to avoid recomputing
   - Use batch operations for multiple queries
   - Consider lower-dimensional models for speed
   - Use metadata filters to narrow search space

4. **Cost Optimization:**
   - Start with Gemini (free) for prototyping
   - Use text-embedding-3-small for production
   - Cache frequently used embeddings
   - Monitor token usage with selectools' built-in tracking
    """
    )


def demonstrate_metadata_filtering() -> None:
    """Demonstrate how to use metadata filters in semantic search."""

    print("\n" + "=" * 100)
    print("🎯 Demonstrating Metadata Filtering")
    print("=" * 100)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipping metadata filtering demo (requires OPENAI_API_KEY)")
        return

    # Set up vector store
    embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
    vector_store = setup_vector_store(embedder, SAMPLE_DOCUMENTS)

    query = "Tell me about programming languages"
    query_embedding = embedder.embed_query(query)

    print(f'\nQuery: "{query}"\n')

    # Search without filter
    print("[1] Unfiltered search (all documents):")
    all_results = vector_store.search(query_embedding, top_k=3)
    for i, result in enumerate(all_results, 1):
        category = result.document.metadata.get("category", "N/A")
        topic = result.document.metadata.get("topic", "N/A")
        print(f"   {i}. Score: {result.score:.3f} | Category: {category} | Topic: {topic}")

    # Search with category filter
    print("\n[2] Filtered search (category = 'languages'):")
    lang_results = vector_store.search(query_embedding, top_k=3, filter={"category": "languages"})
    for i, result in enumerate(lang_results, 1):
        category = result.document.metadata.get("category", "N/A")
        topic = result.document.metadata.get("topic", "N/A")
        print(f"   {i}. Score: {result.score:.3f} | Category: {category} | Topic: {topic}")

    # Search with difficulty filter
    print("\n[3] Filtered search (difficulty = 'beginner'):")
    beginner_results = vector_store.search(
        query_embedding, top_k=3, filter={"difficulty": "beginner"}
    )
    for i, result in enumerate(beginner_results, 1):
        category = result.document.metadata.get("category", "N/A")
        difficulty = result.document.metadata.get("difficulty", "N/A")
        print(
            f"   {i}. Score: {result.score:.3f} | Category: {category} | Difficulty: {difficulty}"
        )

    print("\n💡 Metadata filtering allows you to:")
    print("   - Narrow search to specific document types")
    print("   - Filter by date, author, category, etc.")
    print("   - Combine semantic search with structured filters")


def main() -> None:
    """Run all demonstrations."""

    # Test queries covering different topics
    queries = [
        "What is machine learning?",
        "How do I deploy applications in containers?",
        "What's the difference between SQL and NoSQL?",
        "Tell me about web APIs",
    ]

    # Run comparison
    run_search_comparison(queries)

    # Demonstrate metadata filtering
    demonstrate_metadata_filtering()

    print("\n" + "=" * 100)
    print("✅ Demo Complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
