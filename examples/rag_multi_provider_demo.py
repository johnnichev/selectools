"""
Multi-Provider RAG Comparison Demo

This example demonstrates:
- Comparing OpenAI vs Gemini embeddings
- In-memory vs SQLite performance comparison
- Impact of different chunk sizes
- Top-k parameter tuning
- Comprehensive cost analysis
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from selectools import Agent, AgentConfig, OpenAIProvider
from selectools.embeddings import GeminiEmbeddingProvider, OpenAIEmbeddingProvider
from selectools.models import Gemini, OpenAI
from selectools.rag import Document, DocumentLoader, RAGAgent, TextSplitter, VectorStore

# Sample technical documentation
SAMPLE_DOCS_CONTENT = {
    "intro.txt": """
Selectools: Production-Ready AI Agent Framework

Selectools is a comprehensive Python framework for building AI agents with tool-calling capabilities.
It provides a unified interface for multiple LLM providers, advanced features like conversation memory,
cost tracking, and now includes powerful RAG (Retrieval-Augmented Generation) capabilities.

Key benefits:
- Multi-provider support (OpenAI, Anthropic, Gemini, Ollama)
- Type-safe model selection with IDE autocomplete
- Automatic cost tracking for both LLM and embedding API calls
- Production-ready error handling and validation
    """,
    "architecture.txt": """
Architecture Overview

The framework is built on three main layers:

1. Provider Layer
   - Abstracts different LLM APIs
   - Handles authentication and rate limiting
   - Supports streaming and async operations

2. Agent Layer
   - Manages conversation state and memory
   - Orchestrates tool execution
   - Tracks usage and costs
   - Provides observability hooks

3. Tool Layer
   - Defines callable functions with schemas
   - Validates parameters at registration
   - Supports sync, async, and streaming tools
   - Includes pre-built tool library
    """,
    "rag_features.txt": """
RAG Features (v0.8.0)

Embedding Providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large, ada-002)
- Anthropic via Voyage AI (voyage-3, voyage-3-lite)
- Google Gemini (text-embedding-001, text-embedding-004) - FREE
- Cohere (embed-english-v3.0, embed-multilingual-v3.0)

Vector Stores:
- In-memory: Fast, numpy-based, great for prototyping
- SQLite: Persistent local storage, no external dependencies
- Chroma: Feature-rich, supports metadata filtering
- Pinecone: Cloud-based, highly scalable

Document Processing:
- Load from text, files, directories, PDFs
- Intelligent chunking strategies
- Metadata preservation and filtering
- Batch embedding for efficiency
    """,
    "cost_guide.txt": """
Cost Management

Selectools provides transparent cost tracking:

LLM Costs:
- Automatic token counting
- Real-time cost estimation
- Per-tool usage breakdown
- Configurable cost warnings

Embedding Costs:
- Tracked separately from LLM costs
- Batch operations for efficiency
- Free options available (Gemini)

Cost Optimization Tips:
1. Use GPT-4o-mini for most tasks ($0.15/$0.60 per 1M tokens)
2. Use Gemini embeddings for development (FREE)
3. Cache embeddings to avoid recomputation
4. Use appropriate chunk sizes (500-1000 chars)
5. Set cost warning thresholds
    """,
}


def create_sample_docs(directory: Path) -> List[str]:
    """Create sample documents in the given directory."""
    directory.mkdir(parents=True, exist_ok=True)

    file_paths = []
    for filename, content in SAMPLE_DOCS_CONTENT.items():
        file_path = directory / filename
        file_path.write_text(content.strip())
        file_paths.append(str(file_path))

    return file_paths


def test_configuration(
    config_name: str,
    embedder,
    vector_store_type: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    documents_dir: Path,
    query: str,
) -> Dict[str, Any]:
    """Test a specific RAG configuration and return metrics."""

    print(f"\n{'=' * 100}")
    print(f"Testing Configuration: {config_name}")
    print("=" * 100)
    print(f"  Embedder: {embedder.model}")
    print(f"  Vector Store: {vector_store_type}")
    print(f"  Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"  Top-K: {top_k}")

    # Measure setup time
    setup_start = time.time()

    # Create vector store
    if vector_store_type == "sqlite":
        db_path = f"temp_{config_name.replace(' ', '_').lower()}.db"
        vector_store = VectorStore.create("sqlite", embedder=embedder, db_path=db_path)
        vector_store.clear()  # Clear any existing data
    else:
        vector_store = VectorStore.create("memory", embedder=embedder)

    # Load documents
    loader = DocumentLoader()
    documents = loader.from_directory(str(documents_dir), glob_pattern="*.txt")

    # Chunk documents
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = splitter.split_documents(documents)

    # Add to vector store
    vector_store.add_documents(chunked_docs)

    setup_time = time.time() - setup_start

    # Measure query time
    query_start = time.time()
    query_embedding = embedder.embed_query(query)
    results = vector_store.search(query_embedding, top_k=top_k)
    query_time = time.time() - query_start

    # Analyze results
    avg_score = sum(r.score for r in results) / len(results) if results else 0.0

    print(f"\n📊 Results:")
    print(f"  Setup time: {setup_time*1000:.2f}ms")
    print(f"  Query time: {query_time*1000:.2f}ms")
    print(f"  Documents loaded: {len(documents)}")
    print(f"  Chunks created: {len(chunked_docs)}")
    print(f"  Results returned: {len(results)}")
    print(f"  Average score: {avg_score:.3f}")

    if results:
        print(f"\n  Top result:")
        print(f"    Score: {results[0].score:.3f}")
        print(f"    Text preview: {results[0].document.text[:100]}...")

    # Clean up SQLite database
    if vector_store_type == "sqlite":
        import os as os_module

        try:
            os_module.remove(db_path)
        except Exception:
            pass

    return {
        "config_name": config_name,
        "embedder_model": embedder.model,
        "vector_store": vector_store_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "setup_time_ms": setup_time * 1000,
        "query_time_ms": query_time * 1000,
        "num_documents": len(documents),
        "num_chunks": len(chunked_docs),
        "num_results": len(results),
        "avg_score": avg_score,
        "top_score": results[0].score if results else 0.0,
    }


def compare_embedding_providers():
    """Compare different embedding providers."""

    print("\n" + "=" * 100)
    print("PART 1: Embedding Provider Comparison")
    print("=" * 100)

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    if not has_openai and not has_gemini:
        print("\n❌ Please set at least one API key:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export GEMINI_API_KEY='your-key'")
        return []

    results = []
    query = "What are the main features of RAG in Selectools?"

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        create_sample_docs(docs_dir)

        # Test OpenAI
        if has_openai:
            openai_embedder = OpenAIEmbeddingProvider(
                model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id
            )
            result = test_configuration(
                config_name="OpenAI Small",
                embedder=openai_embedder,
                vector_store_type="memory",
                chunk_size=500,
                chunk_overlap=100,
                top_k=3,
                documents_dir=docs_dir,
                query=query,
            )
            results.append(result)

        # Test Gemini
        if has_gemini:
            gemini_embedder = GeminiEmbeddingProvider(model=Gemini.Embeddings.EMBEDDING_001.id)
            result = test_configuration(
                config_name="Gemini 001",
                embedder=gemini_embedder,
                vector_store_type="memory",
                chunk_size=500,
                chunk_overlap=100,
                top_k=3,
                documents_dir=docs_dir,
                query=query,
            )
            results.append(result)

    return results


def compare_vector_stores():
    """Compare different vector store backends."""

    print("\n" + "=" * 100)
    print("PART 2: Vector Store Comparison")
    print("=" * 100)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipping vector store comparison (requires OPENAI_API_KEY)")
        return []

    results = []
    query = "How does cost tracking work?"
    embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        create_sample_docs(docs_dir)

        # Test in-memory
        result = test_configuration(
            config_name="In-Memory Store",
            embedder=embedder,
            vector_store_type="memory",
            chunk_size=500,
            chunk_overlap=100,
            top_k=3,
            documents_dir=docs_dir,
            query=query,
        )
        results.append(result)

        # Test SQLite
        result = test_configuration(
            config_name="SQLite Store",
            embedder=embedder,
            vector_store_type="sqlite",
            chunk_size=500,
            chunk_overlap=100,
            top_k=3,
            documents_dir=docs_dir,
            query=query,
        )
        results.append(result)

    return results


def compare_chunk_sizes():
    """Compare different chunk size configurations."""

    print("\n" + "=" * 100)
    print("PART 3: Chunk Size Comparison")
    print("=" * 100)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipping chunk size comparison (requires OPENAI_API_KEY)")
        return []

    results = []
    query = "What embedding providers are supported?"
    embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        create_sample_docs(docs_dir)

        # Test different chunk sizes
        for chunk_size in [300, 500, 1000]:
            overlap = chunk_size // 5  # 20% overlap
            result = test_configuration(
                config_name=f"Chunk {chunk_size}",
                embedder=embedder,
                vector_store_type="memory",
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                top_k=3,
                documents_dir=docs_dir,
                query=query,
            )
            results.append(result)

    return results


def compare_top_k():
    """Compare different top-k values."""

    print("\n" + "=" * 100)
    print("PART 4: Top-K Comparison")
    print("=" * 100)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipping top-k comparison (requires OPENAI_API_KEY)")
        return []

    results = []
    query = "What is the architecture of Selectools?"
    embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        create_sample_docs(docs_dir)

        # Test different top-k values
        for top_k in [1, 3, 5]:
            result = test_configuration(
                config_name=f"Top-K {top_k}",
                embedder=embedder,
                vector_store_type="memory",
                chunk_size=500,
                chunk_overlap=100,
                top_k=top_k,
                documents_dir=docs_dir,
                query=query,
            )
            results.append(result)

    return results


def print_summary_table(all_results: List[Dict[str, Any]]):
    """Print a summary table of all results."""

    if not all_results:
        return

    print("\n" + "=" * 100)
    print("SUMMARY: Performance & Quality Comparison")
    print("=" * 100)

    print("\n| Configuration | Setup (ms) | Query (ms) | Chunks | Avg Score | Top Score |")
    print("|---------------|------------|------------|--------|-----------|-----------|")

    for result in all_results:
        print(
            f"| {result['config_name']:13} | "
            f"{result['setup_time_ms']:10.2f} | "
            f"{result['query_time_ms']:10.2f} | "
            f"{result['num_chunks']:6} | "
            f"{result['avg_score']:9.3f} | "
            f"{result['top_score']:9.3f} |"
        )

    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    print(
        """
1. **Embedding Provider Selection:**
   - OpenAI: Higher quality, costs $0.02/1M tokens
   - Gemini: FREE, good quality for most use cases
   - Choose based on budget and quality requirements

2. **Vector Store Selection:**
   - In-Memory: Fastest, no persistence
   - SQLite: Good balance, persistent local storage
   - Chroma/Pinecone: For production at scale

3. **Chunk Size Impact:**
   - Smaller chunks (300-500): More precise, more chunks to search
   - Larger chunks (1000+): More context, fewer chunks
   - Optimal: 500-1000 chars with 20% overlap

4. **Top-K Tuning:**
   - Start with k=3 for most use cases
   - Increase for broader context
   - Decrease for more focused results

5. **Performance Tips:**
   - Cache embeddings when possible
   - Use batch operations
   - Consider async for I/O-bound operations
   - Monitor costs with built-in tracking
    """
    )


def cost_analysis():
    """Provide detailed cost analysis."""

    print("\n" + "=" * 100)
    print("COST ANALYSIS")
    print("=" * 100)

    print(
        """
Embedding Costs (per 1M tokens):
┌────────────────────────────────┬──────────────┐
│ Provider / Model               │ Cost         │
├────────────────────────────────┼──────────────┤
│ OpenAI text-embedding-3-small  │ $0.02        │
│ OpenAI text-embedding-3-large  │ $0.13        │
│ OpenAI text-embedding-ada-002  │ $0.10        │
│ Gemini text-embedding-001      │ FREE         │
│ Gemini text-embedding-004      │ FREE         │
│ Voyage AI voyage-3             │ $0.06        │
│ Voyage AI voyage-3-lite        │ $0.02        │
│ Cohere embed-v3                │ $0.10        │
└────────────────────────────────┴──────────────┘

Example Cost Calculation:
- 1000 documents × 500 chars avg = 500,000 chars
- ~125,000 tokens (assuming 4 chars/token)
- Using OpenAI text-embedding-3-small: $0.0025
- Using Gemini: FREE

LLM Costs (per 1M tokens):
┌────────────────────────────────┬──────────────┬──────────────┐
│ Model                          │ Input        │ Output       │
├────────────────────────────────┼──────────────┼──────────────┤
│ GPT-4o                         │ $2.50        │ $10.00       │
│ GPT-4o-mini                    │ $0.15        │ $0.60        │
│ Claude 3.5 Sonnet              │ $3.00        │ $15.00       │
│ Gemini 2.0 Flash               │ $0.00        │ $0.00 (free) │
│ Ollama (local)                 │ FREE         │ FREE         │
└────────────────────────────────┴──────────────┴──────────────┘

Cost-Optimized Stack:
1. Development: Gemini embeddings + Gemini Flash LLM = FREE
2. Production (quality): OpenAI small embeddings + GPT-4o-mini = ~$0.02-0.60/1M
3. Production (budget): Gemini embeddings + GPT-4o-mini = ~$0.15-0.60/1M
4. Self-hosted: Ollama (local) = FREE (but requires GPU hardware)
    """
    )


def main():
    """Run all comparisons."""

    print("=" * 100)
    print("RAG Multi-Provider Comparison Demo")
    print("=" * 100)
    print("\nThis demo compares different RAG configurations across:")
    print("  - Embedding providers (OpenAI vs Gemini)")
    print("  - Vector store backends (In-memory vs SQLite)")
    print("  - Chunk sizes (300, 500, 1000 chars)")
    print("  - Top-K values (1, 3, 5)")

    all_results = []

    # Run all comparisons
    all_results.extend(compare_embedding_providers())
    all_results.extend(compare_vector_stores())
    all_results.extend(compare_chunk_sizes())
    all_results.extend(compare_top_k())

    # Print summary
    print_summary_table(all_results)

    # Cost analysis
    cost_analysis()

    print("\n" + "=" * 100)
    print("✅ Comparison Complete!")
    print("=" * 100)

    print("\n💡 Next Steps:")
    print("  1. Choose your embedding provider based on quality/cost needs")
    print("  2. Select vector store based on scale and persistence requirements")
    print("  3. Tune chunk size based on your document types")
    print("  4. Adjust top-k based on desired result count")
    print("  5. Monitor costs using selectools' built-in tracking")


if __name__ == "__main__":
    main()
