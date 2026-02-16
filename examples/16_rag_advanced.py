"""
Advanced RAG — PDFs, SQLite persistent storage, custom chunking, metadata filtering.

Prerequisites: OPENAI_API_KEY (examples 14-15)
    pip install selectools[rag]
Run: python examples/16_rag_advanced.py
"""

import os
import tempfile
from pathlib import Path

from selectools import Agent, AgentConfig, OpenAIProvider
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.models import OpenAI
from selectools.rag import (
    Document,
    DocumentLoader,
    RAGAgent,
    RAGTool,
    RecursiveTextSplitter,
    VectorStore,
)


def main() -> None:
    """Run advanced RAG demonstration."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return

    print("=" * 80)
    print("Advanced RAG Demo: PDFs + Persistent Storage + Custom Chunking")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Set up embedding provider and persistent storage
    # -------------------------------------------------------------------------
    print("\n📊 Step 1: Setting up embedding provider and persistent storage...")

    embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
    print(f"   ✓ Embedding model: {embedder.model}")
    print(f"   ✓ Embedding dimension: {embedder.dimension}")

    # Use SQLite for persistent storage
    db_path = "rag_demo_knowledge.db"
    vector_store = VectorStore.create("sqlite", embedder=embedder, db_path=db_path)
    print(f"   ✓ Vector store: SQLite (persistent)")
    print(f"   ✓ Database path: {db_path}")

    # -------------------------------------------------------------------------
    # Step 2: Prepare sample documents (simulate PDFs and text files)
    # -------------------------------------------------------------------------
    print("\n📚 Step 2: Loading documents from multiple sources...")

    # Create temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()

        # Create sample text files (simulating different document types)

        # Technical documentation
        (docs_dir / "architecture.md").write_text(
            """
# System Architecture

## Overview
Selectools is a production-ready framework for building AI agents with tool calling.
The architecture consists of three main layers:

1. **Provider Layer**: Abstracts LLM providers (OpenAI, Anthropic, Gemini, Ollama)
2. **Agent Layer**: Manages conversation flow, tool execution, and memory
3. **Tool Layer**: Defines callable functions with validation and execution

## RAG Integration
The RAG module adds retrieval capabilities through:
- Embedding providers for semantic search
- Vector stores for efficient document retrieval
- Document loaders and chunking strategies
- Pre-built RAG tools and high-level agent API
        """
        )

        # Feature documentation
        (docs_dir / "features.txt").write_text(
            """
Key Features of Selectools:

1. Multi-Provider Support
   - OpenAI (GPT-4, GPT-4o, o-series)
   - Anthropic (Claude 3.5, 4)
   - Google Gemini (2.0, 2.5)
   - Local Ollama models

2. Advanced Capabilities
   - Conversation memory with configurable limits
   - Automatic cost tracking and warnings
   - Tool usage analytics
   - Streaming tool results
   - Observability hooks for monitoring

3. RAG Features (v0.8.0)
   - Multi-provider embeddings (OpenAI, Anthropic, Gemini, Cohere)
   - Vector stores (In-memory, SQLite, Chroma, Pinecone)
   - Document processing and chunking
   - Semantic search tools
   - Cost tracking for embeddings

4. Developer Experience
   - Type-safe model selection with autocomplete
   - Pre-built tool library
   - Tool validation at registration
   - Comprehensive error messages
        """
        )

        # API guide
        (docs_dir / "api_guide.md").write_text(
            """
# API Quick Reference

## Basic Agent Setup

```python
from selectools import Agent, OpenAIProvider
from selectools.models import OpenAI

provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)
agent = Agent(tools=[my_tool], provider=provider)
response = agent.run("What's the weather?")
```

## RAG Agent Setup

```python
from selectools.rag import RAGAgent, VectorStore
from selectools.embeddings import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider()
vector_store = VectorStore.create("memory", embedder=embedder)

agent = RAGAgent.from_directory(
    directory="./docs",
    provider=provider,
    vector_store=vector_store,
    chunk_size=1000,
    top_k=3
)
```

## Cost Tracking

```python
from selectools import AgentConfig

config = AgentConfig(
    cost_warning_threshold=0.10,  # Warn at $0.10
    enable_analytics=True
)
agent = Agent(tools=tools, provider=provider, config=config)
usage = agent.usage
print(f"Total cost: ${usage.total_cost_usd:.4f}")
```
        """
        )

        print(f"   ✓ Created {len(list(docs_dir.glob('*')))} sample documents")

        # -------------------------------------------------------------------------
        # Step 3: Custom chunking strategy
        # -------------------------------------------------------------------------
        print("\n✂️  Step 3: Loading and chunking documents...")

        # Load documents
        loader = DocumentLoader()
        documents = loader.from_directory(str(docs_dir), glob_pattern="**/*.*")
        print(f"   ✓ Loaded {len(documents)} documents")

        # Use RecursiveTextSplitter with custom separators
        # This tries to split on natural boundaries (paragraphs, then lines, then sentences)
        splitter = RecursiveTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""], chunk_size=500, chunk_overlap=100
        )

        chunked_docs = splitter.split_documents(documents)
        print(f"   ✓ Split into {len(chunked_docs)} chunks")
        print(f"   ✓ Chunk size: 500 chars, overlap: 100 chars")

        # Add metadata tags for filtering
        for doc in chunked_docs:
            source = doc.metadata.get("source", "")
            if "architecture" in source:
                doc.metadata["category"] = "architecture"
                doc.metadata["version"] = "v0.8.0"
            elif "features" in source:
                doc.metadata["category"] = "features"
                doc.metadata["version"] = "v0.8.0"
            elif "api" in source:
                doc.metadata["category"] = "api"
                doc.metadata["version"] = "v0.8.0"

        # -------------------------------------------------------------------------
        # Step 4: Add documents to vector store
        # -------------------------------------------------------------------------
        print("\n💾 Step 4: Adding documents to persistent vector store...")

        # Clear any existing data
        vector_store.clear()

        # Add documents (will be embedded automatically)
        doc_ids = vector_store.add_documents(chunked_docs)
        print(f"   ✓ Added {len(doc_ids)} chunks to SQLite database")
        print(f"   ✓ Database: {db_path}")

        # -------------------------------------------------------------------------
        # Step 5: Create RAG agent with custom configuration
        # -------------------------------------------------------------------------
        print("\n🤖 Step 5: Creating RAG agent with cost tracking...")

        # Create RAG tool with custom parameters
        rag_tool = RAGTool(
            vector_store=vector_store,
            top_k=3,
            score_threshold=0.70,  # Only return results with >70% similarity
        )

        # Create agent with cost tracking
        provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)
        agent = Agent(
            tools=[rag_tool.search_knowledge_base],
            provider=provider,
            config=AgentConfig(cost_warning_threshold=0.05, enable_analytics=True),  # Warn at $0.05
        )

        print("   ✓ RAG agent created with:")
        print(f"      - Top-K: 3")
        print(f"      - Score threshold: 0.70")
        print(f"      - Cost warning: $0.05")

        # -------------------------------------------------------------------------
        # Step 6: Query the knowledge base
        # -------------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("💬 Querying the Knowledge Base")
        print("=" * 80)

        queries = [
            "What are the main layers in the Selectools architecture?",
            "What embedding providers are supported?",
            "How do I set up cost tracking?",
            "Tell me about the weather today",  # Should return no results
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n[Query {i}] {query}")
            print("-" * 80)

            from selectools import Message, Role

            response = agent.run([Message(role=Role.USER, content=query)])
            response_text = response.content
            print(
                f"Response: {response_text[:500]}..."
                if len(response_text) > 500
                else f"Response: {response_text}"
            )

            # Show usage after each query
            usage = agent.usage
            llm_cost = usage.total_cost_usd - usage.total_embedding_cost_usd
            print(f"\n📊 Usage so far:")
            print(f"   - LLM tokens: {usage.total_prompt_tokens + usage.total_completion_tokens:,}")
            print(f"   - LLM cost: ${llm_cost:.4f}")
            print(f"   - Embedding tokens: {usage.total_embedding_tokens:,}")
            print(f"   - Embedding cost: ${usage.total_embedding_cost_usd:.4f}")
            print(f"   - Total cost: ${usage.total_cost_usd:.4f}")

        # -------------------------------------------------------------------------
        # Step 7: Demonstrate metadata filtering
        # -------------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("🔍 Demonstrating Metadata Filtering")
        print("=" * 80)

        # Search only in 'features' category
        print("\n[Filtered Search] Searching only in 'features' documents...")
        query_embedding = embedder.embed_query("What are the key features?")
        filtered_results = vector_store.search(
            query_embedding=query_embedding, top_k=2, filter={"category": "features"}
        )

        print(f"   ✓ Found {len(filtered_results)} results in 'features' category:")
        for result in filtered_results:
            print(
                f"      - Score: {result.score:.3f}, Source: {result.document.metadata.get('source', 'N/A')}"
            )
            print(f"        Preview: {result.document.text[:100]}...")

        # -------------------------------------------------------------------------
        # Step 8: Final analytics
        # -------------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("📈 Final Analytics")
        print("=" * 80)

        usage = agent.usage
        print(f"\n{usage}")

        analytics = agent.get_analytics()
        if analytics:
            print(f"\n{analytics.summary()}")

        print("\n" + "=" * 80)
        print("✅ Demo Complete!")
        print("=" * 80)
        print(f"\n💡 The vector database has been saved to: {db_path}")
        print("   You can reuse it in future runs by loading the same database path.")
        print("\n💡 To clean up, delete the database file:")
        print(f"   rm {db_path}")


if __name__ == "__main__":
    main()
