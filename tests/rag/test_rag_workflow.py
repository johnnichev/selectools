"""
Complete RAG workflow integration tests.

Tests the full pipeline: Load → Chunk → Embed → Store → Search → Agent
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Generator, Tuple
from unittest.mock import MagicMock, Mock

import pytest

from selectools import Agent, AgentConfig
from selectools.rag import (
    Document,
    DocumentLoader,
    RAGAgent,
    RAGTool,
    RecursiveTextSplitter,
    SemanticSearchTool,
    TextSplitter,
    VectorStore,
)
from selectools.types import AgentResult, Message, Role

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_embedder() -> Mock:
    """Create a consistent mock embedding provider."""
    embedder = Mock()
    embedder.model = "mock-embedding-model"
    embedder.dimension = 128

    def mock_embed_text(text: str) -> list[float]:
        # Consistent embeddings based on text hash
        hash_val = abs(hash(text)) % 1000
        return [float(hash_val + i) / 1000.0 for i in range(128)]

    def mock_embed_texts(texts: list[str]) -> list[list[float]]:
        return [mock_embed_text(text) for text in texts]

    def mock_embed_query(query: str) -> list[float]:
        return mock_embed_text(query)

    embedder.embed_text.side_effect = mock_embed_text
    embedder.embed_texts.side_effect = mock_embed_texts
    embedder.embed_query.side_effect = mock_embed_query

    return embedder


@pytest.fixture
def mock_provider() -> Mock:
    """Create a mock LLM provider."""
    provider = Mock()
    provider.name = "mock-provider"
    provider.default_model = "mock-model"
    provider.supports_streaming = False
    provider.supports_async = False

    def mock_complete(*args: Any, **kwargs: Any) -> Tuple[Message, Mock]:
        # Simulate tool call
        messages = kwargs.get("messages", [])
        last_message = messages[-1] if messages else None

        # Handle both string and Message object
        content = (
            last_message.content
            if last_message is not None and hasattr(last_message, "content")
            else str(last_message)
        )

        if last_message and "search_knowledge_base" in content:
            # Return tool call response
            return (
                Message(
                    role=Role.ASSISTANT,
                    content='TOOL_CALL: {"tool_name": "search_knowledge_base", "parameters": {"query": "test"}}',
                ),
                Mock(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    embedding_tokens=0,
                    embedding_cost_usd=0.0,
                    cost_usd=0.001,
                    total_cost_usd=0.001,
                    model="mock",
                    provider="mock",
                ),
            )
        else:
            # Return final answer
            return (
                Message(
                    role=Role.ASSISTANT,
                    content="Based on the knowledge base, here is the answer.",
                ),
                Mock(
                    prompt_tokens=5,
                    completion_tokens=15,
                    total_tokens=20,
                    embedding_tokens=0,
                    embedding_cost_usd=0.0,
                    cost_usd=0.0005,
                    total_cost_usd=0.0005,
                    model="mock",
                    provider="mock",
                ),
            )

    provider.complete.side_effect = mock_complete

    return provider


@pytest.fixture
def temp_docs_dir() -> Generator[Path, None, None]:
    """Create temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        (temp_path / "doc1.txt").write_text(
            "Python is a high-level programming language. "
            "It is widely used for web development, data science, and automation."
        )
        (temp_path / "doc2.txt").write_text(
            "Machine learning enables computers to learn from data. "
            "It is a key component of modern AI systems."
        )
        (temp_path / "doc3.txt").write_text(
            "Docker containers provide isolated environments for applications. "
            "Kubernetes orchestrates these containers at scale."
        )

        yield temp_path


# ============================================================================
# Full Pipeline Tests
# ============================================================================


class TestCompleteRAGWorkflow:
    """Test complete RAG workflow from end to end."""

    def test_load_chunk_embed_search(self, mock_embedder: Mock, temp_docs_dir: Path) -> None:
        """Test: Load documents → Chunk → Embed → Store → Search."""
        # 1. Load documents
        loader = DocumentLoader()
        documents = loader.from_directory(str(temp_docs_dir), glob_pattern="*.txt")
        assert len(documents) == 3

        # 2. Chunk documents
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunked_docs = splitter.split_documents(documents)
        assert len(chunked_docs) >= len(documents)

        # 3. Create vector store and embed
        vector_store = VectorStore.create("memory", embedder=mock_embedder)
        doc_ids = vector_store.add_documents(chunked_docs)
        assert len(doc_ids) == len(chunked_docs)

        # 4. Search
        query_embedding = mock_embedder.embed_query("programming language")
        results = vector_store.search(query_embedding, top_k=3)

        assert len(results) > 0
        assert all(hasattr(r, "document") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_rag_agent_from_documents(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test RAGAgent.from_documents() workflow."""
        # Create documents
        docs = [
            Document(text="Python is great for web development.", metadata={}),
            Document(text="JavaScript is used for frontend development.", metadata={}),
            Document(text="Docker simplifies application deployment.", metadata={}),
        ]

        # Create vector store
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        # Create RAG agent
        agent = RAGAgent.from_documents(
            documents=docs,
            provider=mock_provider,
            vector_store=vector_store,
            chunk_size=100,
            chunk_overlap=20,
            top_k=2,
        )

        assert isinstance(agent, Agent)
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "search_knowledge_base"

    def test_rag_agent_from_directory(
        self, mock_embedder: Mock, mock_provider: Mock, temp_docs_dir: Path
    ) -> None:
        """Test RAGAgent.from_directory() workflow."""
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        agent = RAGAgent.from_directory(
            directory=str(temp_docs_dir),
            provider=mock_provider,
            vector_store=vector_store,
            glob_pattern="*.txt",
            chunk_size=100,
            chunk_overlap=20,
            top_k=2,
        )

        assert isinstance(agent, Agent)
        assert len(agent.tools) == 1

    def test_rag_agent_run_query(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test running a query through RAG agent."""
        # Setup
        docs = [
            Document(text="Python supports object-oriented programming.", metadata={}),
            Document(text="Python has a rich ecosystem of libraries.", metadata={}),
        ]

        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        agent = RAGAgent.from_documents(
            documents=docs,
            provider=mock_provider,
            vector_store=vector_store,
            chunk_size=100,
            chunk_overlap=20,
            top_k=1,
        )

        # Run query
        from selectools.types import Message, Role

        response = agent.run([Message(role=Role.USER, content="Tell me about Python")])

        assert isinstance(response, AgentResult)
        assert len(response.content) > 0


# ============================================================================
# Different Vector Store Backend Tests
# ============================================================================


class TestRAGWithDifferentStores:
    """Test RAG workflow with different vector store backends."""

    def test_with_memory_store(
        self, mock_embedder: Mock, mock_provider: Mock, temp_docs_dir: Path
    ) -> None:
        """Test RAG with in-memory vector store."""
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        agent = RAGAgent.from_directory(
            directory=str(temp_docs_dir), provider=mock_provider, vector_store=vector_store
        )

        assert isinstance(agent, Agent)

    def test_with_sqlite_store(
        self, mock_embedder: Mock, mock_provider: Mock, temp_docs_dir: Path
    ) -> None:
        """Test RAG with SQLite vector store."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            vector_store = VectorStore.create("sqlite", embedder=mock_embedder, db_path=db_path)

            agent = RAGAgent.from_directory(
                directory=str(temp_docs_dir), provider=mock_provider, vector_store=vector_store
            )

            assert isinstance(agent, Agent)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


# ============================================================================
# Different Chunking Strategy Tests
# ============================================================================


class TestRAGWithDifferentChunking:
    """Test RAG with different chunking strategies."""

    def test_with_text_splitter(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test RAG with basic TextSplitter."""
        docs = [Document(text="a" * 500, metadata={})]
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        # Load and chunk manually
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunked_docs = splitter.split_documents(docs)

        # Add to store
        vector_store.add_documents(chunked_docs)

        # Create agent with RAG tool
        rag_tool = RAGTool(vector_store=vector_store, top_k=2)
        agent = Agent(tools=[rag_tool.search_knowledge_base], provider=mock_provider)

        assert isinstance(agent, Agent)

    def test_with_recursive_splitter(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test RAG with RecursiveTextSplitter."""
        docs = [Document(text="Paragraph one.\n\nParagraph two.\n\nParagraph three.", metadata={})]
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        # Use recursive splitter
        splitter = RecursiveTextSplitter(
            chunk_size=50, chunk_overlap=10, separators=["\n\n", "\n", " ", ""]
        )
        chunked_docs = splitter.split_documents(docs)

        vector_store.add_documents(chunked_docs)

        rag_tool = RAGTool(vector_store=vector_store, top_k=2)
        agent = Agent(tools=[rag_tool.search_knowledge_base], provider=mock_provider)

        assert isinstance(agent, Agent)


# ============================================================================
# Cost Tracking Integration Tests
# ============================================================================


class TestRAGCostTracking:
    """Test cost tracking integration with RAG."""

    def test_cost_tracking_enabled(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test that cost tracking works with RAG."""
        docs = [Document(text="test document", metadata={})]
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        agent = RAGAgent.from_documents(
            documents=docs,
            provider=mock_provider,
            vector_store=vector_store,
            agent_config=AgentConfig(enable_analytics=True),
        )

        # Run a query
        from selectools.types import Message, Role

        agent.run([Message(role=Role.USER, content="test query")])

        # Check usage
        usage = agent.usage
        assert usage is not None
        assert hasattr(usage, "total_cost_usd")

    def test_analytics_tracking(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test analytics integration with RAG."""
        docs = [Document(text="test document", metadata={})]
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        agent = RAGAgent.from_documents(
            documents=docs,
            provider=mock_provider,
            vector_store=vector_store,
            agent_config=AgentConfig(enable_analytics=True),
        )

        # Run a query
        agent.run("test query")

        # Check analytics
        analytics = agent.get_analytics()
        assert analytics is not None


# ============================================================================
# RAGTool and SemanticSearchTool Tests
# ============================================================================


class TestRAGTools:
    """Test RAG tool implementations."""

    def test_rag_tool_basic(self, mock_embedder: Mock) -> None:
        """Test basic RAGTool functionality."""
        # Setup vector store with documents
        docs = [
            Document(text="Python is a programming language.", metadata={"source": "doc1"}),
            Document(text="Docker is a containerization tool.", metadata={"source": "doc2"}),
        ]

        vector_store = VectorStore.create("memory", embedder=mock_embedder)
        vector_store.add_documents(docs)

        # Create RAG tool
        rag_tool = RAGTool(vector_store=vector_store, top_k=2, score_threshold=0.5)

        # Search - call the underlying function of the decorated tool (pass self explicitly)
        result = rag_tool.search_knowledge_base.function(rag_tool, "programming")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_rag_tool_no_results(self, mock_embedder: Mock) -> None:
        """Test RAGTool with no results above threshold."""
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        # Add document
        docs = [Document(text="Python programming.", metadata={})]
        vector_store.add_documents(docs)

        # Create tool with perfect match threshold
        rag_tool = RAGTool(vector_store=vector_store, top_k=1, score_threshold=1.0)

        result = rag_tool.search_knowledge_base.function(
            rag_tool, "completely unrelated query xyz123"
        )

        assert "No relevant information found" in result

    def test_semantic_search_tool(self, mock_embedder: Mock) -> None:
        """Test SemanticSearchTool."""
        docs = [
            Document(text="First document", metadata={"id": 1}),
            Document(text="Second document", metadata={"id": 2}),
        ]

        vector_store = VectorStore.create("memory", embedder=mock_embedder)
        vector_store.add_documents(docs)

        # Create semantic search tool
        search_tool = SemanticSearchTool(vector_store=vector_store, top_k=2, score_threshold=0.5)

        results = search_tool.search("document")

        assert isinstance(results, list)
        assert len(results) <= 2
        if results:
            # SearchResult objects have document and score attributes
            assert hasattr(results[0], "document")
            assert hasattr(results[0], "score")
            assert hasattr(results[0].document, "text")
            assert hasattr(results[0].document, "metadata")


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestRAGErrorHandling:
    """Test error handling in RAG workflows."""

    def test_empty_documents(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test RAG with empty document list."""
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        agent = RAGAgent.from_documents(
            documents=[], provider=mock_provider, vector_store=vector_store
        )

        assert isinstance(agent, Agent)

    def test_empty_directory(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test RAG with empty directory raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore.create("memory", embedder=mock_embedder)

            # Empty directory should raise ValueError
            with pytest.raises(ValueError, match="No documents found"):
                RAGAgent.from_directory(
                    directory=temp_dir, provider=mock_provider, vector_store=vector_store
                )

    def test_invalid_chunk_size(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test error with invalid chunk size."""
        docs = [Document(text="test", metadata={})]
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        # chunk_overlap >= chunk_size should fail
        with pytest.raises(ValueError):
            RAGAgent.from_documents(
                documents=docs,
                provider=mock_provider,
                vector_store=vector_store,
                chunk_size=100,
                chunk_overlap=100,
            )


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


class TestEndToEndRAGWorkflow:
    """Complete end-to-end RAG workflow tests."""

    def test_complete_workflow(
        self, mock_embedder: Mock, mock_provider: Mock, temp_docs_dir: Path
    ) -> None:
        """Test complete RAG workflow from files to answer."""
        # Create RAG agent from directory
        vector_store = VectorStore.create("memory", embedder=mock_embedder)

        agent = RAGAgent.from_directory(
            directory=str(temp_docs_dir),
            provider=mock_provider,
            vector_store=vector_store,
            glob_pattern="*.txt",
            chunk_size=100,
            chunk_overlap=20,
            top_k=3,
            score_threshold=0.5,
            agent_config=AgentConfig(enable_analytics=True),
        )

        # Run query
        from selectools.types import Message, Role

        response = agent.run([Message(role=Role.USER, content="What is Python used for?")])

        # Verify response
        assert isinstance(response, AgentResult)
        assert len(response.content) > 0

        # Verify usage tracking
        usage = agent.usage
        assert usage.total_cost_usd >= 0
        assert usage.total_tokens > 0

        # Note: tool_usage tracking requires full agent execution loop, not just mock responses

    def test_multiple_queries(self, mock_embedder: Mock, mock_provider: Mock) -> None:
        """Test multiple sequential queries."""
        docs = [
            Document(text="Python is used for data science.", metadata={}),
            Document(text="Docker is used for containerization.", metadata={}),
        ]

        vector_store = VectorStore.create("memory", embedder=mock_embedder)
        agent = RAGAgent.from_documents(
            documents=docs, provider=mock_provider, vector_store=vector_store
        )

        # Multiple queries
        from selectools.types import Message, Role

        response1 = agent.run([Message(role=Role.USER, content="Tell me about Python")])
        response2 = agent.run([Message(role=Role.USER, content="Tell me about Docker")])

        assert isinstance(response1, AgentResult)
        assert isinstance(response2, AgentResult)

        # Verify usage accumulated
        usage = agent.usage
        assert len(usage.iterations) >= 2
        assert usage.total_tokens > 0
