"""
End-to-end tests for Advanced Chunking and Dynamic Tool Loading.

These tests make REAL API calls to OpenAI for embeddings and completions.
They are skipped when OPENAI_API_KEY is not set.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import pytest

from selectools import Agent, AgentConfig
from selectools.providers.openai_provider import OpenAIProvider
from selectools.rag import Document, RecursiveTextSplitter, VectorStore
from selectools.rag.chunking import ContextualChunker, SemanticChunker
from selectools.tools import ToolLoader, tool
from selectools.tools.base import Tool
from selectools.types import Message, Role

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

skip_no_openai = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")


# ---------------------------------------------------------------------------
# Realistic multi-topic document for chunking tests
# ---------------------------------------------------------------------------

MULTI_TOPIC_DOC = """
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Supervised learning uses labelled datasets to train algorithms that classify data or predict outcomes.
Common algorithms include linear regression, decision trees, and neural networks.

The Python programming language is widely used in data science and web development.
Python was created by Guido van Rossum and first released in 1991.
Its syntax emphasises readability and simplicity, making it ideal for beginners.

Climate change refers to long-term shifts in global temperatures and weather patterns.
Human activities, particularly burning fossil fuels, have been the main driver since the 1800s.
The Paris Agreement aims to limit global warming to 1.5 degrees Celsius above pre-industrial levels.

Quantum computing uses qubits that can exist in superposition of states.
Unlike classical bits which are 0 or 1, qubits can represent both simultaneously.
This enables quantum computers to solve certain problems exponentially faster.
"""


# ===================================================================
# SemanticChunker E2E
# ===================================================================


@skip_no_openai
class TestSemanticChunkerE2E:
    """Real embedding-based semantic chunking with OpenAI."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        from selectools.embeddings import OpenAIEmbeddingProvider

        self.embedder = OpenAIEmbeddingProvider(model="text-embedding-3-small")

    def test_splits_at_topic_boundaries(self) -> None:
        """The semantic chunker should detect topic shifts and split there."""
        chunker = SemanticChunker(
            self.embedder,
            similarity_threshold=0.65,
            min_chunk_sentences=2,
        )
        chunks = chunker.split_text(MULTI_TOPIC_DOC)

        print(f"\nSemantic chunks ({len(chunks)}):")
        for i, c in enumerate(chunks):
            print(f"  [{i}] {c[:80]}...")

        assert len(chunks) >= 2, "Should split into at least 2 topic groups"
        assert len(chunks) <= 8, "Should not over-fragment"

    def test_split_documents_metadata(self) -> None:
        """Documents should carry chunker metadata."""
        chunker = SemanticChunker(self.embedder, similarity_threshold=0.6)
        docs = [Document(text=MULTI_TOPIC_DOC, metadata={"source": "test.txt"})]
        result = chunker.split_documents(docs)

        assert len(result) >= 2
        for doc in result:
            assert doc.metadata["source"] == "test.txt"
            assert doc.metadata["chunker"] == "semantic"
            assert "chunk" in doc.metadata
            assert "total_chunks" in doc.metadata

    def test_similar_text_stays_together(self) -> None:
        """Text on a single topic should produce fewer chunks than diverse text."""
        single_topic = (
            "Neural networks are inspired by biological neurons. "
            "Deep learning uses multiple layers of neural networks. "
            "Backpropagation is the key algorithm for training neural networks. "
            "Convolutional neural networks are used for image recognition."
        )
        diverse_topics = (
            "Neural networks are used in deep learning applications. "
            "The stock market closed higher on Tuesday. "
            "Photosynthesis converts sunlight into chemical energy. "
            "Shakespeare wrote many famous plays in London."
        )
        chunker = SemanticChunker(self.embedder, similarity_threshold=0.5)
        single_chunks = chunker.split_text(single_topic)
        diverse_chunks = chunker.split_text(diverse_topics)

        print(f"\n  Single-topic chunks: {len(single_chunks)}")
        print(f"  Diverse-topic chunks: {len(diverse_chunks)}")

        assert len(single_chunks) <= len(
            diverse_chunks
        ), "Related text should produce fewer or equal chunks than diverse text"

    def test_chunks_into_vector_store(self) -> None:
        """Semantic chunks should be indexable and searchable."""
        chunker = SemanticChunker(self.embedder, similarity_threshold=0.6)
        docs = [Document(text=MULTI_TOPIC_DOC, metadata={"source": "multi"})]
        chunks = chunker.split_documents(docs)

        store = VectorStore.create("memory", embedder=self.embedder)
        ids = store.add_documents(chunks)
        assert len(ids) == len(chunks)

        query_emb = self.embedder.embed_query("quantum computing qubits")
        results = store.search(query_emb, top_k=2)
        assert len(results) >= 1
        assert (
            "quantum" in results[0].document.text.lower()
            or "qubit" in results[0].document.text.lower()
        )


# ===================================================================
# ContextualChunker E2E
# ===================================================================


@skip_no_openai
class TestContextualChunkerE2E:
    """Real LLM-powered contextual chunk enrichment with OpenAI."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.provider = OpenAIProvider()

    def test_enriches_chunks_with_real_context(self) -> None:
        """The LLM should generate meaningful context descriptions."""
        base = RecursiveTextSplitter(chunk_size=300, chunk_overlap=50)
        chunker = ContextualChunker(
            base_chunker=base,
            provider=self.provider,
            model="gpt-4o-mini",
        )

        docs = [Document(text=MULTI_TOPIC_DOC, metadata={"source": "test"})]
        result = chunker.split_documents(docs)

        print(f"\nContextual chunks ({len(result)}):")
        for i, doc in enumerate(result):
            context = doc.metadata.get("context", "")
            print(f"  [{i}] context: {context[:100]}...")
            print(f"       text: {doc.text[:60]}...")

        assert len(result) >= 2
        for doc in result:
            assert doc.text.startswith("[Context]")
            assert doc.metadata["chunker"] == "contextual"
            assert len(doc.metadata["context"]) > 10, "Context description should be meaningful"

    def test_contextual_semantic_pipeline(self) -> None:
        """Full pipeline: semantic split -> contextual enrichment -> vector search."""
        from selectools.embeddings import OpenAIEmbeddingProvider

        embedder = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        semantic = SemanticChunker(embedder, similarity_threshold=0.6)
        contextual = ContextualChunker(
            base_chunker=semantic,
            provider=self.provider,
            model="gpt-4o-mini",
        )

        docs = [Document(text=MULTI_TOPIC_DOC)]
        enriched = contextual.split_documents(docs)

        assert len(enriched) >= 2
        for doc in enriched:
            assert "[Context]" in doc.text
            assert doc.metadata.get("context")

        store = VectorStore.create("memory", embedder=embedder)
        store.add_documents(enriched)

        query_emb = embedder.embed_query(
            "What programming language was Guido van Rossum involved with?"
        )
        results = store.search(query_emb, top_k=2)

        assert len(results) >= 1
        top_text = results[0].document.text.lower()
        assert "python" in top_text or "guido" in top_text


# ===================================================================
# Dynamic Tool Loading E2E
# ===================================================================

PLUGIN_FILE_CODE = """
from selectools.tools import tool

@tool(name="celsius_to_fahrenheit", description="Convert Celsius to Fahrenheit")
def celsius_to_fahrenheit(celsius: float) -> str:
    f = celsius * 9 / 5 + 32
    return f"{celsius}°C = {f}°F"

@tool(name="fahrenheit_to_celsius", description="Convert Fahrenheit to Celsius")
def fahrenheit_to_celsius(fahrenheit: float) -> str:
    c = (fahrenheit - 32) * 5 / 9
    return f"{fahrenheit}°F = {c:.1f}°C"
"""


class TestDynamicToolLoadingE2E:
    """Real tool loading from files + agent integration."""

    def test_load_and_execute_from_file(self, tmp_path: Path) -> None:
        """Load tools from a real .py file and execute them."""
        plugin = tmp_path / "converters.py"
        plugin.write_text(PLUGIN_FILE_CODE)

        tools = ToolLoader.from_file(str(plugin))
        assert len(tools) == 2

        c2f = next(t for t in tools if t.name == "celsius_to_fahrenheit")
        result = c2f.execute({"celsius": 100.0})
        assert "100" in result
        assert "212" in result

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Load tools from a plugin directory."""
        (tmp_path / "math_tools.py").write_text(
            """
from selectools.tools import tool

@tool(name="add", description="Add two numbers")
def add(a: float, b: float) -> str:
    return str(a + b)
"""
        )
        (tmp_path / "text_tools.py").write_text(
            """
from selectools.tools import tool

@tool(name="upper", description="Uppercase a string")
def upper(text: str) -> str:
    return text.upper()
"""
        )
        (tmp_path / "_internal.py").write_text(
            """
from selectools.tools import tool

@tool(name="hidden", description="Should not be loaded")
def hidden() -> str:
    return "hidden"
"""
        )

        tools = ToolLoader.from_directory(str(tmp_path))
        names = {t.name for t in tools}
        assert "add" in names
        assert "upper" in names
        assert "hidden" not in names

    def test_hot_reload(self, tmp_path: Path) -> None:
        """Reload a file and get updated tools."""
        plugin = tmp_path / "versioned.py"
        plugin.write_text(
            """
from selectools.tools import tool

@tool(name="greet", description="Greet v1")
def greet(name: str) -> str:
    return f"Hello {name}"
"""
        )
        tools_v1 = ToolLoader.from_file(str(plugin))
        assert tools_v1[0].description == "Greet v1"
        assert tools_v1[0].execute({"name": "World"}) == "Hello World"

        plugin.write_text(
            """
from selectools.tools import tool

@tool(name="greet", description="Greet v2")
def greet(name: str) -> str:
    return f"Hey {name}!"
"""
        )
        tools_v2 = ToolLoader.reload_file(str(plugin))
        assert tools_v2[0].description == "Greet v2"
        assert tools_v2[0].execute({"name": "World"}) == "Hey World!"


# ===================================================================
# Agent Dynamic Tools E2E (with real provider)
# ===================================================================


@skip_no_openai
class TestAgentDynamicToolsE2E:
    """Real agent with dynamically added/removed tools."""

    def test_add_tool_and_use_it(self, tmp_path: Path) -> None:
        """Dynamically add a tool and have the agent use it."""

        @tool(name="multiply", description="Multiply two integers")
        def multiply(a: int, b: int) -> str:
            return str(a * b)

        @tool(name="divide", description="Divide two numbers")
        def divide(a: float, b: float) -> str:
            if b == 0:
                return "Cannot divide by zero"
            return str(a / b)

        provider = OpenAIProvider()
        agent = Agent(
            tools=[multiply],
            provider=provider,
            config=AgentConfig(model="gpt-4o-mini"),
        )

        assert len(agent.tools) == 1

        agent.add_tool(divide)
        assert len(agent.tools) == 2
        assert "divide" in agent._tools_by_name

        response = agent.run([Message(role=Role.USER, content="What is 144 divided by 12?")])

        print(f"\nAgent response: {response.message.content}")
        assert response.tool_calls is not None
        assert any(tc.tool_name == "divide" for tc in response.tool_calls)
        assert "12" in response.message.content

    def test_remove_tool_agent_cannot_use_it(self) -> None:
        """After removing a tool, the agent should not be able to call it."""

        @tool(name="secret_tool", description="Access secret data")
        def secret_tool(key: str) -> str:
            return f"Secret: {key}"

        @tool(name="public_tool", description="Get public info")
        def public_tool(topic: str) -> str:
            return f"Public info about {topic}"

        provider = OpenAIProvider()
        agent = Agent(
            tools=[secret_tool, public_tool],
            provider=provider,
            config=AgentConfig(model="gpt-4o-mini"),
        )

        agent.remove_tool("secret_tool")
        assert "secret_tool" not in agent._tools_by_name
        assert len(agent.tools) == 1

    def test_replace_tool_hot_swap(self, tmp_path: Path) -> None:
        """Replace a tool with an updated version."""

        @tool(name="lookup", description="Look up a value (v1)")
        def lookup_v1(key: str) -> str:
            return f"v1: {key}"

        @tool(name="lookup", description="Look up a value (v2 - improved)")
        def lookup_v2(key: str) -> str:
            return f"v2-improved: {key}"

        provider = OpenAIProvider()
        agent = Agent(
            tools=[lookup_v1],
            provider=provider,
            config=AgentConfig(model="gpt-4o-mini"),
        )

        assert agent._tools_by_name["lookup"].description == "Look up a value (v1)"

        old = agent.replace_tool(lookup_v2)
        assert old is not None
        assert old.description == "Look up a value (v1)"
        assert agent._tools_by_name["lookup"].description == "Look up a value (v2 - improved)"
        assert len(agent.tools) == 1

    def test_load_from_file_and_add_to_agent(self, tmp_path: Path) -> None:
        """Full E2E: ToolLoader.from_file -> agent.add_tools -> agent.run."""

        @tool(name="base_tool", description="Base tool that always exists")
        def base_tool(input: str) -> str:
            return f"base: {input}"

        plugin = tmp_path / "temperature.py"
        plugin.write_text(PLUGIN_FILE_CODE)

        provider = OpenAIProvider()
        agent = Agent(
            tools=[base_tool],
            provider=provider,
            config=AgentConfig(model="gpt-4o-mini"),
        )

        loaded = ToolLoader.from_file(str(plugin))
        agent.add_tools(loaded)

        assert len(agent.tools) == 3
        assert "celsius_to_fahrenheit" in agent._tools_by_name
        assert "fahrenheit_to_celsius" in agent._tools_by_name

        response = agent.run(
            [Message(role=Role.USER, content="Convert 37 degrees Celsius to Fahrenheit")]
        )

        print(f"\nAgent response: {response.message.content}")
        assert response.tool_calls is not None
        assert any(tc.tool_name == "celsius_to_fahrenheit" for tc in response.tool_calls)
        assert "98" in response.message.content or "99" in response.message.content
