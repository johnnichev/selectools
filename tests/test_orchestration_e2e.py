"""
End-to-end orchestration tests with real LLM providers.

These tests make actual API calls. Run with:

    pytest tests/test_orchestration_e2e.py -v --run-e2e

    # Single provider:
    pytest tests/test_orchestration_e2e.py -v --run-e2e -k openai
    pytest tests/test_orchestration_e2e.py -v --run-e2e -k anthropic
    pytest tests/test_orchestration_e2e.py -v --run-e2e -k gemini

Required env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
"""

from __future__ import annotations

import os

import pytest

from selectools import Agent, AgentConfig, AgentGraph, tool
from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState
from selectools.types import Message, Role

# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------


@tool(description="Summarize text into a single sentence")
def summarize(text: str) -> str:
    """Summarize the given text."""
    return f"Summary: {text[:100]}"


@tool(description="Translate text to Spanish")
def translate_to_spanish(text: str) -> str:
    """Translate text to Spanish."""
    return f"Translated: {text}"


@tool(description="Count words in text")
def count_words(text: str) -> str:
    """Count words in text."""
    count = len(text.split())
    return f"Word count: {count}"


# ---------------------------------------------------------------------------
# Provider fixtures
# ---------------------------------------------------------------------------


def _make_agent(provider, model, tools):
    """Create an agent with the given provider, model, and tools."""
    return Agent(
        provider=provider,
        tools=tools,
        config=AgentConfig(model=model, max_iterations=3),
    )


# ---------------------------------------------------------------------------
# OpenAI E2E
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.openai
class TestOrchestrationOpenAI:
    """Real LLM calls through OpenAI for orchestration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        from selectools.providers.openai_provider import OpenAIProvider

        self.provider = OpenAIProvider()
        self.model = "gpt-4.1-mini"

    def test_linear_graph_real_llm(self):
        """Two agents in sequence with real LLM calls."""
        agent_a = _make_agent(self.provider, self.model, [summarize])
        agent_b = _make_agent(self.provider, self.model, [count_words])

        graph = AgentGraph.chain(agent_a, agent_b)
        result = graph.run("The quick brown fox jumps over the lazy dog. This is a test sentence.")

        assert result.content, "Graph should produce non-empty content"
        assert len(result.node_results) == 2, "Both nodes should have results"
        assert result.total_usage.total_tokens > 0, "Should have token usage"

    def test_single_agent_graph(self):
        """Single agent in a graph with real LLM."""
        agent = _make_agent(self.provider, self.model, [summarize])

        graph = AgentGraph.chain(agent)
        result = graph.run("Explain quantum computing in simple terms")

        assert result.content
        assert result.steps >= 1

    def test_callable_node_with_llm_agent(self):
        """Mix callable nodes with real LLM agents."""
        agent = _make_agent(self.provider, self.model, [summarize])

        def uppercase_node(state: GraphState) -> GraphState:
            last = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            state.data[STATE_KEY_LAST_OUTPUT] = last.upper()
            return state

        graph = AgentGraph()
        graph.add_node("llm", agent, next_node="transform")
        graph.add_node("transform", uppercase_node, next_node=AgentGraph.END)
        result = graph.run("Say hello world")

        assert result.content
        # The uppercase transform should have made the content uppercase
        assert result.content == result.content.upper()


# ---------------------------------------------------------------------------
# Anthropic E2E
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.anthropic
class TestOrchestrationAnthropic:
    """Real LLM calls through Anthropic for orchestration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        from selectools.providers.anthropic_provider import AnthropicProvider

        self.provider = AnthropicProvider()
        self.model = "claude-haiku-4-5-20251001"

    def test_linear_graph_real_llm(self):
        """Two agents in sequence with real Anthropic calls."""
        agent_a = _make_agent(self.provider, self.model, [summarize])
        agent_b = _make_agent(self.provider, self.model, [count_words])

        graph = AgentGraph.chain(agent_a, agent_b)
        result = graph.run(
            "Python is a popular programming language used for web, AI, and scripting."
        )

        assert result.content
        assert len(result.node_results) == 2
        assert result.total_usage.total_tokens > 0

    def test_system_messages_dont_crash(self):
        """Verify SYSTEM messages in history don't crash Anthropic.

        This is the exact bug a user reported — prompt compression, entity
        memory, and knowledge graph inject Role.SYSTEM messages into history.
        Anthropic rejects 'system' role in messages with a 400 error.
        """
        agent = _make_agent(self.provider, self.model, [summarize])

        # Simulate what prompt compression does: inject a SYSTEM message
        # into the agent's history before running through the graph
        def inject_system_context(state: GraphState) -> GraphState:
            state.messages.append(
                Message(role=Role.SYSTEM, content="[Compressed context] User likes Python.")
            )
            state.messages.append(Message(role=Role.USER, content="What do I like?"))
            state.data[STATE_KEY_LAST_OUTPUT] = "What do I like?"
            return state

        graph = AgentGraph()
        graph.add_node("inject", inject_system_context, next_node="llm")
        graph.add_node("llm", agent, next_node=AgentGraph.END)
        result = graph.run("Tell me about myself")

        # This must not crash with "Unexpected role system"
        assert result.content

    def test_conditional_routing_real_llm(self):
        """Conditional routing with real Anthropic LLM."""
        agent = _make_agent(self.provider, self.model, [summarize])

        def route(state: GraphState) -> str:
            last = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            return AgentGraph.END if len(last) > 10 else "retry"

        graph = AgentGraph()
        graph.add_node("agent", agent)
        graph.add_node("retry", agent, next_node=AgentGraph.END)
        graph.add_conditional_edge("agent", route, path_map={"retry": "retry"})
        result = graph.run("Summarize: AI is transforming every industry in 2026.")

        assert result.content
        assert result.steps >= 1


# ---------------------------------------------------------------------------
# Gemini E2E
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.gemini
class TestOrchestrationGemini:
    """Real LLM calls through Gemini for orchestration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")
        from selectools.providers.gemini_provider import GeminiProvider

        self.provider = GeminiProvider()
        self.model = "gemini-3-flash-preview"

    def test_linear_graph_real_llm(self):
        """Two agents in sequence with real Gemini calls."""
        agent_a = _make_agent(self.provider, self.model, [summarize])
        agent_b = _make_agent(self.provider, self.model, [count_words])

        graph = AgentGraph.chain(agent_a, agent_b)
        result = graph.run("Machine learning models learn patterns from data to make predictions.")

        assert result.content
        assert len(result.node_results) == 2
        assert result.total_usage.total_tokens > 0

    def test_system_messages_dont_crash_gemini(self):
        """Verify SYSTEM messages in history don't crash Gemini."""
        agent = _make_agent(self.provider, self.model, [summarize])

        def inject_system_context(state: GraphState) -> GraphState:
            state.messages.append(
                Message(role=Role.SYSTEM, content="[Entity context] User is a developer.")
            )
            state.messages.append(Message(role=Role.USER, content="What's my role?"))
            state.data[STATE_KEY_LAST_OUTPUT] = "What's my role?"
            return state

        graph = AgentGraph()
        graph.add_node("inject", inject_system_context, next_node="llm")
        graph.add_node("llm", agent, next_node=AgentGraph.END)
        result = graph.run("Tell me about myself")

        assert result.content


# ---------------------------------------------------------------------------
# Cross-provider
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestOrchestrationCrossProvider:
    """Tests that mix providers in the same graph."""

    @pytest.fixture(autouse=True)
    def setup(self):
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        if not (has_openai and has_anthropic):
            pytest.skip("Need both OPENAI_API_KEY and ANTHROPIC_API_KEY")

        from selectools.providers.anthropic_provider import AnthropicProvider
        from selectools.providers.openai_provider import OpenAIProvider

        self.openai = OpenAIProvider()
        self.anthropic = AnthropicProvider()

    def test_mixed_provider_graph(self):
        """OpenAI agent feeds into Anthropic agent in one graph."""
        agent_openai = _make_agent(self.openai, "gpt-4.1-mini", [summarize])
        agent_anthropic = _make_agent(self.anthropic, "claude-haiku-4-5-20251001", [count_words])

        graph = AgentGraph.chain(
            agent_openai, agent_anthropic, names=["openai_step", "anthropic_step"]
        )
        result = graph.run("Explain the concept of neural networks briefly.")

        assert result.content
        assert "openai_step" in result.node_results
        assert "anthropic_step" in result.node_results
        assert result.total_usage.total_tokens > 0
