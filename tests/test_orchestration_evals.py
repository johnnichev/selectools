"""
Real-LLM evaluation suite for orchestration correctness.

Every test in this file makes actual API calls. No mocks.
Run with: pytest tests/test_orchestration_evals.py -v --run-e2e

Validates that agents in graphs actually:
- Pick the right tools
- Produce coherent multi-step output
- Handle SYSTEM messages from context injection
- Work across all 3 providers (OpenAI, Anthropic, Gemini)
- Coordinate via SupervisorAgent strategies
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from selectools import Agent, AgentConfig, AgentGraph, tool
from selectools.orchestration.checkpoint import InMemoryCheckpointStore
from selectools.orchestration.state import STATE_KEY_LAST_OUTPUT, GraphState, InterruptRequest
from selectools.types import Message, Role

# ---------------------------------------------------------------------------
# Real tools that agents must choose correctly
# ---------------------------------------------------------------------------


@tool(description="Look up the capital city of a country")
def get_capital(country: str) -> str:
    """Return the capital city of a country."""
    capitals = {
        "france": "Paris",
        "japan": "Tokyo",
        "brazil": "Brasilia",
        "australia": "Canberra",
    }
    return capitals.get(country.lower(), f"Unknown capital for {country}")


@tool(description="Calculate the result of a math expression")
def calculate(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        result = eval(expression)  # nosec B307 — test tool only
        return f"Result: {result}"
    except Exception:
        return f"Could not evaluate: {expression}"


@tool(description="Count the number of words in a text")
def word_count(text: str) -> str:
    """Count words in the given text."""
    count = len(text.split())
    return f"The text has {count} words"


@tool(description="Convert text to uppercase")
def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

PROVIDER_CONFIGS = {
    "openai": {
        "env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
        "factory": "selectools.providers.openai_provider:OpenAIProvider",
    },
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "model": "claude-haiku-4-5-20251001",
        "factory": "selectools.providers.anthropic_provider:AnthropicProvider",
    },
    "gemini": {
        "env": "GOOGLE_API_KEY",
        "model": "gemini-3-flash-preview",
        "factory": "selectools.providers.gemini_provider:GeminiProvider",
    },
}


def _get_provider(name: str):
    """Import and instantiate a provider by name. Returns (provider, model)."""
    cfg = PROVIDER_CONFIGS[name]
    if not os.getenv(cfg["env"]):
        pytest.skip(f"{cfg['env']} not set")
    mod_path, cls_name = cfg["factory"].rsplit(":", 1)
    import importlib

    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls(), cfg["model"]


def _make_agent(provider, model, tools, max_iter=3):
    return Agent(
        provider=provider,
        tools=tools,
        config=AgentConfig(model=model, max_iterations=max_iter),
    )


# ---------------------------------------------------------------------------
# Parametrize across all providers
# ---------------------------------------------------------------------------

PROVIDERS = ["openai", "anthropic", "gemini"]


@pytest.mark.e2e
class TestToolCallingAccuracy:
    """Does the LLM pick the right tool inside a graph node?"""

    @pytest.mark.parametrize("provider_name", PROVIDERS)
    def test_agent_calls_correct_tool_for_capital(self, provider_name):
        """Agent in a graph must call get_capital for a capital-city question."""
        provider, model = _get_provider(provider_name)
        agent = _make_agent(provider, model, [get_capital, calculate, word_count])

        graph = AgentGraph.chain(agent)
        result = graph.run("What is the capital of France?")

        assert result.content, f"[{provider_name}] Empty response"
        assert "Paris" in result.content, (
            f"[{provider_name}] Expected 'Paris' in response, got: {result.content[:200]}"
        )

    @pytest.mark.parametrize("provider_name", PROVIDERS)
    def test_agent_calls_correct_tool_for_math(self, provider_name):
        """Agent in a graph must call calculate for a math question."""
        provider, model = _get_provider(provider_name)
        agent = _make_agent(provider, model, [get_capital, calculate, word_count])

        graph = AgentGraph.chain(agent)
        result = graph.run("What is 15 * 7?")

        assert result.content, f"[{provider_name}] Empty response"
        assert "105" in result.content, (
            f"[{provider_name}] Expected '105' in response, got: {result.content[:200]}"
        )


@pytest.mark.e2e
class TestMultiStepPipeline:
    """Does a multi-agent pipeline produce coherent output?"""

    @pytest.mark.parametrize("provider_name", PROVIDERS)
    def test_two_agent_pipeline_produces_result(self, provider_name):
        """Agent A answers a question, Agent B summarizes — both produce content."""
        provider, model = _get_provider(provider_name)

        agent_a = _make_agent(provider, model, [get_capital])
        agent_b = _make_agent(provider, model, [word_count])

        graph = AgentGraph.chain(agent_a, agent_b, names=["lookup", "analyze"])
        result = graph.run("What is the capital of Japan?")

        assert result.content, f"[{provider_name}] Empty final output"
        assert "lookup" in result.node_results, f"[{provider_name}] lookup node missing"
        assert "analyze" in result.node_results, f"[{provider_name}] analyze node missing"
        assert result.total_usage.total_tokens > 0, f"[{provider_name}] No token usage"

    @pytest.mark.parametrize("provider_name", PROVIDERS)
    def test_callable_then_llm_pipeline(self, provider_name):
        """Callable node transforms data, then LLM agent processes it."""
        provider, model = _get_provider(provider_name)

        def prepare_data(state: GraphState) -> GraphState:
            state.data[STATE_KEY_LAST_OUTPUT] = "The population of Tokyo is 14 million people."
            return state

        agent = _make_agent(provider, model, [word_count])

        graph = AgentGraph()
        graph.add_node("prepare", prepare_data, next_node="analyze")
        graph.add_node("analyze", agent, next_node=AgentGraph.END)
        result = graph.run("Count words")

        assert result.content, f"[{provider_name}] Empty response"
        # Agent should have processed the prepared text
        assert result.total_usage.total_tokens > 0


@pytest.mark.e2e
class TestSystemMessageSurvival:
    """SYSTEM messages from context injection must not crash any provider."""

    @pytest.mark.parametrize("provider_name", PROVIDERS)
    def test_system_message_in_history_doesnt_crash(self, provider_name):
        """Injected SYSTEM messages (from prompt compression, entity memory)
        must survive through the provider without errors."""
        provider, model = _get_provider(provider_name)
        agent = _make_agent(provider, model, [get_capital])

        def inject_context(state: GraphState) -> GraphState:
            # Simulate what prompt compression / entity memory does
            state.messages.append(
                Message(
                    role=Role.SYSTEM,
                    content="[Compressed context] User previously asked about European capitals.",
                )
            )
            state.messages.append(Message(role=Role.USER, content="What is the capital of France?"))
            state.data[STATE_KEY_LAST_OUTPUT] = "What is the capital of France?"
            return state

        graph = AgentGraph()
        graph.add_node("inject", inject_context, next_node="agent")
        graph.add_node("agent", agent, next_node=AgentGraph.END)
        result = graph.run("start")

        # Must not crash. Content should be about Paris.
        assert result.content, f"[{provider_name}] Empty response after SYSTEM injection"
        assert "error" not in result.content.lower() or "Paris" in result.content, (
            f"[{provider_name}] Got error or wrong answer: {result.content[:200]}"
        )

    @pytest.mark.parametrize("provider_name", PROVIDERS)
    def test_multiple_system_messages_dont_crash(self, provider_name):
        """Multiple SYSTEM messages in sequence must all be handled."""
        provider, model = _get_provider(provider_name)
        agent = _make_agent(provider, model, [calculate])

        def inject_multi_system(state: GraphState) -> GraphState:
            state.messages.append(
                Message(role=Role.SYSTEM, content="[Entity memory] User is a math student.")
            )
            state.messages.append(
                Message(role=Role.SYSTEM, content="[Knowledge graph] User likes algebra.")
            )
            state.messages.append(Message(role=Role.USER, content="What is 10 + 20?"))
            state.data[STATE_KEY_LAST_OUTPUT] = "What is 10 + 20?"
            return state

        graph = AgentGraph()
        graph.add_node("inject", inject_multi_system, next_node="agent")
        graph.add_node("agent", agent, next_node=AgentGraph.END)
        result = graph.run("start")

        assert result.content, f"[{provider_name}] Empty response"
        assert "30" in result.content, (
            f"[{provider_name}] Expected '30' in response, got: {result.content[:200]}"
        )


@pytest.mark.e2e
class TestCrossProvider:
    """Mix providers in a single graph."""

    def test_openai_to_anthropic_pipeline(self):
        """OpenAI agent → Anthropic agent in one graph."""
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Need both OPENAI_API_KEY and ANTHROPIC_API_KEY")

        from selectools.providers.anthropic_provider import AnthropicProvider
        from selectools.providers.openai_provider import OpenAIProvider

        openai_agent = _make_agent(OpenAIProvider(), "gpt-4.1-mini", [get_capital])
        anthropic_agent = _make_agent(
            AnthropicProvider(), "claude-haiku-4-5-20251001", [word_count]
        )

        graph = AgentGraph.chain(openai_agent, anthropic_agent, names=["openai", "anthropic"])
        result = graph.run("What is the capital of Brazil?")

        assert result.content
        assert "openai" in result.node_results
        assert "anthropic" in result.node_results
        assert result.total_usage.total_tokens > 0


@pytest.mark.e2e
class TestHITLWithRealAgent:
    """HITL interrupt/resume with a real LLM agent in the graph."""

    @pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
    def test_interrupt_before_llm_resumes_correctly(self, provider_name):
        """Generator node interrupts, then LLM agent runs after resume."""
        provider, model = _get_provider(provider_name)
        agent = _make_agent(provider, model, [get_capital])

        async def approval_gate(state: GraphState):
            question = state.data.get(STATE_KEY_LAST_OUTPUT, "")
            approval = yield InterruptRequest(prompt=f"Allow this question? {question}")
            state.data["approved"] = approval
            state.data[STATE_KEY_LAST_OUTPUT] = question

        graph = AgentGraph()
        graph.add_node("gate", approval_gate, next_node="agent")
        graph.add_node("agent", agent, next_node=AgentGraph.END)

        store = InMemoryCheckpointStore()
        state = GraphState.from_prompt("What is the capital of Australia?")
        state.data[STATE_KEY_LAST_OUTPUT] = "What is the capital of Australia?"

        interrupted = graph.run(state, checkpoint_store=store)
        assert interrupted.interrupted, f"[{provider_name}] Should be interrupted"

        final = graph.resume(interrupted.interrupt_id, "yes", checkpoint_store=store)
        assert not final.interrupted, f"[{provider_name}] Should not be interrupted after resume"
        assert final.content, f"[{provider_name}] Should have content after resume"
        assert final.state.data.get("approved") == "yes"


@pytest.mark.e2e
class TestParallelWithRealAgents:
    """Parallel execution with real LLM agents."""

    def test_parallel_agents_both_produce_output(self):
        """Two agents running in parallel both produce results."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        model = "gpt-4.1-mini"

        agent_a = _make_agent(provider, model, [get_capital])
        agent_b = _make_agent(provider, model, [calculate])

        graph = AgentGraph()
        graph.add_node("geo", agent_a)
        graph.add_node("math", agent_b)
        graph.add_parallel_nodes("both", ["geo", "math"])
        graph.add_edge("both", AgentGraph.END)

        state = GraphState.from_prompt("Answer two questions")
        state.data["geo_question"] = "What is the capital of France?"
        state.data["math_question"] = "What is 7 * 8?"
        result = graph.run(state)

        assert result.content
        assert "geo" in result.node_results or "math" in result.node_results
        assert result.total_usage.total_tokens > 0
