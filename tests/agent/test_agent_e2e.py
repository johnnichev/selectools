"""
End-to-End integration tests for Agent v0.9.0 features.

Tests cover:
1. Custom System Prompt Injection
2. Structured AgentResult
3. Reusable Agent Instances (reset)

These tests are marked with @pytest.mark.e2e and require an OPENAI_API_KEY environment variable.
To run them: pytest --run-e2e tests/test_agent_e2e.py
"""

import os

import pytest

from selectools import Agent, AgentConfig, AgentResult, Message, Role, tool
from selectools.providers.openai_provider import OpenAIProvider
from selectools.providers.stubs import LocalProvider

# Helper to skip if not running E2E
pytestmark = pytest.mark.e2e


@pytest.fixture
def e2e_provider():
    """Returns a real OpenAIProvider if key exists, else skips."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAIProvider(api_key=api_key, default_model="gpt-4o-mini")


@tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    if "london" in city.lower():
        return "rainy, 15C"
    if "paris" in city.lower():
        return "sunny, 20C"
    return "unknown"


class TestAgentE2E:
    def test_custom_system_prompt_e2e(self, e2e_provider):
        """Verify custom system prompt changes agent behavior."""
        agent = Agent(
            tools=[get_weather],
            provider=e2e_provider,
            config=AgentConfig(
                system_prompt="You are a pirate. Always speak like a pirate. Start response with 'Ahoy'."
            ),
        )

        result = agent.run([Message(role=Role.USER, content="Hello")])

        assert isinstance(result, AgentResult)
        assert "Ahoy" in result.content
        assert len(result.content) > 10

    def test_structured_agent_result_e2e(self, e2e_provider):
        """Verify tool calls are correctly captured in AgentResult."""
        agent = Agent(
            tools=[get_weather],
            provider=e2e_provider,
            config=AgentConfig(max_iterations=3),
        )

        result = agent.run([Message(role=Role.USER, content="What is the weather in London?")])

        assert isinstance(result, AgentResult)
        assert "rainy" in result.content.lower()

        # Verify tool tracking
        assert result.iterations > 1
        assert len(result.tool_calls) >= 1

        # Check specific tool call details
        weather_call = next((tc for tc in result.tool_calls if tc.tool_name == "get_weather"), None)
        assert weather_call is not None
        assert "london" in weather_call.parameters["city"].lower()

    def test_agent_reset_e2e(self, e2e_provider):
        """Verify agent reset clears history and allows reuse."""
        agent = Agent(
            tools=[get_weather],
            provider=e2e_provider,
            config=AgentConfig(max_iterations=3),
        )

        # First conversation
        agent.run([Message(role=Role.USER, content="The magic number is 42.")])

        # Verify memory without reset
        r2 = agent.run([Message(role=Role.USER, content="What is the magic number?")])
        assert "42" in r2.content

        # Reset
        agent.reset()

        # Verify clean slate
        r3 = agent.run([Message(role=Role.USER, content="What is the magic number?")])
        assert "42" not in r3.content  # Should not know the number anymore
        assert agent.total_tokens > 0  # Usage from r3 only (previous cleared) but still > 0

    def test_reusable_agent_multiple_turns(self, e2e_provider):
        """Verify reusable agent handles multiple independent tasks sequentially."""
        agent = Agent(tools=[get_weather], provider=e2e_provider)

        # Task 1
        r1 = agent.run([Message(role=Role.USER, content="Weather in Paris?")])
        assert "sunny" in r1.content.lower()

        agent.reset()

        # Task 2
        r2 = agent.run([Message(role=Role.USER, content="Weather in London?")])
        assert "rainy" in r2.content.lower()

        # Ensure no cross-contamination
        assert r1.tool_calls[0].parameters["city"].lower() == "paris"
        assert r2.tool_calls[0].parameters["city"].lower() == "london"
