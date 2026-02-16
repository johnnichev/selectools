"""
End-to-End tests using real provider APIs.

These tests require real API keys and make actual API calls.
They are skipped by default and must be run explicitly:

    # Run all e2e tests (requires all API keys):
    pytest tests/test_e2e_providers.py -v --run-e2e

    # Run only OpenAI tests:
    pytest tests/test_e2e_providers.py -v --run-e2e -k openai

    # Run only Anthropic tests:
    pytest tests/test_e2e_providers.py -v --run-e2e -k anthropic

    # Run only Gemini tests:
    pytest tests/test_e2e_providers.py -v --run-e2e -k gemini

Required environment variables:
    - OPENAI_API_KEY: For OpenAI tests
    - ANTHROPIC_API_KEY: For Anthropic tests
    - GEMINI_API_KEY (or GOOGLE_API_KEY): For Gemini tests
"""

from __future__ import annotations

import os

import pytest

from selectools import Agent, AgentConfig, Message, Role, Tool, tool
from selectools.memory import ConversationMemory

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def calculator_tool() -> Tool:
    """Simple calculator tool for testing."""

    @tool(
        description="Perform basic arithmetic calculations. Supports add, subtract, multiply, divide."
    )
    def calculator(operation: str, a: float, b: float) -> str:
        """
        Perform a calculation.

        Args:
            operation: One of 'add', 'subtract', 'multiply', 'divide'
            a: First number
            b: Second number
        """
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            result = a / b
        else:
            return f"Error: Unknown operation '{operation}'"
        return f"Result: {result}"

    return calculator


@pytest.fixture
def weather_tool() -> Tool:
    """Mock weather tool for testing."""

    @tool(description="Get current weather for a city. Returns temperature and conditions.")
    def get_weather(city: str) -> str:
        """
        Get weather for a city.

        Args:
            city: City name (e.g., 'London', 'New York', 'Tokyo')
        """
        # Mock responses for testing
        weather_data = {
            "london": "London: 15°C, Cloudy",
            "new york": "New York: 22°C, Sunny",
            "tokyo": "Tokyo: 28°C, Humid",
            "paris": "Paris: 18°C, Partly Cloudy",
        }
        city_lower = city.lower()
        if city_lower in weather_data:
            return weather_data[city_lower]
        return f"{city}: Weather data not available"

    return get_weather


# =============================================================================
# OpenAI Provider Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.openai
class TestOpenAIProvider:
    """End-to-end tests for OpenAI provider."""

    @pytest.fixture(autouse=True)
    def check_api_key(self) -> None:
        """Skip if OpenAI API key is not available."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_openai_basic_completion(self) -> None:
        """Test basic completion with OpenAI."""
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(default_model="gpt-4o-mini")
        response, usage = provider.complete(
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant. Be brief.",
            messages=[Message(role=Role.USER, content="Say 'hello' and nothing else.")],
            max_tokens=50,
        )

        assert response is not None
        assert len(response.content) > 0
        assert "hello" in response.content.lower()
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.cost_usd > 0
        print(f"\n  Response: {response}")
        print(f"  Usage: {usage.total_tokens} tokens, ${usage.cost_usd:.6f}")

    def test_openai_tool_calling(self, calculator_tool: Tool) -> None:
        """Test tool calling with OpenAI."""
        from selectools.providers.openai_provider import OpenAIProvider

        agent = Agent(
            tools=[calculator_tool],
            provider=OpenAIProvider(default_model="gpt-4o-mini"),
            config=AgentConfig(model="gpt-4o-mini", max_iterations=3, verbose=True),
        )

        response = agent.run(
            [
                Message(
                    role=Role.USER, content="What is 15 multiplied by 7? Use the calculator tool."
                )
            ]
        )

        assert response.role == Role.ASSISTANT
        assert "105" in response.content
        assert agent.total_tokens > 0
        print(f"\n  Response: {response.content}")
        print(f"  Total cost: ${agent.total_cost:.6f}")

    def test_openai_streaming(self, weather_tool: Tool) -> None:
        """Test streaming with OpenAI."""
        from selectools.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(default_model="gpt-4o-mini")
        chunks = []

        for chunk in provider.stream(
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant. Be brief.",
            messages=[Message(role=Role.USER, content="Count from 1 to 5.")],
            max_tokens=50,
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(chunks) > 1  # Should be multiple chunks
        assert len(full_response) > 0
        print(f"\n  Streamed response ({len(chunks)} chunks): {full_response}")

    @pytest.mark.asyncio
    async def test_openai_async(self, calculator_tool: Tool) -> None:
        """Test async execution with OpenAI."""
        from selectools.providers.openai_provider import OpenAIProvider

        agent = Agent(
            tools=[calculator_tool],
            provider=OpenAIProvider(default_model="gpt-4o-mini"),
            config=AgentConfig(model="gpt-4o-mini", max_iterations=3),
        )

        response = await agent.arun(
            [Message(role=Role.USER, content="Calculate 100 divided by 4 using the calculator.")]
        )

        assert response.role == Role.ASSISTANT
        assert "25" in response.content
        print(f"\n  Async response: {response.content}")

    def test_openai_with_memory(self, weather_tool: Tool) -> None:
        """Test conversation memory with OpenAI."""
        from selectools.providers.openai_provider import OpenAIProvider

        memory = ConversationMemory(max_messages=10)
        agent = Agent(
            tools=[weather_tool],
            provider=OpenAIProvider(default_model="gpt-4o-mini"),
            config=AgentConfig(model="gpt-4o-mini", max_iterations=3),
            memory=memory,
        )

        # First turn
        response1 = agent.run([Message(role=Role.USER, content="What's the weather in London?")])
        print(f"\n  Turn 1: {response1.content}")

        # Second turn - should remember context
        response2 = agent.run([Message(role=Role.USER, content="How about Paris?")])
        print(f"  Turn 2: {response2.content}")

        assert len(memory) >= 4  # At least 2 user + 2 assistant messages
        assert "paris" in response2.content.lower() or "18" in response2.content


# =============================================================================
# Anthropic Provider Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.anthropic
class TestAnthropicProvider:
    """End-to-end tests for Anthropic provider."""

    @pytest.fixture(autouse=True)
    def check_api_key(self) -> None:
        """Skip if Anthropic API key is not available."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_anthropic_basic_completion(self) -> None:
        """Test basic completion with Anthropic."""
        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(default_model="claude-3-haiku-20240307")
        response, usage = provider.complete(
            model="claude-3-haiku-20240307",
            system_prompt="You are a helpful assistant. Be very brief.",
            messages=[Message(role=Role.USER, content="Say 'hello' and nothing else.")],
            max_tokens=50,
        )

        assert response is not None
        assert len(response.content) > 0
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.cost_usd > 0
        print(f"\n  Response: {response}")
        print(f"  Usage: {usage.total_tokens} tokens, ${usage.cost_usd:.6f}")

    def test_anthropic_tool_calling(self, calculator_tool: Tool) -> None:
        """Test tool calling with Anthropic."""
        from selectools.providers.anthropic_provider import AnthropicProvider

        agent = Agent(
            tools=[calculator_tool],
            provider=AnthropicProvider(default_model="claude-3-haiku-20240307"),
            config=AgentConfig(model="claude-3-haiku-20240307", max_iterations=3, verbose=True),
        )

        response = agent.run(
            [Message(role=Role.USER, content="What is 42 plus 58? Use the calculator tool.")]
        )

        assert response.role == Role.ASSISTANT
        assert "100" in response.content
        assert agent.total_tokens > 0
        print(f"\n  Response: {response.content}")
        print(f"  Total cost: ${agent.total_cost:.6f}")

    def test_anthropic_streaming(self) -> None:
        """Test streaming with Anthropic."""
        from selectools.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(default_model="claude-3-haiku-20240307")
        chunks = []

        for chunk in provider.stream(
            model="claude-3-haiku-20240307",
            system_prompt="You are a helpful assistant. Be brief.",
            messages=[Message(role=Role.USER, content="List 3 colors.")],
            max_tokens=50,
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(chunks) > 1
        assert len(full_response) > 0
        print(f"\n  Streamed response ({len(chunks)} chunks): {full_response}")

    @pytest.mark.asyncio
    async def test_anthropic_async(self, calculator_tool: Tool) -> None:
        """Test async execution with Anthropic."""
        from selectools.providers.anthropic_provider import AnthropicProvider

        agent = Agent(
            tools=[calculator_tool],
            provider=AnthropicProvider(default_model="claude-3-haiku-20240307"),
            config=AgentConfig(model="claude-3-haiku-20240307", max_iterations=3),
        )

        response = await agent.arun(
            [Message(role=Role.USER, content="Calculate 200 minus 50 using the calculator.")]
        )

        assert response.role == Role.ASSISTANT
        assert "150" in response.content
        print(f"\n  Async response: {response.content}")

    def test_anthropic_with_memory(self, weather_tool: Tool) -> None:
        """Test conversation memory with Anthropic."""
        from selectools.providers.anthropic_provider import AnthropicProvider

        memory = ConversationMemory(max_messages=10)
        agent = Agent(
            tools=[weather_tool],
            provider=AnthropicProvider(default_model="claude-3-haiku-20240307"),
            config=AgentConfig(model="claude-3-haiku-20240307", max_iterations=3),
            memory=memory,
        )

        response1 = agent.run([Message(role=Role.USER, content="What's the weather in Tokyo?")])
        print(f"\n  Turn 1: {response1.content}")

        response2 = agent.run([Message(role=Role.USER, content="And in New York?")])
        print(f"  Turn 2: {response2.content}")

        assert len(memory) >= 4


# =============================================================================
# Gemini Provider Tests
# =============================================================================


@pytest.mark.e2e
@pytest.mark.gemini
class TestGeminiProvider:
    """End-to-end tests for Gemini provider."""

    @pytest.fixture(autouse=True)
    def check_api_key(self) -> None:
        """Skip if Gemini API key is not available."""
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set")

    def test_gemini_basic_completion(self) -> None:
        """Test basic completion with Gemini."""
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(default_model="gemini-2.0-flash")
        response, usage = provider.complete(
            model="gemini-2.0-flash",
            system_prompt="You are a helpful assistant. Be very brief.",
            messages=[Message(role=Role.USER, content="Say 'hello' and nothing else.")],
            max_tokens=50,
        )

        assert response is not None
        assert len(response.content) > 0
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.cost_usd >= 0  # Gemini can be free tier
        print(f"\n  Response: {response}")
        print(f"  Usage: {usage.total_tokens} tokens, ${usage.cost_usd:.6f}")

    def test_gemini_tool_calling(self, calculator_tool: Tool) -> None:
        """Test tool calling with Gemini."""
        from selectools.providers.gemini_provider import GeminiProvider

        agent = Agent(
            tools=[calculator_tool],
            provider=GeminiProvider(default_model="gemini-2.0-flash"),
            config=AgentConfig(model="gemini-2.0-flash", max_iterations=3, verbose=True),
        )

        response = agent.run(
            [Message(role=Role.USER, content="What is 8 times 9? Use the calculator tool.")]
        )

        assert response.role == Role.ASSISTANT
        assert "72" in response.content
        assert agent.total_tokens > 0
        print(f"\n  Response: {response.content}")
        print(f"  Total cost: ${agent.total_cost:.6f}")

    def test_gemini_streaming(self) -> None:
        """Test streaming with Gemini."""
        from selectools.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(default_model="gemini-2.0-flash")
        chunks = []

        for chunk in provider.stream(
            model="gemini-2.0-flash",
            system_prompt="You are a helpful assistant. Be brief.",
            messages=[Message(role=Role.USER, content="Name 3 fruits.")],
            max_tokens=50,
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(chunks) >= 1
        assert len(full_response) > 0
        print(f"\n  Streamed response ({len(chunks)} chunks): {full_response}")

    @pytest.mark.asyncio
    async def test_gemini_async(self, calculator_tool: Tool) -> None:
        """Test async execution with Gemini."""
        from selectools.providers.gemini_provider import GeminiProvider

        agent = Agent(
            tools=[calculator_tool],
            provider=GeminiProvider(default_model="gemini-2.0-flash"),
            config=AgentConfig(model="gemini-2.0-flash", max_iterations=3),
        )

        response = await agent.arun(
            [Message(role=Role.USER, content="Calculate 144 divided by 12 using the calculator.")]
        )

        assert response.role == Role.ASSISTANT
        assert "12" in response.content
        print(f"\n  Async response: {response.content}")

    def test_gemini_with_memory(self, weather_tool: Tool) -> None:
        """Test conversation memory with Gemini."""
        from selectools.providers.gemini_provider import GeminiProvider

        memory = ConversationMemory(max_messages=10)
        agent = Agent(
            tools=[weather_tool],
            provider=GeminiProvider(default_model="gemini-2.0-flash"),
            config=AgentConfig(model="gemini-2.0-flash", max_iterations=3),
            memory=memory,
        )

        response1 = agent.run([Message(role=Role.USER, content="What's the weather in Paris?")])
        print(f"\n  Turn 1: {response1.content}")

        response2 = agent.run([Message(role=Role.USER, content="Compare it to London.")])
        print(f"  Turn 2: {response2.content}")

        assert len(memory) >= 4


# =============================================================================
# Cross-Provider Tests
# =============================================================================


@pytest.mark.e2e
class TestCrossProvider:
    """Tests that run across multiple providers."""

    def test_cost_tracking_comparison(self, calculator_tool: Tool) -> None:
        """Compare cost tracking across providers."""
        results = {}

        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            from selectools.providers.openai_provider import OpenAIProvider

            agent = Agent(
                tools=[calculator_tool],
                provider=OpenAIProvider(default_model="gpt-4o-mini"),
                config=AgentConfig(model="gpt-4o-mini", max_iterations=2),
            )
            agent.run([Message(role=Role.USER, content="What is 2+2? Use calculator.")])
            results["openai"] = {
                "tokens": agent.total_tokens,
                "cost": agent.total_cost,
            }

        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            from selectools.providers.anthropic_provider import AnthropicProvider

            agent = Agent(
                tools=[calculator_tool],
                provider=AnthropicProvider(default_model="claude-3-haiku-20240307"),
                config=AgentConfig(model="claude-3-haiku-20240307", max_iterations=2),
            )
            agent.run([Message(role=Role.USER, content="What is 2+2? Use calculator.")])
            results["anthropic"] = {
                "tokens": agent.total_tokens,
                "cost": agent.total_cost,
            }

        # Gemini
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            from selectools.providers.gemini_provider import GeminiProvider

            agent = Agent(
                tools=[calculator_tool],
                provider=GeminiProvider(default_model="gemini-2.0-flash"),
                config=AgentConfig(model="gemini-2.0-flash", max_iterations=2),
            )
            agent.run([Message(role=Role.USER, content="What is 2+2? Use calculator.")])
            results["gemini"] = {
                "tokens": agent.total_tokens,
                "cost": agent.total_cost,
            }

        if not results:
            pytest.skip("No API keys available")

        print("\n  Cost Comparison:")
        for provider, data in results.items():
            print(f"    {provider}: {data['tokens']} tokens, ${data['cost']:.6f}")


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    # When run directly, execute e2e tests
    pytest.main([__file__, "-v", "--run-e2e", "-s"])
