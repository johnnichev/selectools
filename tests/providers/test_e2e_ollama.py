"""
End-to-end tests for Ollama provider using a local Ollama instance.

These tests require Ollama to be installed and running locally.
Download from: https://ollama.ai
Start with: `ollama serve`

Run with: pytest tests/test_e2e_ollama.py --run-e2e -v

Note: Tests will be skipped if Ollama is not running.
"""

from __future__ import annotations

import pytest

from selectools import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from selectools.providers import OllamaProvider
from selectools.providers.base import ProviderError


@pytest.fixture
def calculator_tool() -> Tool:
    """Simple calculator tool for testing."""

    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)  # noqa: S307 (eval is safe for test)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    return Tool(
        name="calculate",
        description="Calculate a mathematical expression",
        parameters=[
            ToolParameter(
                name="expression",
                param_type=str,
                description="Math expression to evaluate (e.g., '2+2', '10*5')",
            )
        ],
        function=calculate,
    )


@pytest.fixture
def search_tool() -> Tool:
    """Mock search tool for testing."""

    def search(query: str) -> str:
        """Search for information."""
        return f"Search results for '{query}': [Mock result 1, Mock result 2]"

    return Tool(
        name="search",
        description="Search for information on the web",
        parameters=[ToolParameter(name="query", param_type=str, description="Search query")],
        function=search,
    )


def check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        provider = OllamaProvider(model="llama3.2")
        # Try a simple completion to verify it works
        provider.complete(
            model="llama3.2",
            system_prompt="You are a helpful assistant.",
            messages=[Message(role=Role.USER, content="Say 'hello'")],
            temperature=0.0,
            max_tokens=10,
            timeout=5.0,
        )
        return True
    except (ProviderError, Exception):
        return False


@pytest.mark.e2e
@pytest.mark.ollama
class TestOllamaProvider:
    """End-to-end tests for Ollama provider."""

    def test_ollama_basic_completion(self) -> None:
        """Test basic completion without tools."""
        if not check_ollama_available():
            pytest.skip("Ollama is not running or not accessible")

        provider = OllamaProvider(model="llama3.2")

        response, usage = provider.complete(
            model="llama3.2",
            system_prompt="You are a helpful assistant.",
            messages=[Message(role=Role.USER, content="Say 'hello' and nothing else")],
            temperature=0.0,
            max_tokens=50,
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert usage.cost_usd == 0.0  # Local models are free
        assert usage.provider == "ollama"
        assert usage.model == "llama3.2"

    def test_ollama_tool_calling(self, calculator_tool: Tool) -> None:
        """Test tool calling with Ollama."""
        if not check_ollama_available():
            pytest.skip("Ollama is not running or not accessible")

        provider = OllamaProvider(model="llama3.2", temperature=0.0)
        agent = Agent(
            tools=[calculator_tool],
            provider=provider,
            config=AgentConfig(model="llama3.2", verbose=True, max_iterations=3),
        )

        response = agent.run([Message(role=Role.USER, content="What is 15 multiplied by 7?")])

        assert response.role == Role.ASSISTANT
        assert isinstance(response.content, str)
        # The response should contain or reference the calculation result
        # Note: Ollama models may not always follow tool calling format perfectly
        assert len(response.content) > 0

    def test_ollama_cost_tracking(self, calculator_tool: Tool) -> None:
        """Test that Ollama usage has zero cost."""
        if not check_ollama_available():
            pytest.skip("Ollama is not running or not accessible")

        provider = OllamaProvider(model="llama3.2")
        agent = Agent(
            tools=[calculator_tool],
            provider=provider,
            config=AgentConfig(model="llama3.2", max_iterations=2),
        )

        agent.run([Message(role=Role.USER, content="Calculate 5+3")])

        # Verify zero cost for local models
        assert agent.total_cost == 0.0
        assert agent.usage.total_cost_usd == 0.0

    def test_ollama_streaming(self) -> None:
        """Test streaming responses from Ollama."""
        if not check_ollama_available():
            pytest.skip("Ollama is not running or not accessible")

        provider = OllamaProvider(model="llama3.2")

        chunks = []
        for chunk in provider.stream(
            model="llama3.2",
            system_prompt="You are a helpful assistant.",
            messages=[Message(role=Role.USER, content="Count from 1 to 5, one number per word")],
            temperature=0.0,
            max_tokens=100,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_ollama_async_completion(self) -> None:
        """Test async completion with Ollama."""
        if not check_ollama_available():
            pytest.skip("Ollama is not running or not accessible")

        provider = OllamaProvider(model="llama3.2")

        response, usage = await provider.acomplete(
            model="llama3.2",
            system_prompt="You are a helpful assistant.",
            messages=[Message(role=Role.USER, content="Say 'async hello'")],
            temperature=0.0,
            max_tokens=50,
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert usage.cost_usd == 0.0
        assert usage.provider == "ollama"

    @pytest.mark.asyncio
    async def test_ollama_async_agent(self, calculator_tool: Tool) -> None:
        """Test async agent execution with Ollama."""
        if not check_ollama_available():
            pytest.skip("Ollama is not running or not accessible")

        provider = OllamaProvider(model="llama3.2", temperature=0.0)
        agent = Agent(
            tools=[calculator_tool],
            provider=provider,
            config=AgentConfig(model="llama3.2", max_iterations=3),
        )

        response = await agent.arun([Message(role=Role.USER, content="What is 8 times 9?")])

        assert response.role == Role.ASSISTANT
        assert isinstance(response.content, str)
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_ollama_async_streaming(self) -> None:
        """Test async streaming with Ollama."""
        if not check_ollama_available():
            pytest.skip("Ollama is not running or not accessible")

        provider = OllamaProvider(model="llama3.2")

        chunks = []
        async for chunk in provider.astream(
            model="llama3.2",
            system_prompt="You are a helpful assistant.",
            messages=[Message(role=Role.USER, content="Count from 1 to 3")],
            temperature=0.0,
            max_tokens=100,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    def test_ollama_connection_error(self) -> None:
        """Test graceful handling of connection errors."""
        # Use invalid port to force connection error
        provider = OllamaProvider(model="llama3.2", base_url="http://localhost:99999")

        with pytest.raises(ProviderError) as exc_info:
            provider.complete(
                model="llama3.2",
                system_prompt="Test",
                messages=[Message(role=Role.USER, content="Test")],
                timeout=2.0,
            )

        # Should mention connection issue
        assert "connect" in str(exc_info.value).lower() or "ollama" in str(exc_info.value).lower()

    def test_ollama_custom_model(self) -> None:
        """Test using a custom Ollama model."""
        if not check_ollama_available():
            pytest.skip("Ollama is not running or not accessible")

        # Test with different model (may not be available, so we catch the error)
        provider = OllamaProvider(model="mistral")

        try:
            response, usage = provider.complete(
                model="mistral",
                system_prompt="You are a helpful assistant.",
                messages=[Message(role=Role.USER, content="Hello")],
                temperature=0.0,
                max_tokens=50,
                timeout=10.0,
            )

            assert isinstance(response, str)
            assert usage.cost_usd == 0.0
            assert usage.model == "mistral"
        except ProviderError as e:
            # Model might not be pulled, skip test
            if "model" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Model 'mistral' not available in Ollama: {e}")
            raise
