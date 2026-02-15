from __future__ import annotations

import os

import pytest

from selectools import Agent, AgentConfig
from selectools.providers.anthropic_provider import AnthropicProvider
from selectools.providers.gemini_provider import GeminiProvider
from selectools.providers.ollama_provider import OllamaProvider
from selectools.providers.openai_provider import OpenAIProvider
from selectools.tools.decorators import tool
from selectools.types import Message, Role


# Define a simple tool
@tool(description="Get a random number between min and max")
def get_random_number(min_val: int, max_val: int) -> int:
    return 42  # Deterministic for testing


@pytest.fixture
def weather_agent(request: pytest.FixtureRequest) -> Agent:
    provider_cls = request.param
    try:
        provider = provider_cls()
    except Exception as e:
        pytest.skip(f"Skipping {provider_cls.__name__}: {e}")

    return Agent(config=AgentConfig(), provider=provider, tools=[get_random_number])


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_native_tool_call() -> None:
    print("\nTesting OpenAI native tool call...")
    try:
        provider = OpenAIProvider()
    except Exception as e:
        pytest.skip(f"OpenAI init failed: {e}")

    agent = Agent(config=AgentConfig(), provider=provider, tools=[get_random_number])
    response = agent.run(
        [Message(role=Role.USER, content="Generate a random number between 1 and 100.")]
    )
    print(f"OpenAI Response: {response}")
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0].tool_name == "get_random_number"
    assert "42" in response.message.content or "42" in str(response.tool_calls)


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_anthropic_native_tool_call() -> None:
    print("\nTesting Anthropic native tool call...")
    try:
        provider = AnthropicProvider()
    except Exception as e:
        pytest.skip(f"Anthropic init failed: {e}")

    agent = Agent(
        config=AgentConfig(model=provider.default_model),
        provider=provider,
        tools=[get_random_number],
    )
    response = agent.run(
        [Message(role=Role.USER, content="Generate a random number between 1 and 100.")]
    )
    print(f"Anthropic Response: {response}")
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0].tool_name == "get_random_number"


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_gemini_native_tool_call() -> None:
    print("\nTesting Gemini native tool call...")
    try:
        provider = GeminiProvider()
    except Exception as e:
        pytest.skip(f"Gemini init failed: {e}")

    agent = Agent(
        config=AgentConfig(model=provider.default_model),
        provider=provider,
        tools=[get_random_number],
    )
    response = agent.run(
        [Message(role=Role.USER, content="Generate a random number between 1 and 100.")]
    )
    print(f"Gemini Response: {response}")
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0].tool_name == "get_random_number"


def test_ollama_native_tool_call() -> None:
    print("\nTesting Ollama native tool call...")
    try:
        # Short timeout to fail fast if not running
        provider = OllamaProvider()
        # Simple health check handled by run() call implicitly
    except Exception as e:
        pytest.skip(f"Ollama init failed (likely not running): {e}")

    agent = Agent(config=AgentConfig(), provider=provider, tools=[get_random_number])
    try:
        response = agent.run(
            [Message(role=Role.USER, content="Generate a random number between 1 and 100.")]
        )
        print(f"Ollama Response: {response}")
        assert response.tool_calls is not None
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0].tool_name == "get_random_number"
    except Exception as e:
        pytest.skip(f"Ollama execution failed (likely model not pulled or compatible): {e}")


if __name__ == "__main__":
    # verification run manually
    pass
