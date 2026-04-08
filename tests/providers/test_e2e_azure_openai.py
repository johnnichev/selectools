"""End-to-end tests for AzureOpenAIProvider against a real Azure endpoint.

``test_azure_openai.py`` mocks the OpenAI client. This file uses the real
``AzureOpenAI`` client and hits an actual Azure OpenAI Service deployment.

Required env vars:
    - AZURE_OPENAI_ENDPOINT: e.g. https://my-resource.openai.azure.com
    - AZURE_OPENAI_API_KEY: Azure API key
    - AZURE_OPENAI_DEPLOYMENT: deployment name (defaults to "gpt-4o-mini" if missing)

Run with:

    pytest tests/providers/test_e2e_azure_openai.py --run-e2e -v
"""

from __future__ import annotations

import os

import pytest

from selectools import Agent, AgentConfig, tool
from selectools.providers.azure_openai_provider import AzureOpenAIProvider

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def azure_or_skip() -> None:
    if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        pytest.skip("AZURE_OPENAI_ENDPOINT not set — skipping Azure e2e")
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        pytest.skip("AZURE_OPENAI_API_KEY not set — skipping Azure e2e")


@tool()
def _noop() -> str:
    """Return a fixed string."""
    return "noop"


class TestAzureOpenAIRealEndpoint:
    def test_simple_completion(self, azure_or_skip: None) -> None:
        """Real Azure OpenAI call returns a non-empty response."""
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        provider = AzureOpenAIProvider(azure_deployment=deployment)
        agent = Agent(
            tools=[_noop],
            provider=provider,
            config=AgentConfig(model=deployment, max_tokens=20),
        )
        result = agent.run("Reply with exactly the word OK and nothing else.")
        assert result.content
        assert result.usage.total_tokens > 0

    def test_tool_calling_round_trip(self, azure_or_skip: None) -> None:
        """Real Azure OpenAI invokes a tool and returns a final answer."""
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

        @tool()
        def get_capital(country: str) -> str:
            """Return the capital of a country."""
            capitals = {"france": "Paris", "japan": "Tokyo", "italy": "Rome"}
            return capitals.get(country.lower(), "unknown")

        agent = Agent(
            tools=[get_capital],
            provider=AzureOpenAIProvider(azure_deployment=deployment),
            config=AgentConfig(model=deployment, max_tokens=100),
        )
        result = agent.run("What is the capital of France? Use the get_capital tool.")
        assert result.content
        assert "Paris" in result.content or "paris" in result.content.lower()
