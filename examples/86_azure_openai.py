#!/usr/bin/env python3
"""
Azure OpenAI Provider -- use OpenAI models via Azure endpoints.

Requires: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars.
Run: python examples/86_azure_openai.py
"""

print("=== Azure OpenAI Provider Example ===\n")

print(
    """
from selectools import Agent, AgentConfig
from selectools.providers import AzureOpenAIProvider

# Option 1: Explicit configuration
provider = AzureOpenAIProvider(
    azure_endpoint="https://my-resource.openai.azure.com",
    api_key="your-azure-api-key",
    azure_deployment="gpt-4o",  # Your deployment name
    api_version="2024-10-21",
)

# Option 2: Environment variables
# Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
provider = AzureOpenAIProvider(azure_deployment="gpt-4o")

# Option 3: Azure AD authentication (no API key needed)
provider = AzureOpenAIProvider(
    azure_endpoint="https://my-resource.openai.azure.com",
    azure_ad_token="your-aad-token",
    azure_deployment="gpt-4o",
)

# Use like any other provider
agent = Agent(
    tools=[],
    provider=provider,
    config=AgentConfig(model="gpt-4o"),
)
result = agent.run("Hello from Azure!")
print(result.content)

# Supports all features: streaming, tool calling, structured output
async for chunk in agent.astream("Stream from Azure"):
    print(chunk.content, end="")
"""
)

print("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY to use.")
print("Done!")
