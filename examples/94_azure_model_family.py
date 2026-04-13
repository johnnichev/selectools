"""
Azure OpenAI with Model Family — correct token parameter for custom deployments.

Since v0.22.0 (BUG-28), AzureOpenAIProvider accepts a `model_family` parameter
so deployments with custom names (e.g., "prod-chat") still use the correct
`max_completion_tokens` parameter for GPT-5-family models.

Prerequisites: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
Run: python examples/94_azure_model_family.py
"""

from selectools import Agent
from selectools.providers.azure_openai_provider import AzureOpenAIProvider


def main() -> None:
    # The problem: Azure deployments use custom names
    # A deployment named "prod-chat" running gpt-5-mini won't match
    # the "gpt-5" prefix that selectools uses for family detection.
    # Without model_family, this deployment would receive max_tokens
    # instead of max_completion_tokens, causing a BadRequestError.

    print("Azure OpenAI Model Family Detection:")
    print()

    # Solution: pass model_family to tell selectools the underlying model
    provider = AzureOpenAIProvider.__new__(AzureOpenAIProvider)
    provider._model_family = None
    print(f"  Without model_family:")
    print(f"    'prod-chat'  -> {provider._get_token_key('prod-chat')}")
    print(f"    'gpt-5-mini' -> {provider._get_token_key('gpt-5-mini')}")

    provider._model_family = "gpt-5"
    print(f"\n  With model_family='gpt-5':")
    print(f"    'prod-chat'  -> {provider._get_token_key('prod-chat')}")
    print(f"    'gpt-5-mini' -> {provider._get_token_key('gpt-5-mini')}")

    print("\n✓ model_family overrides deployment-name-based detection")
    print()
    print("Usage:")
    print("  provider = AzureOpenAIProvider(")
    print('      azure_endpoint="https://my-resource.openai.azure.com",')
    print('      azure_deployment="prod-chat",')
    print('      model_family="gpt-5",  # Underlying model family')
    print("  )")


if __name__ == "__main__":
    main()
