"""
Azure OpenAI Service provider adapter.

Thin wrapper around :class:`OpenAIProvider` that swaps the standard
``openai.OpenAI`` client for ``openai.AzureOpenAI``.  All
complete/stream/acomplete/astream behaviour is inherited unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ..env import load_default_env
from ..exceptions import ProviderConfigurationError
from ..pricing import calculate_cost
from ..stability import beta
from .base import ProviderError
from .openai_provider import OpenAIProvider, _uses_max_completion_tokens


@beta
class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI Service provider.

    Uses the OpenAI SDK's built-in Azure support.  All complete/stream/
    acomplete/astream methods are inherited from :class:`OpenAIProvider`.

    Example::

        >>> from selectools.providers import AzureOpenAIProvider
        >>> provider = AzureOpenAIProvider(
        ...     azure_endpoint="https://my-resource.openai.azure.com",
        ...     api_key="...",
        ...     azure_deployment="gpt-4o",
        ... )
        >>> agent = Agent(tools=[...], provider=provider)
    """

    name = "azure-openai"

    def __init__(
        self,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-10-21",
        azure_deployment: str | None = None,
        azure_ad_token: str | None = None,
    ):
        """Initialise the Azure OpenAI provider.

        Args:
            azure_endpoint: The Azure resource endpoint, e.g.
                ``https://my-resource.openai.azure.com``.
                Falls back to the ``AZURE_OPENAI_ENDPOINT`` env var.
            api_key: Azure API key.  Falls back to ``AZURE_OPENAI_API_KEY``.
                Can be omitted when *azure_ad_token* is provided.
            api_version: Azure OpenAI API version string.
            azure_deployment: The deployment name to use as the default model.
                Falls back to ``AZURE_OPENAI_DEPLOYMENT``.
            azure_ad_token: An Azure Active Directory token for AAD-based auth.
                When set, *api_key* is not required.

        Raises:
            ProviderConfigurationError: If the endpoint or credentials are
                missing.
        """
        load_default_env()

        self._azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self._azure_endpoint:
            raise ProviderConfigurationError(
                provider_name="Azure OpenAI",
                missing_config="Azure endpoint",
                env_var="AZURE_OPENAI_ENDPOINT",
            )

        resolved_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not resolved_key and not azure_ad_token:
            raise ProviderConfigurationError(
                provider_name="Azure OpenAI",
                missing_config="API key or Azure AD token",
                env_var="AZURE_OPENAI_API_KEY",
            )

        try:
            from openai import AsyncAzureOpenAI, AzureOpenAI
        except ImportError as exc:
            raise ProviderError(
                "openai package not installed. Install with `pip install openai`."
            ) from exc

        client_kwargs: Dict[str, Any] = {
            "azure_endpoint": self._azure_endpoint,
            "api_version": api_version,
        }
        if azure_ad_token:
            client_kwargs["azure_ad_token"] = azure_ad_token
            # AzureOpenAI still requires api_key to be a non-empty string
            # when azure_ad_token is used; pass a placeholder.
            client_kwargs["api_key"] = resolved_key or "aad"
        else:
            client_kwargs["api_key"] = resolved_key

        # Bypass OpenAIProvider.__init__ — we wire the clients ourselves.
        self._client = AzureOpenAI(**client_kwargs)
        self._async_client = AsyncAzureOpenAI(**client_kwargs)
        self.default_model = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.api_key = resolved_key or ""

    # -- template method overrides -------------------------------------------

    def _get_provider_name(self) -> str:
        return "azure-openai"

    def _wrap_error(self, exc: Exception, operation: str) -> ProviderError:
        return ProviderError(f"Azure OpenAI {operation} failed: {exc}")


__all__ = ["AzureOpenAIProvider"]
