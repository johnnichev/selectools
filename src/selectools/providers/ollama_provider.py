"""
Ollama provider adapter for local LLM inference.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict

from ..models import Ollama as OllamaModels
from ..stability import stable
from ..types import ToolCall
from ._openai_compat import _OpenAICompatibleBase
from .base import ProviderError


@stable
class OllamaProvider(_OpenAICompatibleBase):
    """
    Adapter for Ollama local models using OpenAI-compatible API.

    Ollama provides a local inference server that supports the OpenAI Chat Completions API format.
    This provider connects to a running Ollama instance (typically at http://localhost:11434).

    Features:
    - No API key required (runs locally)
    - Zero cost ($0.00 for all models)
    - Privacy-preserving (no data sent to cloud)
    - Supports streaming and async operations

    Example:
        >>> from selectools import Agent, AgentConfig
        >>> from selectools.providers import OllamaProvider
        >>>
        >>> provider = OllamaProvider(model="llama3.2")
        >>> agent = Agent(tools=[...], provider=provider)
        >>> response = agent.run([...])

    Note:
        Requires Ollama to be installed and running. Download from https://ollama.ai
        Start Ollama with: `ollama serve`
    """

    name = "ollama"
    supports_streaming = True
    supports_async = True

    def __init__(
        self,
        model: str = OllamaModels.LLAMA_3_2.id,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "llama3.2", "mistral", "codellama").
                   Must be already pulled via `ollama pull <model>`.
            base_url: Base URL for Ollama server. Default: http://localhost:11434
            temperature: Sampling temperature for generation. Default: 0.7

        Raises:
            ProviderError: If openai package is not installed or connection fails.
        """
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as exc:
            raise ProviderError(
                "openai package not installed. Install with `pip install openai`."
            ) from exc

        # Ollama uses OpenAI-compatible API at /v1 endpoint
        self._client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",  # Ollama doesn't require API key, but OpenAI client needs one
        )
        self._async_client = AsyncOpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",
        )
        self.default_model = model
        self.base_url = base_url

    # -- template method overrides -------------------------------------------

    def _get_token_key(self, model: str) -> str:
        return "max_tokens"  # Ollama always uses max_tokens

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0  # Local models are free

    def _get_provider_name(self) -> str:
        return "ollama"

    def _wrap_error(self, exc: Exception, operation: str) -> ProviderError:
        error_msg = str(exc)
        if "Connection" in error_msg or "connect" in error_msg.lower():
            return ProviderError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running (try: ollama serve). Error: {exc}"
            )
        return ProviderError(f"Ollama {operation} failed: {exc}")

    def _parse_tool_call_id(self, tc: Any) -> str:
        return tc.id if tc.id else f"call_{uuid.uuid4().hex}"

    def _parse_tool_call_arguments(self, tc: Any) -> dict:
        """Ollama may return arguments as a dict or a JSON string."""
        try:
            if isinstance(tc.function.arguments, str):
                return json.loads(tc.function.arguments)  # type: ignore[no-any-return]
            else:
                return tc.function.arguments  # type: ignore[no-any-return]
        except (json.JSONDecodeError, TypeError):
            return {}

    # -- tool-call ID helpers (Ollama may not provide IDs) --------------------

    def _format_tool_call_id(self, tc: ToolCall) -> str:
        return tc.id or f"call_{uuid.uuid4().hex}"

    def _initial_tool_call_id(self, tc_delta: Any) -> str:
        return tc_delta.id or f"call_{uuid.uuid4().hex}"


__all__ = ["OllamaProvider"]
