"""
Ollama provider adapter for local LLM inference.
"""

from __future__ import annotations

from typing import List

from ..pricing import calculate_cost
from ..types import Message, Role
from ..usage import UsageStats
from .base import Provider, ProviderError


class OllamaProvider(Provider):
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
        model: str = "llama3.2",
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

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[str, UsageStats]:
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            if "Connection" in error_msg or "connect" in error_msg.lower():
                raise ProviderError(
                    f"Failed to connect to Ollama at {self.base_url}. "
                    f"Make sure Ollama is running (try: ollama serve). Error: {exc}"
                ) from exc
            raise ProviderError(f"Ollama completion failed: {exc}") from exc

        content = response.choices[0].message.content

        # Extract usage stats (Ollama may or may not provide these)
        usage = response.usage
        usage_stats = UsageStats(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cost_usd=0.0,  # Local models are free
            model=model_name,
            provider="ollama",
        )

        return content or "", usage_stats

    def stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ):
        """Stream response chunks. Note: Does not return usage stats."""
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            if "Connection" in error_msg or "connect" in error_msg.lower():
                raise ProviderError(
                    f"Failed to connect to Ollama at {self.base_url}. "
                    f"Make sure Ollama is running (try: ollama serve). Error: {exc}"
                ) from exc
            raise ProviderError(f"Ollama streaming failed: {exc}") from exc

        for chunk in response:
            try:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta or not delta.content:
                    continue
                content = delta.content
                if isinstance(content, list):
                    content = "".join(
                        [part.text for part in content if getattr(part, "text", None)]
                    )
                yield content
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"Ollama stream parsing failed: {exc}") from exc

    def _format_messages(self, system_prompt: str, messages: List[Message]):
        payload = [{"role": "system", "content": system_prompt}]
        for message in messages:
            role = message.role.value
            if role == Role.TOOL.value:
                role = Role.ASSISTANT.value
            payload.append(
                {
                    "role": role,
                    "content": self._format_content(message),
                }
            )
        return payload

    def _format_content(self, message: Message):
        if message.image_base64:
            # Ollama supports vision in some models (e.g., llava)
            return [
                {"type": "text", "text": message.content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{message.image_base64}"},
                },
            ]
        return message.content

    # Async methods
    async def acomplete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[str, UsageStats]:
        """Async version of complete() using AsyncOpenAI client."""
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        try:
            response = await self._async_client.chat.completions.create(
                model=model_name,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:
            error_msg = str(exc)
            if "Connection" in error_msg or "connect" in error_msg.lower():
                raise ProviderError(
                    f"Failed to connect to Ollama at {self.base_url}. "
                    f"Make sure Ollama is running (try: ollama serve). Error: {exc}"
                ) from exc
            raise ProviderError(f"Ollama async completion failed: {exc}") from exc

        content = response.choices[0].message.content

        # Extract usage stats
        usage = response.usage
        usage_stats = UsageStats(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cost_usd=0.0,  # Local models are free
            model=model_name,
            provider="ollama",
        )

        return content or "", usage_stats

    async def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ):
        """Async version of stream() using AsyncOpenAI client."""
        formatted = self._format_messages(system_prompt=system_prompt, messages=messages)
        model_name = model or self.default_model

        try:
            response = await self._async_client.chat.completions.create(
                model=model_name,
                messages=formatted,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=timeout,
            )
        except Exception as exc:
            error_msg = str(exc)
            if "Connection" in error_msg or "connect" in error_msg.lower():
                raise ProviderError(
                    f"Failed to connect to Ollama at {self.base_url}. "
                    f"Make sure Ollama is running (try: ollama serve). Error: {exc}"
                ) from exc
            raise ProviderError(f"Ollama async streaming failed: {exc}") from exc

        async for chunk in response:
            try:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta or not delta.content:
                    continue
                content = delta.content
                if isinstance(content, list):
                    content = "".join(
                        [part.text for part in content if getattr(part, "text", None)]
                    )
                yield content
            except Exception as exc:
                raise ProviderError(f"Ollama async stream parsing failed: {exc}") from exc


__all__ = ["OllamaProvider"]
