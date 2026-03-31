"""
OpenAI provider adapter for the tool-calling library.

Handles the ``max_tokens`` -> ``max_completion_tokens`` migration
automatically based on model family.  Newer models (GPT-5.x, GPT-4.1,
o-series, codex) reject the legacy ``max_tokens`` parameter.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from ..env import load_default_env
from ..exceptions import ProviderConfigurationError
from ..models import OpenAI as OpenAIModels
from ..pricing import calculate_cost
from ..stability import stable
from ._openai_compat import _OpenAICompatibleBase
from .base import ProviderError

_MAX_COMPLETION_TOKENS_PREFIXES: Tuple[str, ...] = (
    "gpt-5",
    "gpt-4.1",
    "o1",
    "o3",
    "o4",
    "codex",
)


def _uses_max_completion_tokens(model: str) -> bool:
    """Return True if *model* requires ``max_completion_tokens`` instead of ``max_tokens``."""
    return any(model.startswith(p) for p in _MAX_COMPLETION_TOKENS_PREFIXES)


@stable
class OpenAIProvider(_OpenAICompatibleBase):
    """Adapter that speaks to OpenAI's Chat Completions API."""

    name = "openai"
    supports_streaming = True
    supports_async = True

    def __init__(self, api_key: str | None = None, default_model: str = OpenAIModels.GPT_5_MINI.id):
        load_default_env()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderConfigurationError(
                provider_name="OpenAI",
                missing_config="API key",
                env_var="OPENAI_API_KEY",
            )

        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as exc:
            raise ProviderError(
                "openai package not installed. Install with `pip install openai`."
            ) from exc

        self._client = OpenAI(api_key=self.api_key)
        self._async_client = AsyncOpenAI(api_key=self.api_key)
        self.default_model = default_model

    # -- template method overrides -------------------------------------------

    def _get_token_key(self, model: str) -> str:
        return "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return calculate_cost(model, prompt_tokens, completion_tokens)

    def _get_provider_name(self) -> str:
        return "openai"

    def _wrap_error(self, exc: Exception, operation: str) -> ProviderError:
        return ProviderError(f"OpenAI {operation} failed: {exc}")

    def _parse_tool_call_id(self, tc: Any) -> str:
        return tc.id

    def _build_astream_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        args["stream_options"] = {"include_usage": True}
        return args


__all__ = ["OpenAIProvider"]
