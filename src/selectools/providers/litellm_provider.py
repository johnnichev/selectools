"""
LiteLLM provider adapter -- instant access to 100+ models.

Delegates completions to the `litellm <https://docs.litellm.ai>`_ library,
which routes ``provider/model`` identifiers (``deepseek/deepseek-chat``,
``groq/llama-3.1-70b``, ``bedrock/anthropic.claude-3-sonnet``, ...) to the
right backend and normalizes every response to the OpenAI wire format.
That lets this adapter reuse the shared ``_OpenAICompatibleBase``
machinery (message formatting, tool mapping, streaming assembly,
BUG-31-safe argument parsing) verbatim.

litellm is an optional dependency::

    pip install selectools[litellm]

Native providers (OpenAI, Anthropic, Gemini, Ollama, Azure OpenAI) remain
the choice for maximum control; ``LiteLLMProvider`` is the long-tail
solution.

Cost tracking: ``UsageStats.cost_usd`` is computed with litellm's own
cost map via ``litellm.cost_per_token`` (cheap, local lookup -- no extra
API call). Models missing from litellm's cost map report ``0.0`` instead
of raising. Prompt-cache token fields are left ``None``; litellm does not
report cache usage uniformly across backends.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from ..env import load_default_env
from ..stability import beta
from ._openai_compat import _OpenAICompatibleBase, _parse_tool_args
from .base import ProviderError

# Keys the base class supplies on every completion call. Allowing them in
# **litellm_kwargs would either be silently overridden (the merge puts
# per-call kwargs last) or, worse, leak into non-streaming paths
# (stream=True would make litellm return a stream wrapper into
# _parse_response). Reject them at construction instead.
_RESERVED_LITELLM_KWARGS = frozenset(
    {"model", "messages", "stream", "tools", "temperature", "max_tokens"}
)


def _import_litellm() -> Any:
    """Lazy-import litellm, raising a helpful error on failure."""
    try:
        import litellm  # noqa: F811
    except ImportError as e:
        raise ImportError(
            "litellm package required for LiteLLMProvider. "
            "Install with: pip install selectools[litellm]"
        ) from e
    return litellm


class _LiteLLMCompletions:
    """Adapts ``litellm.completion``/``acompletion`` to the OpenAI SDK's
    ``client.chat.completions.create(**kwargs)`` surface expected by
    ``_OpenAICompatibleBase``."""

    def __init__(self, fn: Callable[..., Any], defaults: Dict[str, Any]) -> None:
        self._fn = fn
        self._defaults = defaults

    def create(self, **kwargs: Any) -> Any:
        merged = {**self._defaults, **kwargs}
        return self._fn(**merged)


class _LiteLLMChat:
    def __init__(self, fn: Callable[..., Any], defaults: Dict[str, Any]) -> None:
        self.completions = _LiteLLMCompletions(fn, defaults)


class _LiteLLMClientShim:
    def __init__(self, fn: Callable[..., Any], defaults: Dict[str, Any]) -> None:
        self.chat = _LiteLLMChat(fn, defaults)


@beta
class LiteLLMProvider(_OpenAICompatibleBase):
    """Adapter that routes completions through the litellm library.

    Args:
        model: Default litellm model identifier in ``provider/model`` form,
            e.g. ``"deepseek/deepseek-chat"`` or ``"groq/llama-3.1-70b"``.
            Set the same value on ``AgentConfig(model=...)`` -- the agent
            passes its configured model to the provider on every call.
        api_key: Optional explicit API key forwarded to litellm. When
            omitted, litellm reads the provider-specific environment
            variable (``DEEPSEEK_API_KEY``, ``GROQ_API_KEY``, ...).
        api_base: Optional API base URL override (self-hosted gateways,
            proxies).
        **litellm_kwargs: Extra keyword arguments forwarded to every
            ``litellm.completion``/``acompletion`` call, e.g.
            ``drop_params=True`` or ``num_retries=2``. Per-call arguments
            built by the agent loop take precedence over these defaults,
            so keys the base class supplies on every call -- ``model``,
            ``messages``, ``stream``, ``tools``, ``temperature``,
            ``max_tokens`` -- are reserved and raise ``ValueError`` at
            construction. Set temperature/max_tokens on ``AgentConfig``
            instead.

    Example:
        >>> provider = LiteLLMProvider(model="groq/llama-3.1-70b")
        >>> agent = Agent(tools, provider=provider,
        ...               config=AgentConfig(model="groq/llama-3.1-70b"))
    """

    name = "litellm"
    supports_streaming = True
    supports_async = True

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **litellm_kwargs: Any,
    ) -> None:
        load_default_env()
        reserved = _RESERVED_LITELLM_KWARGS.intersection(litellm_kwargs)
        if reserved:
            raise ValueError(
                f"litellm_kwargs key(s) {sorted(reserved)} are reserved: the agent "
                "loop supplies them on every call, so defaults here would be "
                "silently overridden (or, for 'stream', corrupt non-streaming "
                "calls). Use AgentConfig for temperature/max_tokens."
            )
        self._litellm = _import_litellm()

        defaults: Dict[str, Any] = dict(litellm_kwargs)
        if api_key is not None:
            defaults["api_key"] = api_key
        if api_base is not None:
            defaults["api_base"] = api_base

        self._client = _LiteLLMClientShim(self._litellm.completion, defaults)
        self._async_client = _LiteLLMClientShim(self._litellm.acompletion, defaults)
        self.default_model = model

    # -- template method overrides -------------------------------------------

    def _get_token_key(self, model: str) -> str:
        # litellm accepts max_tokens for every backend and translates
        # provider-specific quirks (e.g. max_completion_tokens) itself.
        return "max_tokens"

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        try:
            prompt_cost, completion_cost = self._litellm.cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return float(prompt_cost) + float(completion_cost)
        except Exception:  # noqa: BLE001 -- unknown models must not break completions
            return 0.0

    def _get_provider_name(self) -> str:
        return "litellm"

    def _wrap_error(self, exc: Exception, operation: str) -> ProviderError:
        return ProviderError(f"LiteLLM {operation} failed: {exc}")

    def _parse_tool_call_id(self, tc: Any) -> str:
        return getattr(tc, "id", None) or f"call_{id(tc)}"

    def _parse_tool_call_arguments(self, tc: Any) -> Tuple[Dict[str, Any], Optional[str]]:
        """Parse tool-call arguments, returning ``(params, parse_error)``.

        litellm normalizes to the OpenAI shape (JSON string), but some
        routed backends hand back already-parsed dicts -- accept both,
        mirroring the Ollama override. Anything else (list, None, ...)
        is surfaced as a parse error instead of an uncaught TypeError
        from ``json.loads``.
        """
        arguments = tc.function.arguments
        if isinstance(arguments, dict):
            return arguments, None
        if isinstance(arguments, str):
            return _parse_tool_args(arguments)
        return {}, f"unsupported tool arguments type: {type(arguments).__name__}"

    def _initial_tool_call_id(self, tc_delta: Any) -> Optional[str]:
        # Not every litellm-routed backend supplies an ID on the first delta.
        return getattr(tc_delta, "id", None)


__stability__ = "beta"

__all__ = ["LiteLLMProvider"]
