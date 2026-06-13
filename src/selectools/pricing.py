"""
Pricing tables for major LLM providers.

This module derives pricing from the canonical model registry (models.py).
Prices in USD per 1M tokens (updated June 2026).

Prices are subject to change - check provider documentation for current rates:
- OpenAI: https://developers.openai.com/api/docs/pricing
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
- Google: https://ai.google.dev/gemini-api/docs/pricing
"""

from __future__ import annotations

import logging
from typing import Dict

from .models import ALL_MODELS, MODELS_BY_ID
from .stability import beta, register_stability, stable

logger = logging.getLogger(__name__)

# Backward-compatible pricing dict (auto-generated from models.py)
PRICING: Dict[str, Dict[str, float]] = {
    model.id: {"prompt": model.prompt_cost, "completion": model.completion_cost}
    for model in ALL_MODELS
}
register_stability("PRICING", "stable")

# Prompt-caching multipliers, applied to the model's prompt (input) rate.
# Anthropic's published rates (https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
# - cache reads bill at 0.1x the base input price
# - cache writes with the 5-minute TTL bill at 1.25x the base input price.
#   selectools' cache_system/cache_tools emit {"type": "ephemeral"} markers,
#   which default to the 5-minute TTL. (1-hour-TTL writes bill at 2x, but
#   selectools never requests that TTL.)
CACHE_READ_COST_MULTIPLIER = 0.1
CACHE_WRITE_COST_MULTIPLIER = 1.25


@stable
def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    """
    Calculate cost in USD for given token usage.

    Uses the canonical model registry (models.py) for pricing information.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022").
        prompt_tokens: Number of prompt/input tokens billed at the full rate.
            For Anthropic responses this is ``usage.input_tokens``, which
            already excludes cached tokens.
        completion_tokens: Number of completion/output tokens.
        cache_read_input_tokens: Tokens served from the provider prompt cache
            (Anthropic prompt caching). Billed at 0.1x the prompt rate.
        cache_creation_input_tokens: Tokens written to the provider prompt
            cache (Anthropic prompt caching, 5-minute TTL — what selectools'
            ``cache_system``/``cache_tools`` produce). Billed at 1.25x the
            prompt rate.

    Returns:
        Estimated cost in USD. Returns 0.0 if model pricing is unknown or for local models.

    Note:
        - Local models (not in pricing table) are assumed to be free ($0.00)
        - Logs a warning if the model is not found in the pricing table
        - Cache token counts default to 0, so existing callers are unaffected
    """
    if model not in MODELS_BY_ID:
        logger.warning(
            f"⚠️  Unknown model '{model}' - cannot calculate cost. "
            f"Returning $0.00. Add model to selectools/models.py if known."
        )
        return 0.0

    model_info = MODELS_BY_ID[model]
    prompt_cost = (prompt_tokens / 1_000_000) * model_info.prompt_cost
    completion_cost = (completion_tokens / 1_000_000) * model_info.completion_cost
    cache_read_cost = (
        (cache_read_input_tokens / 1_000_000) * model_info.prompt_cost * CACHE_READ_COST_MULTIPLIER
    )
    cache_write_cost = (
        (cache_creation_input_tokens / 1_000_000)
        * model_info.prompt_cost
        * CACHE_WRITE_COST_MULTIPLIER
    )

    return prompt_cost + completion_cost + cache_read_cost + cache_write_cost


@beta
def calculate_cost_with_cached_input(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """
    Calculate cost in USD when part of the input was served from a prompt cache.

    Generalizes cache-rate pricing for providers whose usage objects report a
    cached-token count that is INCLUDED in the total input count (OpenAI
    ``usage.prompt_tokens_details.cached_tokens``, Gemini
    ``usage_metadata.cached_content_token_count``). The cached portion is
    billed at the model's registry ``cached_prompt_cost`` when published,
    falling back to the full prompt rate otherwise (which reproduces the
    legacy two-term formula exactly).

    Not for Anthropic responses: Anthropic's ``input_tokens`` EXCLUDES cache
    tokens and cache writes carry a premium — use :func:`calculate_cost` with
    its ``cache_read_input_tokens``/``cache_creation_input_tokens`` params.

    Args:
        model: Model name (e.g., "gpt-5.5", "gemini-2.5-flash").
        input_tokens: Total prompt/input tokens, INCLUDING any cached tokens.
        output_tokens: Number of completion/output tokens.
        cached_input_tokens: Portion of ``input_tokens`` served from the
            provider prompt cache. Clamped to ``[0, input_tokens]``.

    Returns:
        Estimated cost in USD. Returns 0.0 if model pricing is unknown or for local models.

    Note:
        Gemini additionally bills cache STORAGE per 1M-token-hour; that
        time-based charge cannot be derived from token counts and is not
        included here.
    """
    if model not in MODELS_BY_ID:
        logger.warning(
            f"⚠️  Unknown model '{model}' - cannot calculate cost. "
            f"Returning $0.00. Add model to selectools/models.py if known."
        )
        return 0.0

    model_info = MODELS_BY_ID[model]
    cached = min(max(cached_input_tokens, 0), max(input_tokens, 0))
    uncached = max(input_tokens - cached, 0)
    cached_rate = (
        model_info.cached_prompt_cost
        if model_info.cached_prompt_cost is not None
        else model_info.prompt_cost
    )

    return (
        (uncached / 1_000_000) * model_info.prompt_cost
        + (cached / 1_000_000) * cached_rate
        + (output_tokens / 1_000_000) * model_info.completion_cost
    )


@stable
def calculate_embedding_cost(model: str, tokens: int) -> float:
    """
    Calculate cost in USD for embedding token usage.

    Uses the canonical model registry (models.py) for pricing information.

    Args:
        model: Embedding model name (e.g., "text-embedding-3-small").
        tokens: Number of tokens embedded.

    Returns:
        Estimated cost in USD. Returns 0.0 if model pricing is unknown or for free models.

    Note:
        - Embedding models only have input/prompt cost (no completion cost)
        - Free models (like Gemini embeddings) return $0.00
        - Logs a warning if the model is not found in the pricing table
    """
    if model not in MODELS_BY_ID:
        logger.warning(
            f"⚠️  Unknown embedding model '{model}' - cannot calculate cost. "
            f"Returning $0.00. Add model to selectools/models.py if known."
        )
        return 0.0

    model_info = MODELS_BY_ID[model]
    embedding_cost = (tokens / 1_000_000) * model_info.prompt_cost

    return embedding_cost


@stable
def get_model_pricing(model: str) -> Dict[str, float] | None:
    """
    Get pricing information for a specific model (backward compatible).

    Args:
        model: Model name.

    Returns:
        Dictionary with 'prompt' and 'completion' prices per 1M tokens,
        or None if model is not found.

    Note:
        For full model information including context window and capabilities,
        use models.MODELS_BY_ID[model_id] instead.
    """
    return PRICING.get(model)


__stability__ = "stable"

__all__ = [
    "PRICING",
    "calculate_cost",
    "calculate_cost_with_cached_input",
    "calculate_embedding_cost",
    "get_model_pricing",
]
