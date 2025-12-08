"""
Pricing tables for major LLM providers.

Prices in USD per 1M tokens (as of December 2024).
Prices are subject to change - check provider documentation for current rates.
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Pricing in USD per 1M tokens - will expand pretty soon
PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-4-turbo-preview": {"prompt": 10.00, "completion": 30.00},
    "gpt-4": {"prompt": 30.00, "completion": 60.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "gpt-3.5-turbo-0125": {"prompt": 0.50, "completion": 1.50},
    # Anthropic Claude
    "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-sonnet-20240620": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-haiku-20241022": {"prompt": 0.80, "completion": 4.00},
    "claude-3-opus-20240229": {"prompt": 15.00, "completion": 75.00},
    "claude-3-sonnet-20240229": {"prompt": 3.00, "completion": 15.00},
    "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    # Google Gemini
    "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},
    "gemini-1.5-pro-latest": {"prompt": 1.25, "completion": 5.00},
    "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
    "gemini-1.5-flash-latest": {"prompt": 0.075, "completion": 0.30},
    "gemini-1.0-pro": {"prompt": 0.50, "completion": 1.50},
    "gemini-pro": {"prompt": 0.50, "completion": 1.50},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate cost in USD for given token usage.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022").
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of completion/output tokens.

    Returns:
        Estimated cost in USD. Returns 0.0 if model pricing is unknown.

    Note:
        Logs a warning if the model is not found in the pricing table.
    """
    if model not in PRICING:
        logger.warning(
            f"⚠️  Unknown model '{model}' - cannot calculate cost. "
            f"Returning $0.00. Add pricing to selectools/pricing.py if known."
        )
        return 0.0

    pricing = PRICING[model]
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]

    return prompt_cost + completion_cost


def get_model_pricing(model: str) -> Dict[str, float] | None:
    """
    Get pricing information for a specific model.

    Args:
        model: Model name.

    Returns:
        Dictionary with 'prompt' and 'completion' prices per 1M tokens,
        or None if model is not found.
    """
    return PRICING.get(model)


__all__ = ["PRICING", "calculate_cost", "get_model_pricing"]
