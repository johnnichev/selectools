"""
Pricing tables for major LLM providers.

This module derives pricing from the canonical model registry (models.py).
Prices in USD per 1M tokens (updated December 2024).

Prices are subject to change - check provider documentation for current rates:
- OpenAI: https://openai.com/api/pricing/
- Anthropic: https://www.anthropic.com/pricing
- Google: https://ai.google.dev/pricing
"""

from __future__ import annotations

import logging
from typing import Dict

from .models import ALL_MODELS, MODELS_BY_ID

logger = logging.getLogger(__name__)

# Backward-compatible pricing dict (auto-generated from models.py)
PRICING: Dict[str, Dict[str, float]] = {
    model.id: {"prompt": model.prompt_cost, "completion": model.completion_cost}
    for model in ALL_MODELS
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate cost in USD for given token usage.

    Uses the canonical model registry (models.py) for pricing information.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022").
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of completion/output tokens.

    Returns:
        Estimated cost in USD. Returns 0.0 if model pricing is unknown or for local models.

    Note:
        - Local models (not in pricing table) are assumed to be free ($0.00)
        - Logs a warning if the model is not found in the pricing table
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

    return prompt_cost + completion_cost


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


__all__ = ["PRICING", "calculate_cost", "get_model_pricing"]
