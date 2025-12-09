"""
Pricing tables for major LLM providers.

Prices in USD per 1M tokens (updated December 2024).
Prices are subject to change - check provider documentation for current rates:
- OpenAI: https://openai.com/api/pricing/
- Anthropic: https://www.anthropic.com/pricing
- Google: https://ai.google.dev/pricing
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Pricing in USD per 1M tokens (Standard tier, text tokens)
PRICING: Dict[str, Dict[str, float]] = {
    # ===== OpenAI GPT-5 Series (Latest Generation) =====
    "gpt-5.1": {"prompt": 1.25, "completion": 10.00},
    "gpt-5": {"prompt": 1.25, "completion": 10.00},
    "gpt-5-mini": {"prompt": 0.25, "completion": 2.00},
    "gpt-5-nano": {"prompt": 0.05, "completion": 0.40},
    "gpt-5.1-chat-latest": {"prompt": 1.25, "completion": 10.00},
    "gpt-5-chat-latest": {"prompt": 1.25, "completion": 10.00},
    "gpt-5.1-codex-max": {"prompt": 1.25, "completion": 10.00},
    "gpt-5.1-codex": {"prompt": 1.25, "completion": 10.00},
    "gpt-5-codex": {"prompt": 1.25, "completion": 10.00},
    "gpt-5-pro": {"prompt": 15.00, "completion": 120.00},
    "gpt-5.1-codex-mini": {"prompt": 0.25, "completion": 2.00},
    "gpt-5-search-api": {"prompt": 1.25, "completion": 10.00},
    "codex-mini-latest": {"prompt": 1.50, "completion": 6.00},
    # ===== OpenAI GPT-4.1 Series =====
    "gpt-4.1": {"prompt": 2.00, "completion": 8.00},
    "gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60},
    "gpt-4.1-nano": {"prompt": 0.10, "completion": 0.40},
    # ===== OpenAI GPT-4o Series =====
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-2024-11-20": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-2024-08-06": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-2024-05-13": {"prompt": 5.00, "completion": 15.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o-mini-2024-07-18": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o-mini-search-preview": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o-search-preview": {"prompt": 2.50, "completion": 10.00},
    # ===== OpenAI GPT-4o Realtime/Audio =====
    "gpt-realtime": {"prompt": 4.00, "completion": 16.00},  # Text tokens
    "gpt-realtime-mini": {"prompt": 0.60, "completion": 2.40},  # Text tokens
    "gpt-4o-realtime-preview": {"prompt": 5.00, "completion": 20.00},  # Text tokens
    "gpt-4o-mini-realtime-preview": {"prompt": 0.60, "completion": 2.40},  # Text tokens
    "gpt-audio": {"prompt": 2.50, "completion": 10.00},  # Text tokens
    "gpt-audio-mini": {"prompt": 0.60, "completion": 2.40},  # Text tokens
    "gpt-4o-audio-preview": {"prompt": 2.50, "completion": 10.00},  # Text tokens
    "gpt-4o-mini-audio-preview": {"prompt": 0.15, "completion": 0.60},  # Text tokens
    # ===== OpenAI o-series (Reasoning Models) =====
    "o1": {"prompt": 15.00, "completion": 60.00},
    "o1-2024-12-17": {"prompt": 15.00, "completion": 60.00},
    "o1-pro": {"prompt": 150.00, "completion": 600.00},
    "o1-mini": {"prompt": 1.10, "completion": 4.40},
    "o3-pro": {"prompt": 20.00, "completion": 80.00},
    "o3": {"prompt": 2.00, "completion": 8.00},
    "o3-deep-research": {"prompt": 10.00, "completion": 40.00},
    "o3-mini": {"prompt": 1.10, "completion": 4.40},
    "o4-mini": {"prompt": 1.10, "completion": 4.40},
    "o4-mini-deep-research": {"prompt": 2.00, "completion": 8.00},
    # ===== OpenAI GPT-4 Turbo (Legacy) =====
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-4-turbo-2024-04-09": {"prompt": 10.00, "completion": 30.00},
    "gpt-4-turbo-preview": {"prompt": 10.00, "completion": 30.00},
    "gpt-4-0125-preview": {"prompt": 10.00, "completion": 30.00},
    "gpt-4-1106-preview": {"prompt": 10.00, "completion": 30.00},
    "gpt-4-1106-vision-preview": {"prompt": 10.00, "completion": 30.00},
    # ===== OpenAI GPT-4 Classic (Legacy) =====
    "gpt-4": {"prompt": 30.00, "completion": 60.00},
    "gpt-4-0613": {"prompt": 30.00, "completion": 60.00},
    "gpt-4-0314": {"prompt": 30.00, "completion": 60.00},
    "gpt-4-32k": {"prompt": 60.00, "completion": 120.00},
    # ===== OpenAI GPT-3.5 Turbo (Legacy) =====
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "gpt-3.5-turbo-0125": {"prompt": 0.50, "completion": 1.50},
    "gpt-3.5-turbo-1106": {"prompt": 1.00, "completion": 2.00},
    "gpt-3.5-turbo-0613": {"prompt": 1.50, "completion": 2.00},
    "gpt-3.5-0301": {"prompt": 1.50, "completion": 2.00},
    "gpt-3.5-turbo-instruct": {"prompt": 1.50, "completion": 2.00},
    "gpt-3.5-turbo-16k": {"prompt": 3.00, "completion": 4.00},
    "gpt-3.5-turbo-16k-0613": {"prompt": 3.00, "completion": 4.00},
    # ===== OpenAI Base Models (Legacy) =====
    "davinci-002": {"prompt": 2.00, "completion": 2.00},
    "babbage-002": {"prompt": 0.40, "completion": 0.40},
    # ===== OpenAI Aliases =====
    "chatgpt-4o-latest": {"prompt": 5.00, "completion": 15.00},
    "computer-use-preview": {"prompt": 3.00, "completion": 12.00},
    # ===== Anthropic Claude 4.5 Series (Latest) =====
    "claude-opus-4-5": {"prompt": 5.00, "completion": 25.00},
    "claude-sonnet-4-5": {"prompt": 3.00, "completion": 15.00},
    "claude-haiku-4-5": {"prompt": 1.00, "completion": 5.00},
    # ===== Anthropic Claude 4.1 Series =====
    "claude-opus-4-11": {"prompt": 15.00, "completion": 75.00},
    # ===== Anthropic Claude 4 Series =====
    "claude-opus-4-01": {"prompt": 15.00, "completion": 75.00},
    "claude-sonnet-4-01": {"prompt": 3.00, "completion": 15.00},
    # ===== Anthropic Claude 3.7 Series =====
    "claude-3-7-sonnet-latest": {"prompt": 3.00, "completion": 15.00},  # Deprecated
    # ===== Anthropic Claude 3.5 Series (Legacy) =====
    "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-sonnet-20240620": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-sonnet-latest": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-haiku-20241022": {"prompt": 0.80, "completion": 4.00},
    "claude-3-5-haiku-latest": {"prompt": 0.80, "completion": 4.00},
    # ===== Anthropic Claude 3 Series (Legacy, Deprecated) =====
    "claude-3-opus-20240229": {"prompt": 15.00, "completion": 75.00},
    "claude-3-opus": {"prompt": 15.00, "completion": 75.00},
    "claude-3-sonnet-20240229": {"prompt": 3.00, "completion": 15.00},
    "claude-3-sonnet": {"prompt": 3.00, "completion": 15.00},
    "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
    # ===== Google Gemini 3 Series (Latest) =====
    "gemini-3-pro-preview": {"prompt": 2.00, "completion": 12.00},  # ≤200k tokens
    "gemini-3-pro-image-preview": {"prompt": 2.00, "completion": 12.00},  # Text tokens
    # ===== Google Gemini 2.5 Series =====
    "gemini-2.5-pro": {"prompt": 1.25, "completion": 10.00},  # ≤200k tokens
    "gemini-2.5-flash": {"prompt": 0.30, "completion": 2.50},  # Text/image/video
    "gemini-2.5-flash-preview-09-2025": {"prompt": 0.30, "completion": 2.50},
    "gemini-2.5-flash-lite": {"prompt": 0.10, "completion": 0.40},
    "gemini-2.5-flash-lite-preview-09-2025": {"prompt": 0.10, "completion": 0.40},
    "gemini-2.5-flash-native-audio-preview-09-2025": {"prompt": 0.50, "completion": 2.00},  # Text
    "gemini-2.5-flash-image": {"prompt": 0.30, "completion": 2.50},  # Text tokens
    "gemini-2.5-flash-preview-tts": {"prompt": 0.50, "completion": 10.00},
    "gemini-2.5-pro-preview-tts": {"prompt": 1.00, "completion": 20.00},
    "gemini-2.5-computer-use-preview-10-2025": {"prompt": 1.25, "completion": 10.00},
    # ===== Google Gemini 2.0 Series =====
    "gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40},  # Text/image/video
    "gemini-2.0-flash-lite": {"prompt": 0.075, "completion": 0.30},
    "gemini-live-2.5-flash-preview": {"prompt": 0.50, "completion": 2.00},  # Text
    "gemini-2.0-flash-live-001": {"prompt": 0.35, "completion": 1.50},  # Text
    # ===== Google Gemini Robotics =====
    "gemini-robotics-er-1.5-preview": {"prompt": 0.30, "completion": 2.50},
    # ===== Google Gemini 1.5 Series (Stable) =====
    "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},
    "gemini-1.5-pro-latest": {"prompt": 1.25, "completion": 5.00},
    "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
    "gemini-1.5-flash-latest": {"prompt": 0.075, "completion": 0.30},
    # ===== Google Gemini 1.0 Series (Legacy) =====
    "gemini-1.0-pro": {"prompt": 0.50, "completion": 1.50},
    "gemini-pro": {"prompt": 0.50, "completion": 1.50},
    # ===== Google Gemma (Open Models - Free on Gemini API) =====
    "gemma-3": {"prompt": 0.00, "completion": 0.00},
    "gemma-3n": {"prompt": 0.00, "completion": 0.00},
    # Ollama (Local Models - Free)
    "llama3.2": {"prompt": 0.00, "completion": 0.00},
    "llama3.1": {"prompt": 0.00, "completion": 0.00},
    "llama3": {"prompt": 0.00, "completion": 0.00},
    "llama2": {"prompt": 0.00, "completion": 0.00},
    "mistral": {"prompt": 0.00, "completion": 0.00},
    "mixtral": {"prompt": 0.00, "completion": 0.00},
    "codellama": {"prompt": 0.00, "completion": 0.00},
    "phi": {"prompt": 0.00, "completion": 0.00},
    "neural-chat": {"prompt": 0.00, "completion": 0.00},
    "starling-lm": {"prompt": 0.00, "completion": 0.00},
    "qwen": {"prompt": 0.00, "completion": 0.00},
    "gemma": {"prompt": 0.00, "completion": 0.00},
    "vicuna": {"prompt": 0.00, "completion": 0.00},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate cost in USD for given token usage.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022").
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of completion/output tokens.

    Returns:
        Estimated cost in USD. Returns 0.0 if model pricing is unknown or for local models.

    Note:
        - Local models (not in PRICING table) are assumed to be free ($0.00)
        - Logs a warning if the model is not found in the pricing table
    """
    if model not in PRICING:
        # Don't log warning for obviously local models (will be caught by pattern)
        # Just return 0.0 since local models are free
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
