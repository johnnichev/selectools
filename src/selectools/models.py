"""
Canonical model registry for all LLM providers.

This module serves as the single source of truth for all model definitions,
including pricing, capabilities, and constraints. Use the typed constants
for IDE autocomplete and type safety.

Example:
    >>> from selectools.models import OpenAI, Anthropic
    >>> model = OpenAI.GPT_4O
    >>> print(f"Cost: ${model.prompt_cost}/${model.completion_cost} per 1M tokens")
    >>> print(f"Context: {model.context_window} tokens")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .stability import register_stability, stable


@stable
class ModelType(str, Enum):
    """Enumeration of model types.

    Inherits from ``str`` so that ``ModelType.CHAT == "chat"`` is ``True``,
    preserving backward compatibility with code that compares against string literals.
    """

    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@stable
@dataclass(frozen=True)
class ModelInfo:
    """
    Complete metadata for an LLM model.

    Attributes:
        id: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        provider: Provider name ("openai", "anthropic", "gemini", "ollama")
        type: Model type (chat, embedding, etc.)
        prompt_cost: Cost in USD per 1M input tokens
        completion_cost: Cost in USD per 1M output tokens
        max_tokens: Maximum output tokens per request
        context_window: Maximum context length in tokens
        cached_prompt_cost: Cost in USD per 1M CACHED input tokens (the
            provider-published cache-hit / cached-input rate). None when the
            provider does not publish a cached rate for this model — costing
            then falls back to the full prompt rate. Same verification
            discipline as the other prices: NO rates from memory, only values
            readable on the provider's official pricing page.
    """

    id: str
    provider: str
    type: ModelType
    prompt_cost: float
    completion_cost: float
    max_tokens: int
    context_window: int
    cached_prompt_cost: Optional[float] = None


# =============================================================================
# OpenAI Models
# =============================================================================


@stable
class OpenAI:
    """OpenAI GPT models with pricing and capabilities.

    Verified against https://developers.openai.com/api/docs/pricing,
    https://developers.openai.com/api/docs/models, and
    https://developers.openai.com/api/docs/deprecations (2026-06-12).

    cached_prompt_cost ("Cached input" column) verified against
    https://developers.openai.com/api/docs/pricing (2026-06-12). The page
    only lists current models; legacy/deprecated models no longer appear
    there, so their cached rates are left absent (costing falls back to the
    full prompt rate). gpt-5.5-pro and gpt-5.4-pro are listed WITHOUT a
    cached-input price, so the field stays absent for them by design.
    """

    # ===== GPT-5.5 Series (Latest Flagship) =====
    GPT_5_5 = ModelInfo(
        id="gpt-5.5",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=5.00,
        completion_cost=30.00,
        max_tokens=128000,
        context_window=1050000,
        cached_prompt_cost=0.50,
    )
    GPT_5_5_PRO = ModelInfo(
        id="gpt-5.5-pro",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=30.00,
        completion_cost=180.00,
        max_tokens=128000,
        context_window=1050000,
    )

    # ===== GPT-5.4 Series =====
    GPT_5_4 = ModelInfo(
        id="gpt-5.4",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=2.50,
        completion_cost=15.00,
        max_tokens=128000,
        context_window=1050000,
        cached_prompt_cost=0.25,
    )
    GPT_5_4_PRO = ModelInfo(
        id="gpt-5.4-pro",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=30.00,
        completion_cost=180.00,
        max_tokens=128000,
        context_window=1050000,
    )
    GPT_5_4_MINI = ModelInfo(
        id="gpt-5.4-mini",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.75,
        completion_cost=4.50,
        max_tokens=128000,
        context_window=400000,
        cached_prompt_cost=0.075,
    )
    GPT_5_4_NANO = ModelInfo(
        id="gpt-5.4-nano",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.20,
        completion_cost=1.25,
        max_tokens=128000,
        context_window=400000,
        cached_prompt_cost=0.02,
    )

    # ===== GPT-5.3 Series =====
    # DEPRECATED: gpt-5.3-chat-latest shuts down 2026-08-10 (replacement: gpt-5.5).
    GPT_5_3_CHAT_LATEST = ModelInfo(
        id="gpt-5.3-chat-latest",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.75,
        completion_cost=14.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_3_CODEX = ModelInfo(
        id="gpt-5.3-codex",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.75,
        completion_cost=14.00,
        max_tokens=16384,
        context_window=128000,
        cached_prompt_cost=0.175,
    )

    # ===== GPT-5.2 Series =====
    GPT_5_2 = ModelInfo(
        id="gpt-5.2",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.75,
        completion_cost=14.00,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-5.2-chat-latest shuts down 2026-08-10 (replacement: gpt-5.5).
    GPT_5_2_CHAT_LATEST = ModelInfo(
        id="gpt-5.2-chat-latest",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.75,
        completion_cost=14.00,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-5.2-codex shuts down 2026-07-23 (replacement: gpt-5.5).
    GPT_5_2_CODEX = ModelInfo(
        id="gpt-5.2-codex",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.75,
        completion_cost=14.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_2_PRO = ModelInfo(
        id="gpt-5.2-pro",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=21.00,
        completion_cost=168.00,
        max_tokens=16384,
        context_window=128000,
    )

    # ===== GPT-5.1 Series =====
    GPT_5_1 = ModelInfo(
        id="gpt-5.1",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5 = ModelInfo(
        id="gpt-5",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_MINI = ModelInfo(
        id="gpt-5-mini",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.25,
        completion_cost=2.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_NANO = ModelInfo(
        id="gpt-5-nano",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.05,
        completion_cost=0.40,
        max_tokens=128000,
        context_window=400000,
    )
    # DEPRECATED: gpt-5.1-chat-latest shuts down 2026-07-23 (replacement: gpt-5.5).
    GPT_5_1_CHAT_LATEST = ModelInfo(
        id="gpt-5.1-chat-latest",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-5-chat-latest shuts down 2026-07-23 (replacement: gpt-5.5).
    GPT_5_CHAT_LATEST = ModelInfo(
        id="gpt-5-chat-latest",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-5.1-codex-max shuts down 2026-07-23 (replacement: gpt-5.5).
    GPT_5_1_CODEX_MAX = ModelInfo(
        id="gpt-5.1-codex-max",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-5.1-codex shuts down 2026-07-23 (replacement: gpt-5.5).
    GPT_5_1_CODEX = ModelInfo(
        id="gpt-5.1-codex",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-5-codex shuts down 2026-07-23 (replacement: gpt-5.5).
    GPT_5_CODEX = ModelInfo(
        id="gpt-5-codex",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_PRO = ModelInfo(
        id="gpt-5-pro",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=15.00,
        completion_cost=120.00,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-5.1-codex-mini shuts down 2026-07-23 (replacement: gpt-5.4-mini).
    GPT_5_1_CODEX_MINI = ModelInfo(
        id="gpt-5.1-codex-mini",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.25,
        completion_cost=2.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_SEARCH_API = ModelInfo(
        id="gpt-5-search-api",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    # ===== GPT-4.1 Series =====
    GPT_4_1 = ModelInfo(
        id="gpt-4.1",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=2.00,
        completion_cost=8.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4_1_MINI = ModelInfo(
        id="gpt-4.1-mini",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.40,
        completion_cost=1.60,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-4.1-nano shuts down 2026-10-23 (replacement: gpt-5.4-nano).
    GPT_4_1_NANO = ModelInfo(
        id="gpt-4.1-nano",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.10,
        completion_cost=0.40,
        max_tokens=8192,
        context_window=128000,
    )

    # ===== GPT-4o Series =====
    GPT_4O = ModelInfo(
        id="gpt-4o",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4O_2024_11_20 = ModelInfo(
        id="gpt-4o-2024-11-20",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4O_2024_08_06 = ModelInfo(
        id="gpt-4o-2024-08-06",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-4o-2024-05-13 shuts down 2026-10-23 (replacement: gpt-5.5).
    GPT_4O_2024_05_13 = ModelInfo(
        id="gpt-4o-2024-05-13",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=5.00,
        completion_cost=15.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4O_MINI = ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.15,
        completion_cost=0.60,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4O_MINI_2024_07_18 = ModelInfo(
        id="gpt-4o-mini-2024-07-18",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.15,
        completion_cost=0.60,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-4o-mini-search-preview shuts down 2026-07-23 (replacement: gpt-5.4-mini).
    GPT_4O_MINI_SEARCH_PREVIEW = ModelInfo(
        id="gpt-4o-mini-search-preview",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.15,
        completion_cost=0.60,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: gpt-4o-search-preview shuts down 2026-07-23 (replacement: gpt-5.4-mini).
    GPT_4O_SEARCH_PREVIEW = ModelInfo(
        id="gpt-4o-search-preview",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )

    # ===== GPT Realtime/Audio =====
    # Text-token pricing; audio tokens are billed separately ($32.00/$64.00 per 1M).
    # cached_prompt_cost is the text cached-input rate.
    GPT_REALTIME_2 = ModelInfo(
        id="gpt-realtime-2",
        provider="openai",
        type=ModelType.AUDIO,
        prompt_cost=4.00,
        completion_cost=24.00,
        max_tokens=32000,
        context_window=128000,
        cached_prompt_cost=0.40,
    )
    GPT_REALTIME = ModelInfo(
        id="gpt-realtime",
        provider="openai",
        type=ModelType.AUDIO,
        prompt_cost=4.00,
        completion_cost=16.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_REALTIME_1_5 = ModelInfo(
        id="gpt-realtime-1.5",
        provider="openai",
        type=ModelType.AUDIO,
        prompt_cost=4.00,
        completion_cost=16.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_REALTIME_MINI = ModelInfo(
        id="gpt-realtime-mini",
        provider="openai",
        type=ModelType.AUDIO,
        prompt_cost=0.60,
        completion_cost=2.40,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_AUDIO = ModelInfo(
        id="gpt-audio",
        provider="openai",
        type=ModelType.AUDIO,
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_AUDIO_1_5 = ModelInfo(
        id="gpt-audio-1.5",
        provider="openai",
        type=ModelType.AUDIO,
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_AUDIO_MINI = ModelInfo(
        id="gpt-audio-mini",
        provider="openai",
        type=ModelType.AUDIO,
        prompt_cost=0.60,
        completion_cost=2.40,
        max_tokens=4096,
        context_window=128000,
    )
    # ===== o-series (Reasoning Models) =====
    # DEPRECATED: o1 / o1-2024-12-17 shut down 2026-10-23 (replacement: gpt-5.5).
    O1 = ModelInfo(
        id="o1",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=15.00,
        completion_cost=60.00,
        max_tokens=32768,
        context_window=200000,
    )
    O1_2024_12_17 = ModelInfo(
        id="o1-2024-12-17",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=15.00,
        completion_cost=60.00,
        max_tokens=32768,
        context_window=200000,
    )
    # DEPRECATED: o1-pro shuts down 2026-10-23 (replacement: gpt-5.5-pro).
    O1_PRO = ModelInfo(
        id="o1-pro",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=150.00,
        completion_cost=600.00,
        max_tokens=32768,
        context_window=200000,
    )
    O3_PRO = ModelInfo(
        id="o3-pro",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=20.00,
        completion_cost=80.00,
        max_tokens=32768,
        context_window=200000,
    )
    O3 = ModelInfo(
        id="o3",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=2.00,
        completion_cost=8.00,
        max_tokens=32768,
        context_window=200000,
    )
    # DEPRECATED: o3-deep-research shuts down 2026-07-23 (replacement: gpt-5.5-pro).
    O3_DEEP_RESEARCH = ModelInfo(
        id="o3-deep-research",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=10.00,
        completion_cost=40.00,
        max_tokens=32768,
        context_window=200000,
    )
    # DEPRECATED: o3-mini shuts down 2026-10-23 (replacement: gpt-5.5).
    O3_MINI = ModelInfo(
        id="o3-mini",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.10,
        completion_cost=4.40,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: o4-mini shuts down 2026-10-23 (replacement: gpt-5.4-mini).
    O4_MINI = ModelInfo(
        id="o4-mini",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.10,
        completion_cost=4.40,
        max_tokens=16384,
        context_window=128000,
    )
    # DEPRECATED: o4-mini-deep-research shuts down 2026-07-23 (replacement: gpt-5.5-pro).
    O4_MINI_DEEP_RESEARCH = ModelInfo(
        id="o4-mini-deep-research",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=2.00,
        completion_cost=8.00,
        max_tokens=32768,
        context_window=200000,
    )

    # ===== GPT-4 Turbo (Legacy) =====
    # DEPRECATED: gpt-4-turbo and gpt-4-turbo-2024-04-09 shut down 2026-10-23
    # (replacement: gpt-5.5).
    GPT_4_TURBO = ModelInfo(
        id="gpt-4-turbo",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=10.00,
        completion_cost=30.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4_TURBO_2024_04_09 = ModelInfo(
        id="gpt-4-turbo-2024-04-09",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=10.00,
        completion_cost=30.00,
        max_tokens=4096,
        context_window=128000,
    )

    # ===== GPT-4 Classic (Legacy) =====
    # DEPRECATED: gpt-4 and gpt-4-0613 shut down 2026-10-23 (replacement: gpt-5.5).
    GPT_4 = ModelInfo(
        id="gpt-4",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=30.00,
        completion_cost=60.00,
        max_tokens=8192,
        context_window=8192,
    )
    GPT_4_0613 = ModelInfo(
        id="gpt-4-0613",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=30.00,
        completion_cost=60.00,
        max_tokens=8192,
        context_window=8192,
    )

    # ===== GPT-3.5 Turbo (Legacy) =====
    # DEPRECATED: gpt-3.5-turbo and gpt-3.5-turbo-0125 shut down 2026-10-23
    # (replacement: gpt-5.4-mini).
    GPT_3_5_TURBO = ModelInfo(
        id="gpt-3.5-turbo",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.50,
        completion_cost=1.50,
        max_tokens=4096,
        context_window=16385,
    )
    GPT_3_5_TURBO_0125 = ModelInfo(
        id="gpt-3.5-turbo-0125",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.50,
        completion_cost=1.50,
        max_tokens=4096,
        context_window=16385,
    )
    # DEPRECATED: gpt-3.5-turbo-1106 shuts down 2026-09-28 (replacement: gpt-5.4-mini).
    GPT_3_5_TURBO_1106 = ModelInfo(
        id="gpt-3.5-turbo-1106",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.00,
        completion_cost=2.00,
        max_tokens=4096,
        context_window=16385,
    )
    # DEPRECATED: gpt-3.5-turbo-instruct shuts down 2026-09-28 (replacement: gpt-5.4-mini).
    GPT_3_5_TURBO_INSTRUCT = ModelInfo(
        id="gpt-3.5-turbo-instruct",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=1.50,
        completion_cost=2.00,
        max_tokens=4096,
        context_window=8192,
    )

    # ===== Base Models (Legacy) =====
    # DEPRECATED: davinci-002 and babbage-002 shut down 2026-09-28
    # (replacement: gpt-5.4-mini).
    DAVINCI_002 = ModelInfo(
        id="davinci-002",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=2.00,
        completion_cost=2.00,
        max_tokens=16384,
        context_window=16384,
    )
    BABBAGE_002 = ModelInfo(
        id="babbage-002",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=0.40,
        completion_cost=0.40,
        max_tokens=16384,
        context_window=16384,
    )

    # ===== Aliases =====
    # DEPRECATED: computer-use-preview shuts down 2026-07-23 (replacement: gpt-5.4-mini).
    COMPUTER_USE_PREVIEW = ModelInfo(
        id="computer-use-preview",
        provider="openai",
        type=ModelType.CHAT,
        prompt_cost=3.00,
        completion_cost=12.00,
        max_tokens=8192,
        context_window=128000,
    )

    # ===== Embedding Models =====
    class Embeddings:
        """OpenAI embedding models."""

        TEXT_EMBEDDING_3_SMALL = ModelInfo(
            id="text-embedding-3-small",
            provider="openai",
            type=ModelType.EMBEDDING,
            prompt_cost=0.02,
            completion_cost=0.0,
            max_tokens=8191,
            context_window=8191,
        )
        TEXT_EMBEDDING_3_LARGE = ModelInfo(
            id="text-embedding-3-large",
            provider="openai",
            type=ModelType.EMBEDDING,
            prompt_cost=0.13,
            completion_cost=0.0,
            max_tokens=8191,
            context_window=8191,
        )
        ADA_002 = ModelInfo(
            id="text-embedding-ada-002",
            provider="openai",
            type=ModelType.EMBEDDING,
            prompt_cost=0.10,
            completion_cost=0.0,
            max_tokens=8191,
            context_window=8191,
        )


# =============================================================================
# Anthropic Claude Models
# =============================================================================


@stable
class Anthropic:
    """Anthropic Claude models with pricing and capabilities.

    Verified against https://platform.claude.com/docs/en/about-claude/models/overview.md
    and https://platform.claude.com/docs/en/about-claude/pricing.md (2026-06-12).

    cached_prompt_cost is the "Cache Hits & Refreshes" column of the model
    pricing table at https://platform.claude.com/docs/en/about-claude/pricing
    (verified 2026-06-12; every listed model is exactly 0.1x its base input
    price). Cache WRITE premiums (1.25x 5-min / 2x 1-hour) are handled by the
    multipliers in pricing.py, not by this field.
    """

    # ===== Claude Fable 5 (Latest, most capable) =====
    # 1M context, 128k output. Uses the new tokenizer introduced with Opus 4.7
    # (same text produces roughly 30% more tokens than pre-4.7 models).
    FABLE_5 = ModelInfo(
        id="claude-fable-5",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=10.00,
        completion_cost=50.00,
        max_tokens=128000,
        context_window=1000000,
        cached_prompt_cost=1.00,
    )

    # ===== Claude Opus 4.x Series =====
    OPUS_4_8 = ModelInfo(
        id="claude-opus-4-8",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=5.00,
        completion_cost=25.00,
        max_tokens=128000,
        context_window=1000000,
        cached_prompt_cost=0.50,
    )
    OPUS_4_7 = ModelInfo(
        id="claude-opus-4-7",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=5.00,
        completion_cost=25.00,
        max_tokens=128000,
        context_window=1000000,
        cached_prompt_cost=0.50,
    )
    OPUS_4_6 = ModelInfo(
        id="claude-opus-4-6",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=5.00,
        completion_cost=25.00,
        max_tokens=128000,
        context_window=1000000,
        cached_prompt_cost=0.50,
    )

    # ===== Claude Sonnet 4.6 =====
    SONNET_4_6 = ModelInfo(
        id="claude-sonnet-4-6",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=64000,
        context_window=1000000,
        cached_prompt_cost=0.30,
    )

    # ===== Claude 4.5 Series =====
    OPUS_4_5 = ModelInfo(
        id="claude-opus-4-5",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=5.00,
        completion_cost=25.00,
        max_tokens=64000,
        context_window=200000,
        cached_prompt_cost=0.50,
    )
    SONNET_4_5 = ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=64000,
        context_window=200000,
        cached_prompt_cost=0.30,
    )
    HAIKU_4_5 = ModelInfo(
        id="claude-haiku-4-5",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=1.00,
        completion_cost=5.00,
        max_tokens=64000,
        context_window=200000,
        cached_prompt_cost=0.10,
    )

    # ===== Claude 4.1 Series =====
    # DEPRECATED: retires 2026-08-05. Migrate to claude-opus-4-8.
    OPUS_4_1 = ModelInfo(
        id="claude-opus-4-1",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=15.00,
        completion_cost=75.00,
        max_tokens=32000,
        context_window=200000,
        cached_prompt_cost=1.50,
    )

    # ===== Claude 4 Base Aliases =====
    # DEPRECATED: both retire 2026-06-15. Migrate to claude-opus-4-8 /
    # claude-sonnet-4-6. These aliases resolve to the 20250514 snapshots.
    OPUS_4_0 = ModelInfo(
        id="claude-opus-4-0",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=15.00,
        completion_cost=75.00,
        max_tokens=32000,
        context_window=200000,
        cached_prompt_cost=1.50,
    )
    SONNET_4_0 = ModelInfo(
        id="claude-sonnet-4-0",
        provider="anthropic",
        type=ModelType.CHAT,
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=64000,
        context_window=200000,
        cached_prompt_cost=0.30,
    )

    # ===== Embedding Models (Voyage AI partnership) =====
    class Embeddings:
        """Anthropic/Voyage embedding models."""

        VOYAGE_3 = ModelInfo(
            id="voyage-3",
            provider="anthropic",
            type=ModelType.EMBEDDING,
            prompt_cost=0.06,
            completion_cost=0.0,
            max_tokens=32000,
            context_window=32000,
        )
        VOYAGE_3_LITE = ModelInfo(
            id="voyage-3-lite",
            provider="anthropic",
            type=ModelType.EMBEDDING,
            prompt_cost=0.02,
            completion_cost=0.0,
            max_tokens=32000,
            context_window=32000,
        )


# =============================================================================
# Google Gemini Models
# =============================================================================


@stable
class Gemini:
    """Google Gemini models with pricing and capabilities.

    Verified against https://ai.google.dev/gemini-api/docs/models and
    https://ai.google.dev/gemini-api/docs/pricing (2026-06-12).
    Tiered models record the base (<=200k-token prompt) price; see comments.

    cached_prompt_cost is the per-token "Context caching" price on
    https://ai.google.dev/gemini-api/docs/pricing (verified 2026-06-12).
    Models whose pricing row lists no context-caching price keep the field
    absent. Tiered models record the base (<=200k) caching rate, matching
    the tier recorded for prompt_cost. NOTE: Google also bills cache
    STORAGE per 1M-token-hour (e.g. $1.00 or $4.50/1M/hr); storage is a
    time-based charge that token-count costing cannot capture and is NOT
    included in cost_usd.
    """

    # ===== Gemini 3.5 Series (Latest) =====
    FLASH_3_5 = ModelInfo(
        id="gemini-3.5-flash",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=1.50,
        completion_cost=9.00,
        max_tokens=65536,
        context_window=1048576,
        cached_prompt_cost=0.15,
    )
    LIVE_TRANSLATE_3_5_PREVIEW = ModelInfo(
        id="gemini-3.5-live-translate-preview",
        provider="gemini",
        type=ModelType.AUDIO,
        prompt_cost=3.50,
        completion_cost=21.00,
        max_tokens=65536,
        context_window=131072,
    )

    # ===== Gemini 3.1 Series =====
    # Tiered pricing: $2.00/$12.00 for prompts <=200k tokens,
    # $4.00/$18.00 for prompts >200k tokens. Base tier recorded here
    # (caching: $0.20 <=200k / $0.40 >200k; base tier recorded).
    PRO_3_1 = ModelInfo(
        id="gemini-3.1-pro-preview",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=2.00,
        completion_cost=12.00,
        max_tokens=65536,
        context_window=1048576,
        cached_prompt_cost=0.20,
    )
    FLASH_LITE_3_1 = ModelInfo(
        id="gemini-3.1-flash-lite",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=0.25,
        completion_cost=1.50,
        max_tokens=65536,
        context_window=1048576,
        cached_prompt_cost=0.025,
    )
    # Text output $3.00/1M; image output is billed at $60.00/1M image tokens.
    FLASH_IMAGE_3_1 = ModelInfo(
        id="gemini-3.1-flash-image",
        provider="gemini",
        type=ModelType.MULTIMODAL,
        prompt_cost=0.50,
        completion_cost=3.00,
        max_tokens=32768,
        context_window=131072,
    )
    FLASH_LIVE_3_1_PREVIEW = ModelInfo(
        id="gemini-3.1-flash-live-preview",
        provider="gemini",
        type=ModelType.AUDIO,
        prompt_cost=0.75,
        completion_cost=4.50,
        max_tokens=65536,
        context_window=131072,
    )
    FLASH_TTS_3_1_PREVIEW = ModelInfo(
        id="gemini-3.1-flash-tts-preview",
        provider="gemini",
        type=ModelType.AUDIO,
        prompt_cost=1.00,
        completion_cost=20.00,
        max_tokens=16384,
        context_window=8192,
    )

    # ===== Gemini 3 Series =====
    # Text output $12.00/1M; image output is billed at $120.00/1M image tokens.
    PRO_3_IMAGE = ModelInfo(
        id="gemini-3-pro-image",
        provider="gemini",
        type=ModelType.MULTIMODAL,
        prompt_cost=2.00,
        completion_cost=12.00,
        max_tokens=32768,
        context_window=65536,
    )
    FLASH_3_PREVIEW = ModelInfo(
        id="gemini-3-flash-preview",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=0.50,
        completion_cost=3.00,
        max_tokens=65536,
        context_window=1048576,
        cached_prompt_cost=0.05,
    )

    # ===== Gemini 2.5 Series =====
    # Tiered pricing: $1.25/$10.00 for prompts <=200k tokens,
    # $2.50/$15.00 for prompts >200k tokens. Base tier recorded here
    # (caching: $0.125 <=200k / $0.25 >200k; base tier recorded).
    PRO_2_5 = ModelInfo(
        id="gemini-2.5-pro",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=65536,
        context_window=1048576,
        cached_prompt_cost=0.125,
    )
    FLASH_2_5 = ModelInfo(
        id="gemini-2.5-flash",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=0.30,
        completion_cost=2.50,
        max_tokens=65536,
        context_window=1048576,
        cached_prompt_cost=0.03,
    )
    FLASH_LITE_2_5 = ModelInfo(
        id="gemini-2.5-flash-lite",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=0.10,
        completion_cost=0.40,
        max_tokens=65536,
        context_window=1048576,
        cached_prompt_cost=0.01,
    )
    FLASH_NATIVE_AUDIO_2_5_PREVIEW = ModelInfo(
        id="gemini-2.5-flash-native-audio-preview-12-2025",
        provider="gemini",
        type=ModelType.AUDIO,
        prompt_cost=0.50,
        completion_cost=2.00,
        max_tokens=8192,
        context_window=131072,
    )
    # Image output is billed per image ($0.039/image standard), not per token;
    # completion_cost reflects the legacy per-token figure and is approximate.
    FLASH_IMAGE_2_5 = ModelInfo(
        id="gemini-2.5-flash-image",
        provider="gemini",
        type=ModelType.IMAGE,
        prompt_cost=0.30,
        completion_cost=2.50,
        max_tokens=32768,
        context_window=65536,
    )
    FLASH_TTS_2_5_PREVIEW = ModelInfo(
        id="gemini-2.5-flash-preview-tts",
        provider="gemini",
        type=ModelType.AUDIO,
        prompt_cost=0.50,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=8192,
    )
    PRO_TTS_2_5_PREVIEW = ModelInfo(
        id="gemini-2.5-pro-preview-tts",
        provider="gemini",
        type=ModelType.AUDIO,
        prompt_cost=1.00,
        completion_cost=20.00,
        max_tokens=16384,
        context_window=8192,
    )
    # Tiered pricing like gemini-2.5-pro; base tier recorded here.
    COMPUTER_USE_2_5_PREVIEW = ModelInfo(
        id="gemini-2.5-computer-use-preview-10-2025",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=64000,
        context_window=128000,
    )

    # ===== Gemini Robotics =====
    ROBOTICS_ER_1_6 = ModelInfo(
        id="gemini-robotics-er-1.6-preview",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=1.00,
        completion_cost=5.00,
        max_tokens=65536,
        context_window=131072,
    )
    # Still served (Preview), superseded by 1.6. Pricing carried over
    # from the previous registry revision (not republished by Google).
    ROBOTICS_ER_1_5 = ModelInfo(
        id="gemini-robotics-er-1.5-preview",
        provider="gemini",
        type=ModelType.CHAT,
        prompt_cost=0.30,
        completion_cost=2.50,
        max_tokens=8192,
        context_window=1000000,
    )

    # ===== Embedding Models =====
    class Embeddings:
        """Google Gemini embedding models."""

        EMBEDDING_001 = ModelInfo(
            id="gemini-embedding-001",
            provider="gemini",
            type=ModelType.EMBEDDING,
            prompt_cost=0.15,
            completion_cost=0.0,
            max_tokens=2048,
            context_window=2048,
        )
        EMBEDDING_2 = ModelInfo(
            id="gemini-embedding-2",
            provider="gemini",
            type=ModelType.EMBEDDING,
            prompt_cost=0.20,
            completion_cost=0.0,
            max_tokens=8192,
            context_window=8192,
        )


# =============================================================================
# Ollama Local Models (11 total)
# =============================================================================


@stable
class Ollama:
    """Ollama local models - all free ($0.00)."""

    LLAMA_3_2 = ModelInfo(
        id="llama3.2",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    LLAMA_3_1 = ModelInfo(
        id="llama3.1",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    LLAMA_3 = ModelInfo(
        id="llama3",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    LLAMA_2 = ModelInfo(
        id="llama2",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=4096,
    )
    MISTRAL = ModelInfo(
        id="mistral",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    MIXTRAL = ModelInfo(
        id="mixtral",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=32768,
    )
    CODELLAMA = ModelInfo(
        id="codellama",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=16384,
    )
    PHI = ModelInfo(
        id="phi",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=2048,
        context_window=4096,
    )
    NEURAL_CHAT = ModelInfo(
        id="neural-chat",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    STARLING_LM = ModelInfo(
        id="starling-lm",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    QWEN = ModelInfo(
        id="qwen",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=32768,
    )
    GEMMA = ModelInfo(
        id="gemma",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=8192,
        context_window=8192,
    )
    VICUNA = ModelInfo(
        id="vicuna",
        provider="ollama",
        type=ModelType.CHAT,
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )


# =============================================================================
# Cohere Models (3 embedding models)
# =============================================================================


@stable
class Cohere:
    """Cohere embedding models."""

    # ===== Embedding Models =====
    class Embeddings:
        """Cohere embedding models."""

        EMBED_V3 = ModelInfo(
            id="embed-english-v3.0",
            provider="cohere",
            type=ModelType.EMBEDDING,
            prompt_cost=0.10,
            completion_cost=0.0,
            max_tokens=512,
            context_window=512,
        )
        EMBED_MULTILINGUAL_V3 = ModelInfo(
            id="embed-multilingual-v3.0",
            provider="cohere",
            type=ModelType.EMBEDDING,
            prompt_cost=0.10,
            completion_cost=0.0,
            max_tokens=512,
            context_window=512,
        )
        EMBED_V3_LIGHT = ModelInfo(
            id="embed-english-light-v3.0",
            provider="cohere",
            type=ModelType.EMBEDDING,
            prompt_cost=0.10,
            completion_cost=0.0,
            max_tokens=512,
            context_window=512,
        )


# =============================================================================
# Aggregated Model Lists
# =============================================================================


def _collect_all_models() -> List[ModelInfo]:
    """Collect all model definitions from provider classes."""
    models = []

    for provider_class in [OpenAI, Anthropic, Gemini, Ollama, Cohere]:
        for attr_name in dir(provider_class):
            if attr_name.startswith("_"):
                continue
            attr = getattr(provider_class, attr_name)
            if isinstance(attr, ModelInfo):
                models.append(attr)
            # Check for nested Embeddings class
            elif isinstance(attr, type) and attr_name == "Embeddings":
                for embed_attr_name in dir(attr):
                    if embed_attr_name.startswith("_"):
                        continue
                    embed_attr = getattr(attr, embed_attr_name)
                    if isinstance(embed_attr, ModelInfo):
                        models.append(embed_attr)

    return models


ALL_MODELS: List[ModelInfo] = _collect_all_models()
"""Complete list of all models across all providers (chat + embedding)."""

MODELS_BY_ID: Dict[str, ModelInfo] = {model.id: model for model in ALL_MODELS}

register_stability("ALL_MODELS", "stable")
register_stability("MODELS_BY_ID", "stable")
"""Quick lookup dictionary mapping model ID to ModelInfo."""


__stability__ = "stable"

__all__ = [
    "ModelInfo",
    "ModelType",
    "OpenAI",
    "Anthropic",
    "Gemini",
    "Ollama",
    "Cohere",
    "ALL_MODELS",
    "MODELS_BY_ID",
]
