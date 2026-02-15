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
from typing import Dict, List, Literal

ModelType = Literal["chat", "embedding", "image", "audio", "multimodal"]


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
    """

    id: str
    provider: str
    type: ModelType
    prompt_cost: float
    completion_cost: float
    max_tokens: int
    context_window: int


# =============================================================================
# OpenAI Models (65 total)
# =============================================================================


class OpenAI:
    """OpenAI GPT models with pricing and capabilities."""

    # ===== GPT-5.2 Series (Latest) =====
    GPT_5_2 = ModelInfo(
        id="gpt-5.2",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_2_CHAT_LATEST = ModelInfo(
        id="gpt-5.2-chat-latest",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_2_CODEX = ModelInfo(
        id="gpt-5.2-codex",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_2_PRO = ModelInfo(
        id="gpt-5.2-pro",
        provider="openai",
        type="chat",
        prompt_cost=15.00,
        completion_cost=120.00,
        max_tokens=16384,
        context_window=128000,
    )

    # ===== GPT-5.1 Series =====
    GPT_5_1 = ModelInfo(
        id="gpt-5.1",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5 = ModelInfo(
        id="gpt-5",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_MINI = ModelInfo(
        id="gpt-5-mini",
        provider="openai",
        type="chat",
        prompt_cost=0.25,
        completion_cost=2.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_NANO = ModelInfo(
        id="gpt-5-nano",
        provider="openai",
        type="chat",
        prompt_cost=0.05,
        completion_cost=0.40,
        max_tokens=8192,
        context_window=128000,
    )
    GPT_5_1_CHAT_LATEST = ModelInfo(
        id="gpt-5.1-chat-latest",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_CHAT_LATEST = ModelInfo(
        id="gpt-5-chat-latest",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_1_CODEX_MAX = ModelInfo(
        id="gpt-5.1-codex-max",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_1_CODEX = ModelInfo(
        id="gpt-5.1-codex",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_CODEX = ModelInfo(
        id="gpt-5-codex",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_PRO = ModelInfo(
        id="gpt-5-pro",
        provider="openai",
        type="chat",
        prompt_cost=15.00,
        completion_cost=120.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_1_CODEX_MINI = ModelInfo(
        id="gpt-5.1-codex-mini",
        provider="openai",
        type="chat",
        prompt_cost=0.25,
        completion_cost=2.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_5_SEARCH_API = ModelInfo(
        id="gpt-5-search-api",
        provider="openai",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    CODEX_MINI_LATEST = ModelInfo(
        id="codex-mini-latest",
        provider="openai",
        type="chat",
        prompt_cost=1.50,
        completion_cost=6.00,
        max_tokens=16384,
        context_window=128000,
    )

    # ===== GPT-4.1 Series =====
    GPT_4_1 = ModelInfo(
        id="gpt-4.1",
        provider="openai",
        type="chat",
        prompt_cost=2.00,
        completion_cost=8.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4_1_MINI = ModelInfo(
        id="gpt-4.1-mini",
        provider="openai",
        type="chat",
        prompt_cost=0.40,
        completion_cost=1.60,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4_1_NANO = ModelInfo(
        id="gpt-4.1-nano",
        provider="openai",
        type="chat",
        prompt_cost=0.10,
        completion_cost=0.40,
        max_tokens=8192,
        context_window=128000,
    )

    # ===== GPT-4o Series =====
    GPT_4O = ModelInfo(
        id="gpt-4o",
        provider="openai",
        type="chat",
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4O_2024_11_20 = ModelInfo(
        id="gpt-4o-2024-11-20",
        provider="openai",
        type="chat",
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4O_2024_08_06 = ModelInfo(
        id="gpt-4o-2024-08-06",
        provider="openai",
        type="chat",
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4O_2024_05_13 = ModelInfo(
        id="gpt-4o-2024-05-13",
        provider="openai",
        type="chat",
        prompt_cost=5.00,
        completion_cost=15.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4O_MINI = ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        type="chat",
        prompt_cost=0.15,
        completion_cost=0.60,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4O_MINI_2024_07_18 = ModelInfo(
        id="gpt-4o-mini-2024-07-18",
        provider="openai",
        type="chat",
        prompt_cost=0.15,
        completion_cost=0.60,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4O_MINI_SEARCH_PREVIEW = ModelInfo(
        id="gpt-4o-mini-search-preview",
        provider="openai",
        type="chat",
        prompt_cost=0.15,
        completion_cost=0.60,
        max_tokens=16384,
        context_window=128000,
    )
    GPT_4O_SEARCH_PREVIEW = ModelInfo(
        id="gpt-4o-search-preview",
        provider="openai",
        type="chat",
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=16384,
        context_window=128000,
    )

    # ===== GPT-4o Realtime/Audio =====
    GPT_REALTIME = ModelInfo(
        id="gpt-realtime",
        provider="openai",
        type="audio",
        prompt_cost=4.00,
        completion_cost=16.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_REALTIME_MINI = ModelInfo(
        id="gpt-realtime-mini",
        provider="openai",
        type="audio",
        prompt_cost=0.60,
        completion_cost=2.40,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4O_REALTIME_PREVIEW = ModelInfo(
        id="gpt-4o-realtime-preview",
        provider="openai",
        type="audio",
        prompt_cost=5.00,
        completion_cost=20.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4O_MINI_REALTIME_PREVIEW = ModelInfo(
        id="gpt-4o-mini-realtime-preview",
        provider="openai",
        type="audio",
        prompt_cost=0.60,
        completion_cost=2.40,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_AUDIO = ModelInfo(
        id="gpt-audio",
        provider="openai",
        type="audio",
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_AUDIO_MINI = ModelInfo(
        id="gpt-audio-mini",
        provider="openai",
        type="audio",
        prompt_cost=0.60,
        completion_cost=2.40,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4O_AUDIO_PREVIEW = ModelInfo(
        id="gpt-4o-audio-preview",
        provider="openai",
        type="audio",
        prompt_cost=2.50,
        completion_cost=10.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4O_MINI_AUDIO_PREVIEW = ModelInfo(
        id="gpt-4o-mini-audio-preview",
        provider="openai",
        type="audio",
        prompt_cost=0.15,
        completion_cost=0.60,
        max_tokens=16384,
        context_window=128000,
    )

    # ===== o-series (Reasoning Models) =====
    O1 = ModelInfo(
        id="o1",
        provider="openai",
        type="chat",
        prompt_cost=15.00,
        completion_cost=60.00,
        max_tokens=32768,
        context_window=200000,
    )
    O1_2024_12_17 = ModelInfo(
        id="o1-2024-12-17",
        provider="openai",
        type="chat",
        prompt_cost=15.00,
        completion_cost=60.00,
        max_tokens=32768,
        context_window=200000,
    )
    O1_PRO = ModelInfo(
        id="o1-pro",
        provider="openai",
        type="chat",
        prompt_cost=150.00,
        completion_cost=600.00,
        max_tokens=32768,
        context_window=200000,
    )
    O1_MINI = ModelInfo(
        id="o1-mini",
        provider="openai",
        type="chat",
        prompt_cost=1.10,
        completion_cost=4.40,
        max_tokens=16384,
        context_window=128000,
    )
    O3_PRO = ModelInfo(
        id="o3-pro",
        provider="openai",
        type="chat",
        prompt_cost=20.00,
        completion_cost=80.00,
        max_tokens=32768,
        context_window=200000,
    )
    O3 = ModelInfo(
        id="o3",
        provider="openai",
        type="chat",
        prompt_cost=2.00,
        completion_cost=8.00,
        max_tokens=32768,
        context_window=200000,
    )
    O3_DEEP_RESEARCH = ModelInfo(
        id="o3-deep-research",
        provider="openai",
        type="chat",
        prompt_cost=10.00,
        completion_cost=40.00,
        max_tokens=32768,
        context_window=200000,
    )
    O3_MINI = ModelInfo(
        id="o3-mini",
        provider="openai",
        type="chat",
        prompt_cost=1.10,
        completion_cost=4.40,
        max_tokens=16384,
        context_window=128000,
    )
    O4_MINI = ModelInfo(
        id="o4-mini",
        provider="openai",
        type="chat",
        prompt_cost=1.10,
        completion_cost=4.40,
        max_tokens=16384,
        context_window=128000,
    )
    O4_MINI_DEEP_RESEARCH = ModelInfo(
        id="o4-mini-deep-research",
        provider="openai",
        type="chat",
        prompt_cost=2.00,
        completion_cost=8.00,
        max_tokens=32768,
        context_window=200000,
    )

    # ===== GPT-4 Turbo (Legacy) =====
    GPT_4_TURBO = ModelInfo(
        id="gpt-4-turbo",
        provider="openai",
        type="chat",
        prompt_cost=10.00,
        completion_cost=30.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4_TURBO_2024_04_09 = ModelInfo(
        id="gpt-4-turbo-2024-04-09",
        provider="openai",
        type="chat",
        prompt_cost=10.00,
        completion_cost=30.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4_TURBO_PREVIEW = ModelInfo(
        id="gpt-4-turbo-preview",
        provider="openai",
        type="chat",
        prompt_cost=10.00,
        completion_cost=30.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4_0125_PREVIEW = ModelInfo(
        id="gpt-4-0125-preview",
        provider="openai",
        type="chat",
        prompt_cost=10.00,
        completion_cost=30.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4_1106_PREVIEW = ModelInfo(
        id="gpt-4-1106-preview",
        provider="openai",
        type="chat",
        prompt_cost=10.00,
        completion_cost=30.00,
        max_tokens=4096,
        context_window=128000,
    )
    GPT_4_1106_VISION_PREVIEW = ModelInfo(
        id="gpt-4-1106-vision-preview",
        provider="openai",
        type="multimodal",
        prompt_cost=10.00,
        completion_cost=30.00,
        max_tokens=4096,
        context_window=128000,
    )

    # ===== GPT-4 Classic (Legacy) =====
    GPT_4 = ModelInfo(
        id="gpt-4",
        provider="openai",
        type="chat",
        prompt_cost=30.00,
        completion_cost=60.00,
        max_tokens=8192,
        context_window=8192,
    )
    GPT_4_0613 = ModelInfo(
        id="gpt-4-0613",
        provider="openai",
        type="chat",
        prompt_cost=30.00,
        completion_cost=60.00,
        max_tokens=8192,
        context_window=8192,
    )
    GPT_4_0314 = ModelInfo(
        id="gpt-4-0314",
        provider="openai",
        type="chat",
        prompt_cost=30.00,
        completion_cost=60.00,
        max_tokens=8192,
        context_window=8192,
    )
    GPT_4_32K = ModelInfo(
        id="gpt-4-32k",
        provider="openai",
        type="chat",
        prompt_cost=60.00,
        completion_cost=120.00,
        max_tokens=32768,
        context_window=32768,
    )

    # ===== GPT-3.5 Turbo (Legacy) =====
    GPT_3_5_TURBO = ModelInfo(
        id="gpt-3.5-turbo",
        provider="openai",
        type="chat",
        prompt_cost=0.50,
        completion_cost=1.50,
        max_tokens=4096,
        context_window=16385,
    )
    GPT_3_5_TURBO_0125 = ModelInfo(
        id="gpt-3.5-turbo-0125",
        provider="openai",
        type="chat",
        prompt_cost=0.50,
        completion_cost=1.50,
        max_tokens=4096,
        context_window=16385,
    )
    GPT_3_5_TURBO_1106 = ModelInfo(
        id="gpt-3.5-turbo-1106",
        provider="openai",
        type="chat",
        prompt_cost=1.00,
        completion_cost=2.00,
        max_tokens=4096,
        context_window=16385,
    )
    GPT_3_5_TURBO_0613 = ModelInfo(
        id="gpt-3.5-turbo-0613",
        provider="openai",
        type="chat",
        prompt_cost=1.50,
        completion_cost=2.00,
        max_tokens=4096,
        context_window=4096,
    )
    GPT_3_5_0301 = ModelInfo(
        id="gpt-3.5-0301",
        provider="openai",
        type="chat",
        prompt_cost=1.50,
        completion_cost=2.00,
        max_tokens=4096,
        context_window=4096,
    )
    GPT_3_5_TURBO_INSTRUCT = ModelInfo(
        id="gpt-3.5-turbo-instruct",
        provider="openai",
        type="chat",
        prompt_cost=1.50,
        completion_cost=2.00,
        max_tokens=4096,
        context_window=8192,
    )
    GPT_3_5_TURBO_16K = ModelInfo(
        id="gpt-3.5-turbo-16k",
        provider="openai",
        type="chat",
        prompt_cost=3.00,
        completion_cost=4.00,
        max_tokens=16384,
        context_window=16385,
    )
    GPT_3_5_TURBO_16K_0613 = ModelInfo(
        id="gpt-3.5-turbo-16k-0613",
        provider="openai",
        type="chat",
        prompt_cost=3.00,
        completion_cost=4.00,
        max_tokens=16384,
        context_window=16385,
    )

    # ===== Base Models (Legacy) =====
    DAVINCI_002 = ModelInfo(
        id="davinci-002",
        provider="openai",
        type="chat",
        prompt_cost=2.00,
        completion_cost=2.00,
        max_tokens=16384,
        context_window=16384,
    )
    BABBAGE_002 = ModelInfo(
        id="babbage-002",
        provider="openai",
        type="chat",
        prompt_cost=0.40,
        completion_cost=0.40,
        max_tokens=16384,
        context_window=16384,
    )

    # ===== Aliases =====
    CHATGPT_4O_LATEST = ModelInfo(
        id="chatgpt-4o-latest",
        provider="openai",
        type="chat",
        prompt_cost=5.00,
        completion_cost=15.00,
        max_tokens=16384,
        context_window=128000,
    )
    COMPUTER_USE_PREVIEW = ModelInfo(
        id="computer-use-preview",
        provider="openai",
        type="chat",
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
            type="embedding",
            prompt_cost=0.02,
            completion_cost=0.0,
            max_tokens=8191,
            context_window=8191,
        )
        TEXT_EMBEDDING_3_LARGE = ModelInfo(
            id="text-embedding-3-large",
            provider="openai",
            type="embedding",
            prompt_cost=0.13,
            completion_cost=0.0,
            max_tokens=8191,
            context_window=8191,
        )
        ADA_002 = ModelInfo(
            id="text-embedding-ada-002",
            provider="openai",
            type="embedding",
            prompt_cost=0.10,
            completion_cost=0.0,
            max_tokens=8191,
            context_window=8191,
        )


# =============================================================================
# Anthropic Claude Models (18 total)
# =============================================================================


class Anthropic:
    """Anthropic Claude models with pricing and capabilities."""

    # ===== Claude 4.6 Series (Latest) =====
    OPUS_4_6 = ModelInfo(
        id="claude-opus-4-6",
        provider="anthropic",
        type="chat",
        prompt_cost=5.00,
        completion_cost=25.00,
        max_tokens=8192,
        context_window=200000,
    )

    # ===== Claude 4.5 Series =====
    OPUS_4_5 = ModelInfo(
        id="claude-opus-4-5",
        provider="anthropic",
        type="chat",
        prompt_cost=5.00,
        completion_cost=25.00,
        max_tokens=8192,
        context_window=200000,
    )
    SONNET_4_5 = ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        type="chat",
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=8192,
        context_window=200000,
    )
    HAIKU_4_5 = ModelInfo(
        id="claude-haiku-4-5",
        provider="anthropic",
        type="chat",
        prompt_cost=1.00,
        completion_cost=5.00,
        max_tokens=8192,
        context_window=200000,
    )

    # ===== Claude 4.1 Series =====
    OPUS_4_11 = ModelInfo(
        id="claude-opus-4-11",
        provider="anthropic",
        type="chat",
        prompt_cost=15.00,
        completion_cost=75.00,
        max_tokens=8192,
        context_window=200000,
    )

    # ===== Claude 4 Series =====
    OPUS_4_01 = ModelInfo(
        id="claude-opus-4-01",
        provider="anthropic",
        type="chat",
        prompt_cost=15.00,
        completion_cost=75.00,
        max_tokens=8192,
        context_window=200000,
    )
    SONNET_4_01 = ModelInfo(
        id="claude-sonnet-4-01",
        provider="anthropic",
        type="chat",
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=8192,
        context_window=200000,
    )

    # ===== Claude 3.7 Series (Deprecated) =====
    SONNET_3_7_LATEST = ModelInfo(
        id="claude-3-7-sonnet-latest",
        provider="anthropic",
        type="chat",
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=8192,
        context_window=200000,
    )

    # ===== Claude 3.5 Series =====
    SONNET_3_5_20241022 = ModelInfo(
        id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        type="chat",
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=8192,
        context_window=200000,
    )
    SONNET_3_5_20240620 = ModelInfo(
        id="claude-3-5-sonnet-20240620",
        provider="anthropic",
        type="chat",
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=8192,
        context_window=200000,
    )
    SONNET_3_5_LATEST = ModelInfo(
        id="claude-3-5-sonnet-latest",
        provider="anthropic",
        type="chat",
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=8192,
        context_window=200000,
    )
    HAIKU_3_5_20241022 = ModelInfo(
        id="claude-3-5-haiku-20241022",
        provider="anthropic",
        type="chat",
        prompt_cost=0.80,
        completion_cost=4.00,
        max_tokens=8192,
        context_window=200000,
    )
    HAIKU_3_5_LATEST = ModelInfo(
        id="claude-3-5-haiku-latest",
        provider="anthropic",
        type="chat",
        prompt_cost=0.80,
        completion_cost=4.00,
        max_tokens=8192,
        context_window=200000,
    )

    # ===== Claude 3 Series (Legacy) =====
    OPUS_3_20240229 = ModelInfo(
        id="claude-3-opus-20240229",
        provider="anthropic",
        type="chat",
        prompt_cost=15.00,
        completion_cost=75.00,
        max_tokens=4096,
        context_window=200000,
    )
    OPUS_3 = ModelInfo(
        id="claude-3-opus",
        provider="anthropic",
        type="chat",
        prompt_cost=15.00,
        completion_cost=75.00,
        max_tokens=4096,
        context_window=200000,
    )
    SONNET_3_20240229 = ModelInfo(
        id="claude-3-sonnet-20240229",
        provider="anthropic",
        type="chat",
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=4096,
        context_window=200000,
    )
    SONNET_3 = ModelInfo(
        id="claude-3-sonnet",
        provider="anthropic",
        type="chat",
        prompt_cost=3.00,
        completion_cost=15.00,
        max_tokens=4096,
        context_window=200000,
    )
    HAIKU_3_20240307 = ModelInfo(
        id="claude-3-haiku-20240307",
        provider="anthropic",
        type="chat",
        prompt_cost=0.25,
        completion_cost=1.25,
        max_tokens=4096,
        context_window=200000,
    )
    HAIKU_3 = ModelInfo(
        id="claude-3-haiku",
        provider="anthropic",
        type="chat",
        prompt_cost=0.25,
        completion_cost=1.25,
        max_tokens=4096,
        context_window=200000,
    )

    # ===== Embedding Models (Voyage AI partnership) =====
    class Embeddings:
        """Anthropic/Voyage embedding models."""

        VOYAGE_3 = ModelInfo(
            id="voyage-3",
            provider="anthropic",
            type="embedding",
            prompt_cost=0.06,
            completion_cost=0.0,
            max_tokens=32000,
            context_window=32000,
        )
        VOYAGE_3_LITE = ModelInfo(
            id="voyage-3-lite",
            provider="anthropic",
            type="embedding",
            prompt_cost=0.02,
            completion_cost=0.0,
            max_tokens=32000,
            context_window=32000,
        )


# =============================================================================
# Google Gemini Models (26 total)
# =============================================================================


class Gemini:
    """Google Gemini models with pricing and capabilities."""

    # ===== Gemini 3 Series (Latest) =====
    PRO_3 = ModelInfo(
        id="gemini-3-pro-preview",
        provider="gemini",
        type="chat",
        prompt_cost=2.00,
        completion_cost=12.00,
        max_tokens=8192,
        context_window=2000000,
    )
    PRO_3_IMAGE = ModelInfo(
        id="gemini-3-pro-image-preview",
        provider="gemini",
        type="multimodal",
        prompt_cost=2.00,
        completion_cost=12.00,
        max_tokens=8192,
        context_window=2000000,
    )

    # ===== Gemini 2.5 Series =====
    PRO_2_5 = ModelInfo(
        id="gemini-2.5-pro",
        provider="gemini",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=8192,
        context_window=2000000,
    )
    FLASH_2_5 = ModelInfo(
        id="gemini-2.5-flash",
        provider="gemini",
        type="chat",
        prompt_cost=0.30,
        completion_cost=2.50,
        max_tokens=8192,
        context_window=1000000,
    )
    FLASH_2_5_PREVIEW_09_2025 = ModelInfo(
        id="gemini-2.5-flash-preview-09-2025",
        provider="gemini",
        type="chat",
        prompt_cost=0.30,
        completion_cost=2.50,
        max_tokens=8192,
        context_window=1000000,
    )
    FLASH_LITE_2_5 = ModelInfo(
        id="gemini-2.5-flash-lite",
        provider="gemini",
        type="chat",
        prompt_cost=0.10,
        completion_cost=0.40,
        max_tokens=8192,
        context_window=1000000,
    )
    FLASH_LITE_2_5_PREVIEW_09_2025 = ModelInfo(
        id="gemini-2.5-flash-lite-preview-09-2025",
        provider="gemini",
        type="chat",
        prompt_cost=0.10,
        completion_cost=0.40,
        max_tokens=8192,
        context_window=1000000,
    )
    FLASH_NATIVE_AUDIO_2_5_PREVIEW = ModelInfo(
        id="gemini-2.5-flash-native-audio-preview-09-2025",
        provider="gemini",
        type="audio",
        prompt_cost=0.50,
        completion_cost=2.00,
        max_tokens=8192,
        context_window=1000000,
    )
    FLASH_IMAGE_2_5 = ModelInfo(
        id="gemini-2.5-flash-image",
        provider="gemini",
        type="image",
        prompt_cost=0.30,
        completion_cost=2.50,
        max_tokens=8192,
        context_window=1000000,
    )
    FLASH_TTS_2_5_PREVIEW = ModelInfo(
        id="gemini-2.5-flash-preview-tts",
        provider="gemini",
        type="audio",
        prompt_cost=0.50,
        completion_cost=10.00,
        max_tokens=8192,
        context_window=1000000,
    )
    PRO_TTS_2_5_PREVIEW = ModelInfo(
        id="gemini-2.5-pro-preview-tts",
        provider="gemini",
        type="audio",
        prompt_cost=1.00,
        completion_cost=20.00,
        max_tokens=8192,
        context_window=2000000,
    )
    COMPUTER_USE_2_5_PREVIEW = ModelInfo(
        id="gemini-2.5-computer-use-preview-10-2025",
        provider="gemini",
        type="chat",
        prompt_cost=1.25,
        completion_cost=10.00,
        max_tokens=8192,
        context_window=2000000,
    )
    LIVE_2_5_FLASH_PREVIEW = ModelInfo(
        id="gemini-live-2.5-flash-preview",
        provider="gemini",
        type="audio",
        prompt_cost=0.50,
        completion_cost=2.00,
        max_tokens=8192,
        context_window=1000000,
    )

    # ===== Gemini 2.0 Series =====
    FLASH_2_0 = ModelInfo(
        id="gemini-2.0-flash",
        provider="gemini",
        type="chat",
        prompt_cost=0.10,
        completion_cost=0.40,
        max_tokens=8192,
        context_window=1000000,
    )
    FLASH_LITE_2_0 = ModelInfo(
        id="gemini-2.0-flash-lite",
        provider="gemini",
        type="chat",
        prompt_cost=0.075,
        completion_cost=0.30,
        max_tokens=8192,
        context_window=1000000,
    )
    FLASH_LIVE_2_0_001 = ModelInfo(
        id="gemini-2.0-flash-live-001",
        provider="gemini",
        type="audio",
        prompt_cost=0.35,
        completion_cost=1.50,
        max_tokens=8192,
        context_window=1000000,
    )

    # ===== Gemini Robotics =====
    ROBOTICS_ER_1_5 = ModelInfo(
        id="gemini-robotics-er-1.5-preview",
        provider="gemini",
        type="chat",
        prompt_cost=0.30,
        completion_cost=2.50,
        max_tokens=8192,
        context_window=1000000,
    )

    # ===== Gemini 1.5 Series (Stable) =====
    PRO_1_5 = ModelInfo(
        id="gemini-1.5-pro",
        provider="gemini",
        type="chat",
        prompt_cost=1.25,
        completion_cost=5.00,
        max_tokens=8192,
        context_window=2000000,
    )
    PRO_1_5_LATEST = ModelInfo(
        id="gemini-1.5-pro-latest",
        provider="gemini",
        type="chat",
        prompt_cost=1.25,
        completion_cost=5.00,
        max_tokens=8192,
        context_window=2000000,
    )
    FLASH_1_5 = ModelInfo(
        id="gemini-1.5-flash",
        provider="gemini",
        type="chat",
        prompt_cost=0.075,
        completion_cost=0.30,
        max_tokens=8192,
        context_window=1000000,
    )
    FLASH_1_5_LATEST = ModelInfo(
        id="gemini-1.5-flash-latest",
        provider="gemini",
        type="chat",
        prompt_cost=0.075,
        completion_cost=0.30,
        max_tokens=8192,
        context_window=1000000,
    )

    # ===== Gemini 1.0 Series (Legacy) =====
    PRO_1_0 = ModelInfo(
        id="gemini-1.0-pro",
        provider="gemini",
        type="chat",
        prompt_cost=0.50,
        completion_cost=1.50,
        max_tokens=8192,
        context_window=32000,
    )
    PRO = ModelInfo(
        id="gemini-pro",
        provider="gemini",
        type="chat",
        prompt_cost=0.50,
        completion_cost=1.50,
        max_tokens=8192,
        context_window=32000,
    )

    # ===== Gemma (Open Models - Free) =====
    GEMMA_3 = ModelInfo(
        id="gemma-3",
        provider="gemini",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=8192,
        context_window=8192,
    )
    GEMMA_3N = ModelInfo(
        id="gemma-3n",
        provider="gemini",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=8192,
        context_window=8192,
    )

    # ===== Embedding Models =====
    class Embeddings:
        """Google Gemini embedding models."""

        EMBEDDING_001 = ModelInfo(
            id="text-embedding-001",
            provider="gemini",
            type="embedding",
            prompt_cost=0.0,
            completion_cost=0.0,
            max_tokens=2048,
            context_window=2048,
        )
        EMBEDDING_004 = ModelInfo(
            id="text-embedding-004",
            provider="gemini",
            type="embedding",
            prompt_cost=0.0,
            completion_cost=0.0,
            max_tokens=2048,
            context_window=2048,
        )


# =============================================================================
# Ollama Local Models (11 total)
# =============================================================================


class Ollama:
    """Ollama local models - all free ($0.00)."""

    LLAMA_3_2 = ModelInfo(
        id="llama3.2",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    LLAMA_3_1 = ModelInfo(
        id="llama3.1",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    LLAMA_3 = ModelInfo(
        id="llama3",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    LLAMA_2 = ModelInfo(
        id="llama2",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=4096,
    )
    MISTRAL = ModelInfo(
        id="mistral",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    MIXTRAL = ModelInfo(
        id="mixtral",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=32768,
    )
    CODELLAMA = ModelInfo(
        id="codellama",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=16384,
    )
    PHI = ModelInfo(
        id="phi",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=2048,
        context_window=4096,
    )
    NEURAL_CHAT = ModelInfo(
        id="neural-chat",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    STARLING_LM = ModelInfo(
        id="starling-lm",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )
    QWEN = ModelInfo(
        id="qwen",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=32768,
    )
    GEMMA = ModelInfo(
        id="gemma",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=8192,
        context_window=8192,
    )
    VICUNA = ModelInfo(
        id="vicuna",
        provider="ollama",
        type="chat",
        prompt_cost=0.00,
        completion_cost=0.00,
        max_tokens=4096,
        context_window=8192,
    )


# =============================================================================
# Cohere Models (3 embedding models)
# =============================================================================


class Cohere:
    """Cohere embedding models."""

    # ===== Embedding Models =====
    class Embeddings:
        """Cohere embedding models."""

        EMBED_V3 = ModelInfo(
            id="embed-english-v3.0",
            provider="cohere",
            type="embedding",
            prompt_cost=0.10,
            completion_cost=0.0,
            max_tokens=512,
            context_window=512,
        )
        EMBED_MULTILINGUAL_V3 = ModelInfo(
            id="embed-multilingual-v3.0",
            provider="cohere",
            type="embedding",
            prompt_cost=0.10,
            completion_cost=0.0,
            max_tokens=512,
            context_window=512,
        )
        EMBED_V3_LIGHT = ModelInfo(
            id="embed-english-light-v3.0",
            provider="cohere",
            type="embedding",
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
"""Complete list of all 130 models across all providers (chat + embedding)."""

MODELS_BY_ID: Dict[str, ModelInfo] = {model.id: model for model in ALL_MODELS}
"""Quick lookup dictionary mapping model ID to ModelInfo."""


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
