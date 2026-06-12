"""Tests for the canonical model registry."""

from __future__ import annotations

import pytest

from selectools.models import ALL_MODELS, MODELS_BY_ID, Anthropic, Gemini, ModelInfo, Ollama, OpenAI
from selectools.pricing import PRICING


class TestModelInfo:
    """Tests for the ModelInfo dataclass."""

    def test_model_info_creation(self) -> None:
        """Test creating a ModelInfo instance."""
        model = ModelInfo(
            id="test-model",
            provider="test",
            type="chat",
            prompt_cost=1.0,
            completion_cost=2.0,
            max_tokens=4096,
            context_window=8192,
        )
        assert model.id == "test-model"
        assert model.provider == "test"
        assert model.type == "chat"
        assert model.prompt_cost == 1.0
        assert model.completion_cost == 2.0
        assert model.max_tokens == 4096
        assert model.context_window == 8192

    def test_model_info_immutable(self) -> None:
        """Test that ModelInfo is frozen/immutable."""
        model = OpenAI.GPT_4O
        with pytest.raises((AttributeError, TypeError)):  # Frozen dataclass raises these
            model.prompt_cost = 999.99


class TestModelRegistry:
    """Tests for the complete model registry."""

    def test_all_models_count(self) -> None:
        """Test that we have all registered models."""
        assert len(ALL_MODELS) == 115

    def test_models_by_id_count(self) -> None:
        """Test that MODELS_BY_ID has same count as ALL_MODELS."""
        assert len(MODELS_BY_ID) == len(ALL_MODELS)

    def test_models_by_id_lookup(self) -> None:
        """Test looking up models by ID."""
        assert MODELS_BY_ID["gpt-4o"].provider == "openai"
        assert MODELS_BY_ID["claude-sonnet-4-6"].provider == "anthropic"
        assert MODELS_BY_ID["gemini-2.5-flash"].provider == "gemini"
        assert MODELS_BY_ID["llama3.2"].provider == "ollama"

    def test_all_models_unique_ids(self) -> None:
        """Test that all model IDs are unique."""
        model_ids = [m.id for m in ALL_MODELS]
        assert len(model_ids) == len(set(model_ids))

    def test_pricing_dict_matches(self) -> None:
        """Test that PRICING dict is derived from models."""
        for model in ALL_MODELS:
            assert model.id in PRICING
            assert PRICING[model.id]["prompt"] == model.prompt_cost
            assert PRICING[model.id]["completion"] == model.completion_cost


class TestOpenAIModels:
    """Tests for OpenAI model definitions."""

    def test_openai_model_count(self) -> None:
        """Test OpenAI model count."""
        openai_models = [m for m in ALL_MODELS if m.provider == "openai"]
        assert len(openai_models) == 65

    def test_openai_gpt4o(self) -> None:
        """Test GPT-4o model definition."""
        model = OpenAI.GPT_4O
        assert model.id == "gpt-4o"
        assert model.provider == "openai"
        assert model.type == "chat"
        assert model.prompt_cost == 2.50
        assert model.completion_cost == 10.00
        assert model.max_tokens == 16384
        assert model.context_window == 128000

    def test_openai_gpt4o_mini(self) -> None:
        """Test GPT-4o-mini model definition."""
        model = OpenAI.GPT_4O_MINI
        assert model.id == "gpt-4o-mini"
        assert model.prompt_cost == 0.15
        assert model.completion_cost == 0.60

    def test_openai_o1_pro(self) -> None:
        """Test o1-pro (most expensive OpenAI model)."""
        model = OpenAI.O1_PRO
        assert model.id == "o1-pro"
        assert model.prompt_cost == 150.00
        assert model.completion_cost == 600.00


class TestAnthropicModels:
    """Tests for Anthropic Claude model definitions."""

    def test_anthropic_model_count(self) -> None:
        """Test Anthropic model count."""
        anthropic_models = [m for m in ALL_MODELS if m.provider == "anthropic"]
        assert len(anthropic_models) == 13

    def test_anthropic_sonnet_4_5(self) -> None:
        """Test Claude Sonnet 4.5 model definition."""
        model = Anthropic.SONNET_4_5
        assert model.id == "claude-sonnet-4-5"
        assert model.provider == "anthropic"
        assert model.type == "chat"
        assert model.prompt_cost == 3.00
        assert model.completion_cost == 15.00
        assert model.max_tokens == 64000
        assert model.context_window == 200000

    def test_anthropic_opus_4_5(self) -> None:
        """Test Claude Opus 4.5 model definition."""
        model = Anthropic.OPUS_4_5
        assert model.id == "claude-opus-4-5"
        assert model.prompt_cost == 5.00
        assert model.completion_cost == 25.00

    def test_anthropic_haiku_4_5(self) -> None:
        """Test Claude Haiku 4.5 (cheapest Claude model)."""
        model = Anthropic.HAIKU_4_5
        assert model.id == "claude-haiku-4-5"
        assert model.prompt_cost == 1.00
        assert model.completion_cost == 5.00


class TestGeminiModels:
    """Tests for Google Gemini model definitions."""

    def test_gemini_model_count(self) -> None:
        """Test Gemini has 21 models (19 chat/audio/image + 2 embedding)."""
        gemini_models = [m for m in ALL_MODELS if m.provider == "gemini"]
        assert len(gemini_models) == 21

    def test_gemini_flash_2_5(self) -> None:
        """Test Gemini 2.5 Flash model definition."""
        model = Gemini.FLASH_2_5
        assert model.id == "gemini-2.5-flash"
        assert model.provider == "gemini"
        assert model.type == "chat"
        assert model.prompt_cost == 0.30
        assert model.completion_cost == 2.50
        assert model.max_tokens == 65536
        assert model.context_window == 1048576

    def test_gemini_pro_3_1(self) -> None:
        """Test Gemini 3.1 Pro Preview model definition."""
        model = Gemini.PRO_3_1
        assert model.id == "gemini-3.1-pro-preview"
        assert model.prompt_cost == 2.00
        assert model.completion_cost == 12.00
        assert model.context_window == 1048576

    def test_gemini_embedding(self) -> None:
        """Test Gemini embedding model definition."""
        model = Gemini.Embeddings.EMBEDDING_001
        assert model.id == "gemini-embedding-001"
        assert model.prompt_cost == 0.15
        assert model.completion_cost == 0.00


class TestOllamaModels:
    """Tests for Ollama local model definitions."""

    def test_ollama_model_count(self) -> None:
        """Test Ollama has 13 models."""
        ollama_models = [m for m in ALL_MODELS if m.provider == "ollama"]
        assert len(ollama_models) == 13

    def test_ollama_llama_3_2(self) -> None:
        """Test Llama 3.2 model definition."""
        model = Ollama.LLAMA_3_2
        assert model.id == "llama3.2"
        assert model.provider == "ollama"
        assert model.type == "chat"
        assert model.prompt_cost == 0.00
        assert model.completion_cost == 0.00
        assert model.max_tokens == 4096
        assert model.context_window == 8192

    def test_ollama_all_free(self) -> None:
        """Test that all Ollama models are free."""
        ollama_models = [m for m in ALL_MODELS if m.provider == "ollama"]
        for model in ollama_models:
            assert model.prompt_cost == 0.00
            assert model.completion_cost == 0.00


class TestModelMetadataCompleteness:
    """Tests to ensure all models have complete metadata."""

    def test_all_models_have_id(self) -> None:
        """Test that all models have an ID."""
        for model in ALL_MODELS:
            assert model.id
            assert isinstance(model.id, str)
            assert len(model.id) > 0

    def test_all_models_have_valid_provider(self) -> None:
        """Test that all models have a valid provider."""
        valid_providers = {"openai", "anthropic", "gemini", "ollama", "cohere"}
        for model in ALL_MODELS:
            assert model.provider in valid_providers

    def test_all_models_have_type(self) -> None:
        """Test that all models have a type (currently all chat)."""
        for model in ALL_MODELS:
            assert model.type in {"chat", "embedding", "image", "audio", "multimodal"}

    def test_all_models_have_pricing(self) -> None:
        """Test that all models have valid pricing."""
        for model in ALL_MODELS:
            assert model.prompt_cost >= 0.0
            assert model.completion_cost >= 0.0

    def test_all_models_have_max_tokens(self) -> None:
        """Test that all models have max_tokens > 0."""
        for model in ALL_MODELS:
            assert model.max_tokens > 0
            assert model.max_tokens <= 200000  # Reasonable upper bound

    def test_all_models_have_context_window(self) -> None:
        """Test that all models have context_window > 0."""
        for model in ALL_MODELS:
            assert model.context_window > 0
            # Audio (TTS) models can have a larger output limit than input
            # context (e.g. Gemini TTS: 8k in / 16k out), so exempt them.
            if model.type != "audio":
                assert model.context_window >= model.max_tokens


class TestModelTypes:
    """Tests for different model types."""

    def test_chat_models_majority(self) -> None:
        """Test that most models are chat type."""
        chat_models = [m for m in ALL_MODELS if m.type == "chat"]
        assert len(chat_models) >= 80  # Most should be chat

    def test_audio_models_exist(self) -> None:
        """Test that audio models exist."""
        audio_models = [m for m in ALL_MODELS if m.type == "audio"]
        assert len(audio_models) > 0

    def test_multimodal_models_exist(self) -> None:
        """Test that multimodal models exist."""
        multimodal_models = [m for m in ALL_MODELS if m.type == "multimodal"]
        assert len(multimodal_models) > 0


class TestModelPricing:
    """Tests for model pricing values."""

    def test_most_expensive_model(self) -> None:
        """Test identifying the most expensive model."""
        most_expensive = max(ALL_MODELS, key=lambda m: m.prompt_cost + m.completion_cost)
        assert most_expensive.id == "o1-pro"  # OpenAI o1-pro is most expensive

    def test_cheapest_paid_model(self) -> None:
        """Test finding cheapest non-free model."""
        paid_models = [m for m in ALL_MODELS if m.prompt_cost > 0]
        cheapest = min(paid_models, key=lambda m: m.prompt_cost + m.completion_cost)
        assert cheapest.prompt_cost + cheapest.completion_cost < 1.0  # Under $1 per 1M tokens

    def test_free_models_exist(self) -> None:
        """Test that free models exist (Ollama local models)."""
        free_models = [m for m in ALL_MODELS if m.prompt_cost == 0 and m.completion_cost == 0]
        assert len(free_models) >= 13  # At least Ollama models


class TestProviderDefaults:
    """Tests for provider default models."""

    def test_openai_default_exists(self) -> None:
        """Test that OpenAI provider default model (gpt-5-mini) exists."""
        assert OpenAI.GPT_5_MINI.id in MODELS_BY_ID

    def test_anthropic_default_exists(self) -> None:
        """Test that Anthropic provider default model (claude-sonnet-4-6) exists."""
        assert Anthropic.SONNET_4_6.id in MODELS_BY_ID

    def test_gemini_default_exists(self) -> None:
        """Test that Gemini provider default model (gemini-3-flash-preview) exists."""
        assert Gemini.FLASH_3_PREVIEW.id in MODELS_BY_ID

    def test_ollama_default_exists(self) -> None:
        """Test that Ollama default model (Llama 3.2) exists."""
        assert Ollama.LLAMA_3_2.id in MODELS_BY_ID


class TestMaxCompletionTokensDetection:
    """Tests for the max_tokens → max_completion_tokens migration."""

    def test_gpt5_models_use_max_completion_tokens(self) -> None:
        from selectools.providers.openai_provider import _uses_max_completion_tokens

        assert _uses_max_completion_tokens("gpt-5") is True
        assert _uses_max_completion_tokens("gpt-5-mini") is True
        assert _uses_max_completion_tokens("gpt-5-nano") is True
        assert _uses_max_completion_tokens("gpt-5.1") is True
        assert _uses_max_completion_tokens("gpt-5.2") is True
        assert _uses_max_completion_tokens("gpt-5.2-pro") is True

    def test_gpt41_models_use_max_completion_tokens(self) -> None:
        from selectools.providers.openai_provider import _uses_max_completion_tokens

        assert _uses_max_completion_tokens("gpt-4.1") is True
        assert _uses_max_completion_tokens("gpt-4.1-mini") is True
        assert _uses_max_completion_tokens("gpt-4.1-nano") is True

    def test_o_series_uses_max_completion_tokens(self) -> None:
        from selectools.providers.openai_provider import _uses_max_completion_tokens

        assert _uses_max_completion_tokens("o1") is True
        assert _uses_max_completion_tokens("o1-mini") is True
        assert _uses_max_completion_tokens("o1-pro") is True
        assert _uses_max_completion_tokens("o3") is True
        assert _uses_max_completion_tokens("o3-mini") is True
        assert _uses_max_completion_tokens("o3-pro") is True
        assert _uses_max_completion_tokens("o4-mini") is True

    def test_codex_uses_max_completion_tokens(self) -> None:
        from selectools.providers.openai_provider import _uses_max_completion_tokens

        assert _uses_max_completion_tokens("codex-mini-latest") is True

    def test_legacy_models_use_max_tokens(self) -> None:
        from selectools.providers.openai_provider import _uses_max_completion_tokens

        assert _uses_max_completion_tokens("gpt-4o") is False
        assert _uses_max_completion_tokens("gpt-4o-mini") is False
        assert _uses_max_completion_tokens("gpt-4-turbo") is False
        assert _uses_max_completion_tokens("gpt-4") is False
        assert _uses_max_completion_tokens("gpt-3.5-turbo") is False
        assert _uses_max_completion_tokens("davinci-002") is False

    def test_all_registry_models_covered(self) -> None:
        """Every OpenAI chat model in the registry should resolve without error."""
        from selectools.providers.openai_provider import _uses_max_completion_tokens

        for model in ALL_MODELS:
            if model.provider == "openai" and model.type == "chat":
                result = _uses_max_completion_tokens(model.id)
                assert isinstance(result, bool), f"Failed for {model.id}"
