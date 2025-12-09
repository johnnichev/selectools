"""Tests for the canonical model registry."""

import pytest

from selectools.models import ALL_MODELS, MODELS_BY_ID, Anthropic, Gemini, ModelInfo, Ollama, OpenAI
from selectools.pricing import PRICING


class TestModelInfo:
    """Tests for the ModelInfo dataclass."""

    def test_model_info_creation(self):
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

    def test_model_info_immutable(self):
        """Test that ModelInfo is frozen/immutable."""
        model = OpenAI.GPT_4O
        with pytest.raises((AttributeError, TypeError)):  # Frozen dataclass raises these
            model.prompt_cost = 999.99


class TestModelRegistry:
    """Tests for the complete model registry."""

    def test_all_models_count(self):
        """Test that we have all 130 models (120 chat + 10 embedding)."""
        assert len(ALL_MODELS) == 130

    def test_models_by_id_count(self):
        """Test that MODELS_BY_ID has same count as ALL_MODELS."""
        assert len(MODELS_BY_ID) == len(ALL_MODELS)

    def test_models_by_id_lookup(self):
        """Test looking up models by ID."""
        assert MODELS_BY_ID["gpt-4o"].provider == "openai"
        assert MODELS_BY_ID["claude-3-5-sonnet-20241022"].provider == "anthropic"
        assert MODELS_BY_ID["gemini-2.0-flash"].provider == "gemini"
        assert MODELS_BY_ID["llama3.2"].provider == "ollama"

    def test_all_models_unique_ids(self):
        """Test that all model IDs are unique."""
        model_ids = [m.id for m in ALL_MODELS]
        assert len(model_ids) == len(set(model_ids))

    def test_pricing_dict_matches(self):
        """Test that PRICING dict is derived from models."""
        for model in ALL_MODELS:
            assert model.id in PRICING
            assert PRICING[model.id]["prompt"] == model.prompt_cost
            assert PRICING[model.id]["completion"] == model.completion_cost


class TestOpenAIModels:
    """Tests for OpenAI model definitions."""

    def test_openai_model_count(self):
        """Test OpenAI has 67 models (64 chat + 3 embedding)."""
        openai_models = [m for m in ALL_MODELS if m.provider == "openai"]
        assert len(openai_models) == 67

    def test_openai_gpt4o(self):
        """Test GPT-4o model definition."""
        model = OpenAI.GPT_4O
        assert model.id == "gpt-4o"
        assert model.provider == "openai"
        assert model.type == "chat"
        assert model.prompt_cost == 2.50
        assert model.completion_cost == 10.00
        assert model.max_tokens == 16384
        assert model.context_window == 128000

    def test_openai_gpt4o_mini(self):
        """Test GPT-4o-mini model definition."""
        model = OpenAI.GPT_4O_MINI
        assert model.id == "gpt-4o-mini"
        assert model.prompt_cost == 0.15
        assert model.completion_cost == 0.60

    def test_openai_o1_pro(self):
        """Test o1-pro (most expensive OpenAI model)."""
        model = OpenAI.O1_PRO
        assert model.id == "o1-pro"
        assert model.prompt_cost == 150.00
        assert model.completion_cost == 600.00


class TestAnthropicModels:
    """Tests for Anthropic Claude model definitions."""

    def test_anthropic_model_count(self):
        """Test Anthropic has 20 models (18 chat + 2 embedding)."""
        anthropic_models = [m for m in ALL_MODELS if m.provider == "anthropic"]
        assert len(anthropic_models) == 20

    def test_anthropic_sonnet_4_5(self):
        """Test Claude Sonnet 4.5 model definition."""
        model = Anthropic.SONNET_4_5
        assert model.id == "claude-sonnet-4-5"
        assert model.provider == "anthropic"
        assert model.type == "chat"
        assert model.prompt_cost == 3.00
        assert model.completion_cost == 15.00
        assert model.max_tokens == 8192
        assert model.context_window == 200000

    def test_anthropic_opus_4_5(self):
        """Test Claude Opus 4.5 model definition."""
        model = Anthropic.OPUS_4_5
        assert model.id == "claude-opus-4-5"
        assert model.prompt_cost == 5.00
        assert model.completion_cost == 25.00

    def test_anthropic_haiku_3(self):
        """Test Claude Haiku 3 (cheapest Claude model)."""
        model = Anthropic.HAIKU_3
        assert model.id == "claude-3-haiku"
        assert model.prompt_cost == 0.25
        assert model.completion_cost == 1.25


class TestGeminiModels:
    """Tests for Google Gemini model definitions."""

    def test_gemini_model_count(self):
        """Test Gemini has 27 models (25 chat + 2 embedding)."""
        gemini_models = [m for m in ALL_MODELS if m.provider == "gemini"]
        assert len(gemini_models) == 27

    def test_gemini_flash_2_0(self):
        """Test Gemini 2.0 Flash model definition."""
        model = Gemini.FLASH_2_0
        assert model.id == "gemini-2.0-flash"
        assert model.provider == "gemini"
        assert model.type == "chat"
        assert model.prompt_cost == 0.10
        assert model.completion_cost == 0.40
        assert model.max_tokens == 8192
        assert model.context_window == 1000000

    def test_gemini_pro_3(self):
        """Test Gemini 3 Pro Preview model definition."""
        model = Gemini.PRO_3
        assert model.id == "gemini-3-pro-preview"
        assert model.prompt_cost == 2.00
        assert model.completion_cost == 12.00
        assert model.context_window == 2000000  # 2M tokens

    def test_gemini_gemma_free(self):
        """Test Gemma (free open model)."""
        model = Gemini.GEMMA_3
        assert model.id == "gemma-3"
        assert model.prompt_cost == 0.00
        assert model.completion_cost == 0.00


class TestOllamaModels:
    """Tests for Ollama local model definitions."""

    def test_ollama_model_count(self):
        """Test Ollama has 13 models."""
        ollama_models = [m for m in ALL_MODELS if m.provider == "ollama"]
        assert len(ollama_models) == 13

    def test_ollama_llama_3_2(self):
        """Test Llama 3.2 model definition."""
        model = Ollama.LLAMA_3_2
        assert model.id == "llama3.2"
        assert model.provider == "ollama"
        assert model.type == "chat"
        assert model.prompt_cost == 0.00
        assert model.completion_cost == 0.00
        assert model.max_tokens == 4096
        assert model.context_window == 8192

    def test_ollama_all_free(self):
        """Test that all Ollama models are free."""
        ollama_models = [m for m in ALL_MODELS if m.provider == "ollama"]
        for model in ollama_models:
            assert model.prompt_cost == 0.00
            assert model.completion_cost == 0.00


class TestModelMetadataCompleteness:
    """Tests to ensure all models have complete metadata."""

    def test_all_models_have_id(self):
        """Test that all models have an ID."""
        for model in ALL_MODELS:
            assert model.id
            assert isinstance(model.id, str)
            assert len(model.id) > 0

    def test_all_models_have_valid_provider(self):
        """Test that all models have a valid provider."""
        valid_providers = {"openai", "anthropic", "gemini", "ollama", "cohere"}
        for model in ALL_MODELS:
            assert model.provider in valid_providers

    def test_all_models_have_type(self):
        """Test that all models have a type (currently all chat)."""
        for model in ALL_MODELS:
            assert model.type in {"chat", "embedding", "image", "audio", "multimodal"}

    def test_all_models_have_pricing(self):
        """Test that all models have valid pricing."""
        for model in ALL_MODELS:
            assert model.prompt_cost >= 0.0
            assert model.completion_cost >= 0.0

    def test_all_models_have_max_tokens(self):
        """Test that all models have max_tokens > 0."""
        for model in ALL_MODELS:
            assert model.max_tokens > 0
            assert model.max_tokens <= 65536  # Reasonable upper bound

    def test_all_models_have_context_window(self):
        """Test that all models have context_window > 0."""
        for model in ALL_MODELS:
            assert model.context_window > 0
            assert model.context_window >= model.max_tokens


class TestModelTypes:
    """Tests for different model types."""

    def test_chat_models_majority(self):
        """Test that most models are chat type."""
        chat_models = [m for m in ALL_MODELS if m.type == "chat"]
        assert len(chat_models) >= 100  # Most should be chat

    def test_audio_models_exist(self):
        """Test that audio models exist."""
        audio_models = [m for m in ALL_MODELS if m.type == "audio"]
        assert len(audio_models) > 0

    def test_multimodal_models_exist(self):
        """Test that multimodal models exist."""
        multimodal_models = [m for m in ALL_MODELS if m.type == "multimodal"]
        assert len(multimodal_models) > 0


class TestModelPricing:
    """Tests for model pricing values."""

    def test_most_expensive_model(self):
        """Test identifying the most expensive model."""
        most_expensive = max(ALL_MODELS, key=lambda m: m.prompt_cost + m.completion_cost)
        assert most_expensive.id == "o1-pro"  # OpenAI o1-pro is most expensive

    def test_cheapest_paid_model(self):
        """Test finding cheapest non-free model."""
        paid_models = [m for m in ALL_MODELS if m.prompt_cost > 0]
        cheapest = min(paid_models, key=lambda m: m.prompt_cost + m.completion_cost)
        assert cheapest.prompt_cost + cheapest.completion_cost < 1.0  # Under $1 per 1M tokens

    def test_free_models_exist(self):
        """Test that free models exist (Ollama + Gemma)."""
        free_models = [m for m in ALL_MODELS if m.prompt_cost == 0 and m.completion_cost == 0]
        assert len(free_models) >= 13  # At least Ollama models


class TestProviderDefaults:
    """Tests for provider default models."""

    def test_openai_default_exists(self):
        """Test that OpenAI default model (GPT-4O) exists."""
        assert OpenAI.GPT_4O.id in MODELS_BY_ID

    def test_anthropic_default_exists(self):
        """Test that Anthropic default model exists."""
        assert Anthropic.SONNET_3_5_20241022.id in MODELS_BY_ID

    def test_gemini_default_exists(self):
        """Test that Gemini default model (2.0 Flash) exists."""
        assert Gemini.FLASH_2_0.id in MODELS_BY_ID

    def test_ollama_default_exists(self):
        """Test that Ollama default model (Llama 3.2) exists."""
        assert Ollama.LLAMA_3_2.id in MODELS_BY_ID
