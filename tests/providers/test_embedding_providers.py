"""
Unit tests for embedding providers with mocked APIs.

Tests all 4 embedding providers:
- OpenAIEmbeddingProvider
- AnthropicEmbeddingProvider (Voyage AI)
- GeminiEmbeddingProvider
- CohereEmbeddingProvider
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from selectools.embeddings import (
    AnthropicEmbeddingProvider,
    CohereEmbeddingProvider,
    GeminiEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from selectools.models import Anthropic, Cohere, Gemini, OpenAI

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_openai_response() -> Mock:
    """Mock OpenAI embedding API response."""
    mock_response = Mock()
    mock_data_0 = Mock()
    mock_data_0.embedding = [0.1] * 1536
    mock_data_0.index = 0
    mock_data_1 = Mock()
    mock_data_1.embedding = [0.2] * 1536
    mock_data_1.index = 1
    mock_response.data = [mock_data_0, mock_data_1]
    mock_response.usage = Mock(prompt_tokens=100)
    return mock_response


@pytest.fixture
def mock_voyage_response() -> Mock:
    """Mock Voyage AI embedding API response."""
    mock_response = Mock()
    mock_response.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_response.usage = Mock(total_tokens=100)
    return mock_response


@pytest.fixture
def mock_gemini_response() -> Mock:
    """Mock Gemini embedding API response.

    gemini-embedding-001 / gemini-embedding-2 return 3072-dim vectors by
    default — the provider never requests ``output_dimensionality``.
    """
    mock_response = Mock()
    mock_embedding1 = Mock()
    mock_embedding1.values = [0.1] * 3072
    mock_embedding2 = Mock()
    mock_embedding2.values = [0.2] * 3072
    mock_response.embeddings = [mock_embedding1, mock_embedding2]
    return mock_response


@pytest.fixture
def mock_cohere_response() -> Mock:
    """Mock Cohere embedding API response."""
    mock_response = Mock()
    mock_response.embeddings = [[0.1] * 1024, [0.2] * 1024]
    mock_response.meta = Mock(tokens=Mock(input_tokens=100))
    return mock_response


# ============================================================================
# OpenAI Embedding Provider Tests
# ============================================================================


class TestOpenAIEmbeddingProvider:
    """Test OpenAI embedding provider."""

    def test_initialization(self) -> None:
        """Test provider initialization."""
        with patch("openai.OpenAI"):
            provider = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
            assert provider.model == OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id
            assert provider.dimension == 1536

    def test_embed_text(self, mock_openai_response: Mock) -> None:
        """Test embedding a single text."""
        with patch("openai.OpenAI") as MockClient:
            mock_client = MockClient.return_value
            mock_client.embeddings.create.return_value = mock_openai_response

            provider = OpenAIEmbeddingProvider()
            embedding = provider.embed_text("Hello world")

            assert isinstance(embedding, list)
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)
            mock_client.embeddings.create.assert_called_once()

    def test_embed_texts_batch(self, mock_openai_response: Mock) -> None:
        """Test embedding multiple texts in batch."""
        with patch("openai.OpenAI") as MockClient:
            mock_client = MockClient.return_value
            mock_client.embeddings.create.return_value = mock_openai_response

            provider = OpenAIEmbeddingProvider()
            embeddings = provider.embed_texts(["Hello", "World"])

            assert isinstance(embeddings, list)
            assert len(embeddings) == 2
            assert all(len(e) == 1536 for e in embeddings)
            mock_client.embeddings.create.assert_called_once()

    def test_embed_query(self, mock_openai_response: Mock) -> None:
        """Test embedding a query (same as text for OpenAI)."""
        with patch("openai.OpenAI") as MockClient:
            mock_client = MockClient.return_value
            mock_client.embeddings.create.return_value = mock_openai_response

            provider = OpenAIEmbeddingProvider()
            embedding = provider.embed_query("What is AI?")

            assert isinstance(embedding, list)
            assert len(embedding) == 1536

    def test_dimension_property(self) -> None:
        """Test dimension property for different models."""
        with patch("openai.OpenAI"):
            # Small model
            provider_small = OpenAIEmbeddingProvider(
                model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id
            )
            assert provider_small.dimension == 1536

            # Large model
            provider_large = OpenAIEmbeddingProvider(
                model=OpenAI.Embeddings.TEXT_EMBEDDING_3_LARGE.id
            )
            assert provider_large.dimension == 3072

    def test_error_handling_invalid_key(self) -> None:
        """Test error handling with invalid API key."""
        with patch("openai.OpenAI") as MockClient:
            mock_client = MockClient.return_value
            mock_client.embeddings.create.side_effect = Exception("Invalid API key")

            provider = OpenAIEmbeddingProvider()

            with pytest.raises(Exception, match="Invalid API key"):
                provider.embed_text("Test")

    def test_timeout_and_retries_wired_to_client(self) -> None:
        """timeout + max_retries are passed to the OpenAI SDK client.

        Retry/backoff on 429/5xx is delegated to the SDK's built-in
        ``max_retries`` mechanism rather than reimplemented here.
        """
        with patch("openai.OpenAI") as MockClient:
            OpenAIEmbeddingProvider(timeout=12.5, max_retries=5)
            _, kwargs = MockClient.call_args
            assert kwargs["timeout"] == 12.5
            assert kwargs["max_retries"] == 5

    def test_default_timeout_and_retries(self) -> None:
        """Sane defaults are applied so a hung call cannot block forever."""
        with patch("openai.OpenAI") as MockClient:
            OpenAIEmbeddingProvider()
            _, kwargs = MockClient.call_args
            assert kwargs["timeout"] == 60.0
            assert kwargs["max_retries"] == 2


# ============================================================================
# Anthropic (Voyage AI) Embedding Provider Tests
# ============================================================================


class TestAnthropicEmbeddingProvider:
    """Test Anthropic/Voyage AI embedding provider."""

    def test_initialization(self) -> None:
        """Test provider initialization."""
        mock_voyage = Mock()
        mock_client = Mock()
        mock_voyage.Client = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"voyageai": mock_voyage}):
            provider = AnthropicEmbeddingProvider(model=Anthropic.Embeddings.VOYAGE_3_LITE.id)
            assert provider.model == Anthropic.Embeddings.VOYAGE_3_LITE.id
            assert provider.dimension == 1024

    def test_embed_text_document_type(self, mock_voyage_response: Mock) -> None:
        """Test embedding text with document input type."""
        mock_voyage = Mock()
        mock_client = Mock()
        mock_client.embed.return_value = mock_voyage_response
        mock_voyage.Client = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"voyageai": mock_voyage}):
            provider = AnthropicEmbeddingProvider()
            embedding = provider.embed_text("Document text")

            assert isinstance(embedding, list)
            assert len(embedding) == 1024
            # Verify input_type="document" was used
            call_args = mock_client.embed.call_args
            assert call_args[1]["input_type"] == "document"

    def test_embed_query_query_type(self, mock_voyage_response: Mock) -> None:
        """Test embedding query with query input type."""
        mock_voyage = Mock()
        mock_client = Mock()
        mock_client.embed.return_value = mock_voyage_response
        mock_voyage.Client = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"voyageai": mock_voyage}):
            provider = AnthropicEmbeddingProvider()
            embedding = provider.embed_query("Search query")

            assert isinstance(embedding, list)
            # Verify input_type="query" was used
            call_args = mock_client.embed.call_args
            assert call_args[1]["input_type"] == "query"

    def test_embed_texts_batch(self, mock_voyage_response: Mock) -> None:
        """Test embedding multiple texts."""
        mock_voyage = Mock()
        mock_client = Mock()
        mock_client.embed.return_value = mock_voyage_response
        mock_voyage.Client = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"voyageai": mock_voyage}):
            provider = AnthropicEmbeddingProvider()
            embeddings = provider.embed_texts(["Text 1", "Text 2"])

            assert isinstance(embeddings, list)
            assert len(embeddings) == 2

    def test_missing_voyageai_import(self) -> None:
        """Test error when voyageai is not installed."""
        with patch.dict("sys.modules", {"voyageai": None}):
            with pytest.raises(ImportError, match="voyageai package required"):
                AnthropicEmbeddingProvider()


# ============================================================================
# Gemini Embedding Provider Tests
# ============================================================================


class TestGeminiEmbeddingProvider:
    """Test Google Gemini embedding provider."""

    def test_initialization(self) -> None:
        """Test provider initialization."""
        mock_google = Mock()
        mock_genai = Mock()
        mock_client = Mock()
        mock_genai.Client = Mock(return_value=mock_client)
        mock_google.genai = mock_genai

        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            provider = GeminiEmbeddingProvider(
                model=Gemini.Embeddings.EMBEDDING_001.id, api_key="test_key"
            )
            assert provider.model == Gemini.Embeddings.EMBEDDING_001.id
            # Default (untruncated) output dimensionality for
            # gemini-embedding-001 — the provider never requests
            # output_dimensionality, so actual vectors are 3072-dim.
            assert provider.dimension == 3072

    def test_embed_text_document_task(self, mock_gemini_response: Mock) -> None:
        """Test embedding text with RETRIEVAL_DOCUMENT task."""
        mock_google = Mock()
        mock_genai = Mock()
        mock_types = Mock()
        mock_client = Mock()
        mock_models = Mock()
        mock_models.embed_content.return_value = mock_gemini_response
        mock_client.models = mock_models
        mock_genai.Client = Mock(return_value=mock_client)
        mock_genai.types = mock_types
        mock_google.genai = mock_genai

        with patch.dict(
            "sys.modules",
            {"google": mock_google, "google.genai": mock_genai, "google.genai.types": mock_types},
        ):
            provider = GeminiEmbeddingProvider(api_key="test_key")
            embedding = provider.embed_text("Document content")

            assert isinstance(embedding, list)
            assert len(embedding) == 3072
            mock_models.embed_content.assert_called_once()

    def test_embed_query_query_task(self, mock_gemini_response: Mock) -> None:
        """Test embedding query with RETRIEVAL_QUERY task."""
        mock_google = Mock()
        mock_genai = Mock()
        mock_types = Mock()
        mock_client = Mock()
        mock_models = Mock()
        mock_models.embed_content.return_value = mock_gemini_response
        mock_client.models = mock_models
        mock_genai.Client = Mock(return_value=mock_client)
        mock_genai.types = mock_types
        mock_google.genai = mock_genai

        with patch.dict(
            "sys.modules",
            {"google": mock_google, "google.genai": mock_genai, "google.genai.types": mock_types},
        ):
            provider = GeminiEmbeddingProvider(api_key="test_key")
            embedding = provider.embed_query("What is AI?")

            assert isinstance(embedding, list)
            assert len(embedding) == 3072
            mock_models.embed_content.assert_called()

    def test_embed_texts_batch(self, mock_gemini_response: Mock) -> None:
        """Test embedding multiple texts."""
        mock_google = Mock()
        mock_genai = Mock()
        mock_types = Mock()
        mock_client = Mock()
        mock_models = Mock()
        mock_models.embed_content.return_value = mock_gemini_response
        mock_client.models = mock_models
        mock_genai.Client = Mock(return_value=mock_client)
        mock_genai.types = mock_types
        mock_google.genai = mock_genai

        with patch.dict(
            "sys.modules",
            {"google": mock_google, "google.genai": mock_genai, "google.genai.types": mock_types},
        ):
            provider = GeminiEmbeddingProvider(api_key="test_key")
            embeddings = provider.embed_texts(["Text 1", "Text 2"])

            assert isinstance(embeddings, list)
            assert len(embeddings) == 2

    def test_missing_api_key(self) -> None:
        """Test error when API key is not provided."""
        mock_google = Mock()
        mock_genai = Mock()
        # Simulate missing API key by making Client raise ValueError
        mock_genai.Client = Mock(
            side_effect=ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        )
        mock_google.genai = mock_genai

        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="API_KEY"):
                    GeminiEmbeddingProvider()

    def test_embedding_cost(self, mock_gemini_response: Mock) -> None:
        """Test Gemini embedding cost (gemini-embedding-001: $0.15/1M tokens)."""
        mock_google = Mock()
        mock_genai = Mock()
        mock_types = Mock()
        mock_client = Mock()
        mock_models = Mock()
        mock_models.embed_content.return_value = mock_gemini_response
        mock_client.models = mock_models
        mock_genai.Client = Mock(return_value=mock_client)
        mock_genai.types = mock_types
        mock_google.genai = mock_genai

        with patch.dict(
            "sys.modules",
            {"google": mock_google, "google.genai": mock_genai, "google.genai.types": mock_types},
        ):
            provider = GeminiEmbeddingProvider(api_key="test_key")
            provider.embed_text("Test")

            # gemini-embedding-001 is $0.15 per 1M tokens
            from selectools.pricing import calculate_embedding_cost

            cost = calculate_embedding_cost(provider.model, 100)
            assert cost == pytest.approx(100 / 1_000_000 * 0.15)


# ============================================================================
# Gemini Dimension Semantics Regression Tests
# ============================================================================


class TestGeminiDimensionSemantics:
    """Regression tests: declared dimension == actual request/response semantics.

    The provider never passes ``output_dimensionality`` (no MRL truncation
    requested), so the API returns each model's default dimensionality:
    3072 for gemini-embedding-001 and gemini-embedding-2 per
    https://ai.google.dev/gemini-api/docs/embeddings. The declared
    ``dimension`` property must match those actual vectors — it previously
    claimed 768 while real responses were 3072-dim.
    """

    @staticmethod
    def _patched_provider(model: str, response: Mock | None = None) -> Any:
        mock_google = Mock()
        mock_genai = Mock()
        mock_types = Mock()
        mock_client = Mock()
        mock_models = Mock()
        if response is not None:
            mock_models.embed_content.return_value = response
        mock_client.models = mock_models
        mock_genai.Client = Mock(return_value=mock_client)
        mock_genai.types = mock_types
        mock_google.genai = mock_genai
        modules = {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }
        return modules, mock_types, mock_models

    def test_request_does_not_truncate_dimensions(self, mock_gemini_response: Mock) -> None:
        """The embed request must not pass output_dimensionality (MRL truncation)."""
        modules, mock_types, _ = self._patched_provider(
            "gemini-embedding-001", mock_gemini_response
        )
        with patch.dict("sys.modules", modules):
            provider = GeminiEmbeddingProvider(api_key="test_key")
            provider.embed_text("Test")
            # Only task_type is sent — no output_dimensionality, so the API
            # returns the model default (3072 for gemini-embedding-001).
            mock_types.EmbedContentConfig.assert_called_once_with(task_type="retrieval_document")

    def test_declared_dimension_matches_actual_response(self, mock_gemini_response: Mock) -> None:
        """provider.dimension must equal the length of vectors the API returns."""
        modules, _, _ = self._patched_provider("gemini-embedding-001", mock_gemini_response)
        with patch.dict("sys.modules", modules):
            provider = GeminiEmbeddingProvider(api_key="test_key")
            embedding = provider.embed_text("Test")
            assert provider.dimension == len(embedding) == 3072

    def test_gemini_embedding_2_default_dimension(self) -> None:
        """gemini-embedding-2 also defaults to 3072 (was falling to the 768 fallback)."""
        modules, _, _ = self._patched_provider(Gemini.Embeddings.EMBEDDING_2.id)
        with patch.dict("sys.modules", modules):
            provider = GeminiEmbeddingProvider(
                model=Gemini.Embeddings.EMBEDDING_2.id, api_key="test_key"
            )
            assert provider.dimension == 3072

    def test_legacy_embedding_004_dimension_unchanged(self) -> None:
        """Retired text-embedding-004 was natively 768-dim — keep that mapping."""
        modules, _, _ = self._patched_provider("text-embedding-004")
        with patch.dict("sys.modules", modules):
            provider = GeminiEmbeddingProvider(model="text-embedding-004", api_key="test_key")
            assert provider.dimension == 768


# ============================================================================
# Cohere Embedding Provider Tests
# ============================================================================


class TestCohereEmbeddingProvider:
    """Test Cohere embedding provider."""

    def test_initialization(self) -> None:
        """Test provider initialization."""
        mock_cohere_lib = Mock()
        mock_client = Mock()
        mock_cohere_lib.Client = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"cohere": mock_cohere_lib}):
            provider = CohereEmbeddingProvider(
                model=Cohere.Embeddings.EMBED_V3.id, api_key="test_key"
            )
            assert provider.model == Cohere.Embeddings.EMBED_V3.id
            assert provider.dimension == 1024

    def test_embed_text_document_type(self, mock_cohere_response: Mock) -> None:
        """Test embedding text with search_document input type."""
        mock_cohere_lib = Mock()
        mock_client = Mock()
        mock_client.embed.return_value = mock_cohere_response
        mock_cohere_lib.Client = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"cohere": mock_cohere_lib}):
            provider = CohereEmbeddingProvider(api_key="test_key")
            embedding = provider.embed_text("Document text")

            assert isinstance(embedding, list)
            assert len(embedding) == 1024
            # Verify input_type="search_document"
            call_args = mock_client.embed.call_args
            assert call_args[1]["input_type"] == "search_document"

    def test_embed_query_query_type(self, mock_cohere_response: Mock) -> None:
        """Test embedding query with search_query input type."""
        mock_cohere_lib = Mock()
        mock_client = Mock()
        mock_client.embed.return_value = mock_cohere_response
        mock_cohere_lib.Client = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"cohere": mock_cohere_lib}):
            provider = CohereEmbeddingProvider(api_key="test_key")
            embedding = provider.embed_query("Search query")

            assert isinstance(embedding, list)
            # Verify input_type="search_query"
            call_args = mock_client.embed.call_args
            assert call_args[1]["input_type"] == "search_query"

    def test_embed_texts_batch(self, mock_cohere_response: Mock) -> None:
        """Test embedding multiple texts."""
        mock_cohere_lib = Mock()
        mock_client = Mock()
        mock_client.embed.return_value = mock_cohere_response
        mock_cohere_lib.Client = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"cohere": mock_cohere_lib}):
            provider = CohereEmbeddingProvider(api_key="test_key")
            embeddings = provider.embed_texts(["Text 1", "Text 2"])

            assert isinstance(embeddings, list)
            assert len(embeddings) == 2

    def test_missing_cohere_import(self) -> None:
        """Test error when cohere is not installed."""
        with patch.dict("sys.modules", {"cohere": None}):
            with pytest.raises(ImportError, match="cohere package required"):
                CohereEmbeddingProvider()

    def test_multilingual_model(self) -> None:
        """Test multilingual model support."""
        mock_cohere_lib = Mock()
        mock_client = Mock()
        mock_cohere_lib.Client = Mock(return_value=mock_client)

        with patch.dict("sys.modules", {"cohere": mock_cohere_lib}):
            provider = CohereEmbeddingProvider(
                model=Cohere.Embeddings.EMBED_MULTILINGUAL_V3.id, api_key="test_key"
            )
            assert provider.model == Cohere.Embeddings.EMBED_MULTILINGUAL_V3.id


# ============================================================================
# Cross-Provider Tests
# ============================================================================


class TestEmbeddingProviderInterface:
    """Test that all providers implement the same interface."""

    def _create_provider(self, provider_class: Any) -> Any:
        """Helper to create provider with proper mocking."""
        mock_client = Mock()

        if provider_class == OpenAIEmbeddingProvider:
            mock_openai = Mock()
            mock_openai.OpenAI = Mock(return_value=mock_client)
            with patch.dict("sys.modules", {"openai": mock_openai}):
                return provider_class()
        elif provider_class == AnthropicEmbeddingProvider:
            mock_voyage = Mock()
            mock_voyage.Client = Mock(return_value=mock_client)
            with patch.dict("sys.modules", {"voyageai": mock_voyage}):
                return provider_class()
        elif provider_class == GeminiEmbeddingProvider:
            mock_google = Mock()
            mock_genai = Mock()
            mock_genai.Client = Mock(return_value=mock_client)
            mock_google.genai = mock_genai
            with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
                return provider_class(api_key="test")
        elif provider_class == CohereEmbeddingProvider:
            mock_cohere = Mock()
            mock_cohere.Client = Mock(return_value=mock_client)
            with patch.dict("sys.modules", {"cohere": mock_cohere}):
                return provider_class()
        return None

    def test_all_providers_have_embed_text(self) -> None:
        """Test all providers have embed_text method."""
        providers = [
            OpenAIEmbeddingProvider,
            AnthropicEmbeddingProvider,
            GeminiEmbeddingProvider,
            CohereEmbeddingProvider,
        ]

        for provider_class in providers:
            provider = self._create_provider(provider_class)
            assert hasattr(provider, "embed_text")
            assert callable(provider.embed_text)

    def test_all_providers_have_embed_texts(self) -> None:
        """Test all providers have embed_texts method."""
        providers = [
            OpenAIEmbeddingProvider,
            AnthropicEmbeddingProvider,
            GeminiEmbeddingProvider,
            CohereEmbeddingProvider,
        ]

        for provider_class in providers:
            provider = self._create_provider(provider_class)
            assert hasattr(provider, "embed_texts")
            assert callable(provider.embed_texts)

    def test_all_providers_have_dimension(self) -> None:
        """Test all providers have dimension property."""
        providers = [
            OpenAIEmbeddingProvider,
            AnthropicEmbeddingProvider,
            GeminiEmbeddingProvider,
            CohereEmbeddingProvider,
        ]

        for provider_class in providers:
            provider = self._create_provider(provider_class)
            assert hasattr(provider, "dimension")
            assert isinstance(provider.dimension, int)
            assert provider.dimension > 0
