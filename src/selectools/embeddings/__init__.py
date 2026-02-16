"""Embedding providers for generating vector embeddings from text."""

from .provider import EmbeddingProvider

# Providers are imported conditionally as they depend on optional packages
__all__ = [
    "EmbeddingProvider",
]

# Try to import optional embedding providers
try:
    from .openai import OpenAIEmbeddingProvider  # noqa: F401

    __all__.append("OpenAIEmbeddingProvider")
except ImportError:
    pass

try:
    from .anthropic import AnthropicEmbeddingProvider  # noqa: F401

    __all__.append("AnthropicEmbeddingProvider")
except ImportError:
    pass

try:
    from .gemini import GeminiEmbeddingProvider  # noqa: F401

    __all__.append("GeminiEmbeddingProvider")
except ImportError:
    pass

try:
    from .cohere import CohereEmbeddingProvider  # noqa: F401

    __all__.append("CohereEmbeddingProvider")
except ImportError:
    pass
