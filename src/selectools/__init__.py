"""Public exports for the selectools package."""

__version__ = "0.12.0"

# Import submodules (lazy loading for optional dependencies)
from . import embeddings, models, rag, toolbox
from .agent import Agent, AgentConfig
from .analytics import AgentAnalytics, ToolMetrics
from .cache import Cache, CacheKeyBuilder, CacheStats, InMemoryCache
from .exceptions import (
    MemoryLimitExceededError,
    ProviderConfigurationError,
    SelectoolsError,
    ToolExecutionError,
    ToolValidationError,
)
from .memory import ConversationMemory
from .models import ALL_MODELS, MODELS_BY_ID, Anthropic, Cohere, Gemini, ModelInfo, Ollama, OpenAI
from .parser import ToolCallParser
from .pricing import PRICING, calculate_cost, calculate_embedding_cost, get_model_pricing
from .prompt import PromptBuilder
from .providers.anthropic_provider import AnthropicProvider
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.stubs import LocalProvider
from .tools import Tool, ToolParameter, ToolRegistry, tool
from .types import AgentResult, Message, Role, ToolCall
from .usage import AgentUsage, UsageStats

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentAnalytics",
    "ToolMetrics",
    "ConversationMemory",
    "Message",
    "Role",
    "Tool",
    "ToolParameter",
    "ToolCall",
    "ToolCallParser",
    "PromptBuilder",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OllamaProvider",
    "LocalProvider",
    "ToolRegistry",
    "tool",
    # Exceptions
    "SelectoolsError",
    "ToolValidationError",
    "ToolExecutionError",
    "ProviderConfigurationError",
    "MemoryLimitExceededError",
    # Usage tracking
    "UsageStats",
    "AgentUsage",
    # Pricing
    "PRICING",
    "calculate_cost",
    "calculate_embedding_cost",
    "get_model_pricing",
    # Model Registry
    "models",
    "ModelInfo",
    "ALL_MODELS",
    "MODELS_BY_ID",
    "OpenAI",
    "Anthropic",
    "Gemini",
    "Ollama",
    "Cohere",
    # Caching
    "Cache",
    "CacheStats",
    "CacheKeyBuilder",
    "InMemoryCache",
    # Submodules (for lazy loading)
    "embeddings",
    "rag",
    "toolbox",
]
