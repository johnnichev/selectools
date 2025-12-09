"""Public exports for the selectools package."""

# Toolbox is imported separately to avoid pulling in optional dependencies
from . import toolbox
from .agent import Agent, AgentConfig
from .analytics import AgentAnalytics, ToolMetrics
from .exceptions import (
    MemoryLimitExceededError,
    ProviderConfigurationError,
    SelectoolsError,
    ToolExecutionError,
    ToolValidationError,
)
from .memory import ConversationMemory
from .parser import ToolCallParser
from .pricing import PRICING, calculate_cost, get_model_pricing
from .prompt import PromptBuilder
from .providers.anthropic_provider import AnthropicProvider
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.stubs import LocalProvider
from .tools import Tool, ToolParameter, ToolRegistry, tool
from .types import Message, Role, ToolCall
from .usage import AgentUsage, UsageStats

__all__ = [
    "Agent",
    "AgentConfig",
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
    "get_model_pricing",
]
