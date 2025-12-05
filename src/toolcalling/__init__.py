"""Public exports for the toolcalling package."""

from .agent import Agent, AgentConfig
from .types import Message, Role, ToolCall
from .tools import Tool, ToolParameter
from .parser import ToolCallParser
from .prompt import PromptBuilder
from .providers.openai_provider import OpenAIProvider
from .providers.stubs import AnthropicProvider, GeminiProvider, LocalProvider

__all__ = [
    "Agent",
    "AgentConfig",
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
    "LocalProvider",
]
