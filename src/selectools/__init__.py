"""Public exports for the selectools package."""

__version__ = "0.17.3"

# Import submodules (lazy loading for optional dependencies)
from . import embeddings, evals, guardrails, models, rag, toolbox
from .agent import Agent, AgentConfig
from .analytics import AgentAnalytics, ToolMetrics
from .audit import AuditLogger, PrivacyLevel
from .cache import Cache, CacheKeyBuilder, CacheStats, InMemoryCache
from .cancellation import CancellationToken
from .coherence import CoherenceResult
from .entity_memory import Entity, EntityMemory
from .evals import EvalReport, EvalSuite, TestCase
from .exceptions import (
    BudgetExceededError,
    CancellationError,
    GraphExecutionError,
    MemoryLimitExceededError,
    ProviderConfigurationError,
    SelectoolsError,
    ToolExecutionError,
    ToolValidationError,
)
from .guardrails import (
    FormatGuardrail,
    Guardrail,
    GuardrailAction,
    GuardrailError,
    GuardrailResult,
    GuardrailsPipeline,
    LengthGuardrail,
    PIIGuardrail,
    TopicGuardrail,
    ToxicityGuardrail,
)
from .knowledge import KnowledgeMemory
from .knowledge_graph import (
    InMemoryTripleStore,
    KnowledgeGraphMemory,
    SQLiteTripleStore,
    Triple,
    TripleStore,
)
from .memory import ConversationMemory
from .models import (
    ALL_MODELS,
    MODELS_BY_ID,
    Anthropic,
    Cohere,
    Gemini,
    ModelInfo,
    ModelType,
    Ollama,
    OpenAI,
)
from .observer import AgentObserver, AsyncAgentObserver, LoggingObserver, SimpleStepObserver
from .parser import ToolCallParser
from .policy import PolicyDecision, PolicyResult, ToolPolicy
from .pricing import PRICING, calculate_cost, calculate_embedding_cost, get_model_pricing
from .prompt import PromptBuilder
from .providers.anthropic_provider import AnthropicProvider
from .providers.fallback import FallbackProvider
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.stubs import LocalProvider
from .sessions import (
    JsonFileSessionStore,
    RedisSessionStore,
    SessionMetadata,
    SessionStore,
    SQLiteSessionStore,
)
from .structured import ResponseFormat
from .tools import Tool, ToolParameter, ToolRegistry, tool
from .trace import AgentTrace, StepType, TraceStep
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
    "FallbackProvider",
    "ToolRegistry",
    "tool",
    # Cancellation
    "CancellationToken",
    # Exceptions
    "SelectoolsError",
    "ToolValidationError",
    "ToolExecutionError",
    "ProviderConfigurationError",
    "MemoryLimitExceededError",
    "GraphExecutionError",
    "BudgetExceededError",
    "CancellationError",
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
    "ModelType",
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
    # Tool policy
    "ToolPolicy",
    "PolicyDecision",
    "PolicyResult",
    # Structured output
    "ResponseFormat",
    # Observability
    "AgentObserver",
    "AsyncAgentObserver",
    "LoggingObserver",
    "SimpleStepObserver",
    "AgentTrace",
    "StepType",
    "TraceStep",
    # Guardrails
    "guardrails",
    "Guardrail",
    "GuardrailAction",
    "GuardrailError",
    "GuardrailResult",
    "GuardrailsPipeline",
    "FormatGuardrail",
    "LengthGuardrail",
    "PIIGuardrail",
    "TopicGuardrail",
    "ToxicityGuardrail",
    # Audit
    "AuditLogger",
    "PrivacyLevel",
    # Coherence
    "CoherenceResult",
    # Sessions
    "SessionStore",
    "SessionMetadata",
    "JsonFileSessionStore",
    "SQLiteSessionStore",
    "RedisSessionStore",
    # Entity Memory
    "Entity",
    "EntityMemory",
    # Knowledge Memory
    "KnowledgeMemory",
    # Knowledge Graph
    "Triple",
    "TripleStore",
    "InMemoryTripleStore",
    "SQLiteTripleStore",
    "KnowledgeGraphMemory",
    # Submodules (for lazy loading)
    "embeddings",
    "rag",
    "toolbox",
]
