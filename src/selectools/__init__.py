"""Public exports for the selectools package."""

__version__ = "0.20.0"

# Import submodules (lazy loading for optional dependencies)
from . import embeddings, evals, guardrails, models, patterns, rag, toolbox
from .agent import Agent, AgentConfig
from .agent.config_groups import (
    BudgetConfig,
    CoherenceConfig,
    CompressConfig,
    GuardrailsConfig,
    MemoryConfig,
    RetryConfig,
    SessionConfig,
    SummarizeConfig,
    ToolConfig,
    TraceConfig,
)
from .analytics import AgentAnalytics, ToolMetrics
from .audit import AuditLogger, PrivacyLevel
from .cache import Cache, CacheKeyBuilder, CacheStats, InMemoryCache
from .cache_semantic import SemanticCache
from .cancellation import CancellationToken
from .coherence import CoherenceResult
from .compose import compose
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
from .knowledge import (
    FileKnowledgeStore,
    KnowledgeEntry,
    KnowledgeMemory,
    KnowledgeStore,
    SQLiteKnowledgeStore,
)
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
from .orchestration import (
    STATE_KEY_LAST_OUTPUT,
    AgentGraph,
    CheckpointStore,
    ContextMode,
    ErrorPolicy,
    FileCheckpointStore,
    GraphEvent,
    GraphEventType,
    GraphNode,
    GraphResult,
    GraphState,
    InMemoryCheckpointStore,
    InterruptRequest,
    MergePolicy,
    ModelSplit,
    ParallelGroupNode,
    Scatter,
    SQLiteCheckpointStore,
    SubgraphNode,
    SupervisorAgent,
    SupervisorStrategy,
)
from .parser import ToolCallParser
from .patterns import (
    DebateAgent,
    DebateResult,
    DebateRound,
    PlanAndExecuteAgent,
    PlanStep,
    ReflectionRound,
    ReflectiveAgent,
    ReflectiveResult,
    Subtask,
    TeamLeadAgent,
    TeamLeadResult,
)
from .pipeline import Pipeline, Step, StepResult, branch, cache_step, parallel, retry, step
from .policy import PolicyDecision, PolicyResult, ToolPolicy
from .pricing import PRICING, calculate_cost, calculate_embedding_cost, get_model_pricing
from .prompt import REASONING_STRATEGIES, PromptBuilder
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
from .stability import beta, deprecated, stable
from .structured import ResponseFormat
from .token_estimation import TokenEstimate, estimate_run_tokens, estimate_tokens
from .tools import Tool, ToolParameter, ToolRegistry, tool
from .trace import AgentTrace, StepType, TraceStep, trace_to_html
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
    "REASONING_STRATEGIES",
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
    "SemanticCache",
    # Tool policy
    "ToolPolicy",
    "PolicyDecision",
    "PolicyResult",
    # Structured output
    "ResponseFormat",
    # Stability markers
    "stable",
    "beta",
    "deprecated",
    # Observability
    "AgentObserver",
    "AsyncAgentObserver",
    "LoggingObserver",
    "SimpleStepObserver",
    "AgentTrace",
    "StepType",
    "TraceStep",
    "trace_to_html",
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
    "KnowledgeEntry",
    "KnowledgeStore",
    "FileKnowledgeStore",
    "SQLiteKnowledgeStore",
    # Knowledge stores (optional deps: redis, supabase)
    # from selectools.knowledge_store_redis import RedisKnowledgeStore
    # from selectools.knowledge_store_supabase import SupabaseKnowledgeStore
    # Token estimation
    "TokenEstimate",
    "estimate_tokens",
    "estimate_run_tokens",
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
    # Orchestration
    "AgentGraph",
    "GraphResult",
    "ErrorPolicy",
    "GraphState",
    "GraphEvent",
    "GraphEventType",
    "MergePolicy",
    "ContextMode",
    "InterruptRequest",
    "Scatter",
    "STATE_KEY_LAST_OUTPUT",
    "GraphNode",
    "ParallelGroupNode",
    "SubgraphNode",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "FileCheckpointStore",
    "SQLiteCheckpointStore",
    "SupervisorAgent",
    "SupervisorStrategy",
    "ModelSplit",
    # Patterns
    "patterns",
    "PlanAndExecuteAgent",
    "PlanStep",
    "ReflectiveAgent",
    "ReflectionRound",
    "ReflectiveResult",
    "DebateAgent",
    "DebateRound",
    "DebateResult",
    "TeamLeadAgent",
    "Subtask",
    "TeamLeadResult",
]
