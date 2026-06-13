"""Public exports for the selectools package."""

__version__ = "0.26.0"

# Import submodules (lazy loading for optional dependencies)
from . import embeddings, evals, guardrails, models, observe, patterns, rag, toolbox
from .agent import Agent, AgentConfig, PlanningConfig
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
    KnowledgeBackend,
    KnowledgeEntry,
    KnowledgeMemory,
    KnowledgeStore,
    PreSaveHook,
    SQLiteKnowledgeStore,
)
from .knowledge_backends import (
    RedisKnowledgeBackend,
    SupabaseKnowledgeBackend,
)
from .knowledge_graph import (
    InMemoryTripleStore,
    KnowledgeGraphMemory,
    SQLiteTripleStore,
    Triple,
    TripleStore,
)
from .knowledge_sanitizers import (
    dedupe_against,
    defang_delimiters,
    strip_surrogates,
)
from .loop_detection import (
    BaseDetector,
    LoopDetectedError,
    LoopDetection,
    LoopDetector,
    LoopPolicy,
    PingPongDetector,
    RepeatDetector,
    StallDetector,
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
from .pending import (
    ChannelAgent,
    ConfirmOutcome,
    ConfirmParser,
    InMemoryPendingStore,
    PendingAction,
    PendingActionExistsError,
    PendingActionStore,
    PendingConfirmation,
    RedisPendingStore,
    RegexConfirmParser,
    compute_args_digest,
    stash_pending,
)
from .pipeline import Pipeline, Step, StepResult, branch, cache_step, parallel, retry, step
from .policy import ApprovalRequest, PolicyDecision, PolicyResult, ToolPolicy
from .pricing import (
    PRICING,
    calculate_cost,
    calculate_cost_with_cached_input,
    calculate_embedding_cost,
    get_model_pricing,
)
from .prompt import REASONING_STRATEGIES, PromptBuilder
from .providers.anthropic_provider import AnthropicProvider
from .providers.azure_openai_provider import AzureOpenAIProvider
from .providers.fallback import FallbackProvider
from .providers.gemini_provider import GeminiProvider
from .providers.litellm_provider import LiteLLMProvider
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.router import RouterConfig, RouterProvider
from .providers.stubs import LocalProvider
from .results import Ambiguous, Artifact, NotFound, ToolResult, emit_artifact
from .sessions import (
    JsonFileSessionStore,
    RedisSessionStore,
    SessionMetadata,
    SessionSearchResult,
    SessionStore,
    SQLiteSessionStore,
    SupabaseSessionStore,
)
from .stability import beta, deprecated, stable
from .structured import ResponseFormat
from .token_estimation import TokenEstimate, estimate_run_tokens, estimate_tokens
from .tools import Tool, ToolParameter, ToolRegistry, tool
from .trace import AgentTrace, StepType, TraceStep, trace_to_html, trace_to_json
from .types import AgentResult, ContentPart, Message, Role, ToolCall, image_message, text_content
from .unified_memory import (
    DEFAULT_IMPORTANCE_RULES,
    Episode,
    EpisodicMemory,
    ImportanceRule,
    InMemoryKnowledgeStore,
    RecallResult,
    UnifiedMemory,
    score_importance,
)
from .usage import AgentUsage, UsageStats

__all__ = [
    "Agent",
    "AgentConfig",
    "PlanningConfig",
    "AgentResult",
    "AgentAnalytics",
    "ToolMetrics",
    "ConversationMemory",
    "Message",
    "ContentPart",
    "image_message",
    "text_content",
    "Role",
    "Tool",
    "ToolParameter",
    "ToolCall",
    "ToolCallParser",
    "PromptBuilder",
    "REASONING_STRATEGIES",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OllamaProvider",
    "LiteLLMProvider",
    "RouterProvider",
    "RouterConfig",
    "LocalProvider",
    "FallbackProvider",
    "ToolRegistry",
    "tool",
    # Pipeline composition (public since 0.19.3, @beta)
    "Pipeline",
    "Step",
    "StepResult",
    "compose",
    "step",
    "parallel",
    "branch",
    "retry",
    "cache_step",
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
    "calculate_cost_with_cached_input",
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
    "ApprovalRequest",
    "PolicyDecision",
    "PolicyResult",
    # Structured output
    "ResponseFormat",
    # Typed tool results + artifact side-channel (issue #59)
    "ToolResult",
    "Ambiguous",
    "NotFound",
    "Artifact",
    "emit_artifact",
    # Deferred confirmation flow (issue #58)
    "PendingAction",
    "PendingActionStore",
    "PendingActionExistsError",
    "InMemoryPendingStore",
    "RedisPendingStore",
    "ConfirmOutcome",
    "ConfirmParser",
    "RegexConfirmParser",
    "PendingConfirmation",
    "ChannelAgent",
    "stash_pending",
    "compute_args_digest",
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
    "trace_to_json",
    # Loop detection
    "LoopDetector",
    "LoopDetection",
    "LoopDetectedError",
    "LoopPolicy",
    "BaseDetector",
    "RepeatDetector",
    "StallDetector",
    "PingPongDetector",
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
    "SessionSearchResult",
    "JsonFileSessionStore",
    "SQLiteSessionStore",
    "RedisSessionStore",
    "SupabaseSessionStore",
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
    # Knowledge backends (blob persistence between deploys; lazy optional deps)
    "KnowledgeBackend",
    "SupabaseKnowledgeBackend",
    "RedisKnowledgeBackend",
    # Knowledge pre-save sanitizers (beta)
    "PreSaveHook",
    "defang_delimiters",
    "strip_surrogates",
    "dedupe_against",
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
    # Unified Memory (tiered memory with auto-promotion)
    "UnifiedMemory",
    "EpisodicMemory",
    "Episode",
    "ImportanceRule",
    "InMemoryKnowledgeStore",
    "RecallResult",
    "DEFAULT_IMPORTANCE_RULES",
    "score_importance",
    # Submodules (for lazy loading)
    "embeddings",
    "observe",
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
