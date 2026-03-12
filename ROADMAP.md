# Selectools Development Roadmap

> **Status Legend**
>
> - ✅ **Implemented** - Merged and available in latest release
> - 🔵 **In Progress** - Actively being worked on
> - 🟡 **Planned** - Scheduled for implementation
> - ⏸️ **Deferred** - Postponed to later release
> - ❌ **Cancelled** - No longer planned

---

## v0.15.0: Enterprise Reliability (In Progress)

Focus: Guardrails, security hardening, tool output screening, and audit compliance.

### Guardrails Engine

**Problem**: No systematic way to validate what goes into or comes out of the LLM. Users need content moderation, PII detection, topic restriction, and format enforcement.

**What it does**: A pluggable pipeline of validators that run before (input) and after (output) every LLM call. Each guardrail returns pass/fail/rewrite.

**API**:

```python
from selectools.guardrails import GuardrailsPipeline, TopicGuardrail, PIIGuardrail

guardrails = GuardrailsPipeline(
    input=[
        TopicGuardrail(deny=["politics", "religion"]),
        PIIGuardrail(action="redact"),
    ],
    output=[
        FormatGuardrail(require_json=True),
        ToxicityGuardrail(threshold=0.8),
    ],
)

agent = Agent(tools=[...], provider=provider, config=AgentConfig(guardrails=guardrails))
```

**Scope**:

- `Guardrail` protocol: `check(content) -> GuardrailResult(passed, rewritten_content, reason)`
- `GuardrailsPipeline`: ordered list of input and output guardrails
- Built-in guardrails: `TopicGuardrail`, `PIIGuardrail`, `ToxicityGuardrail`, `FormatGuardrail`, `LengthGuardrail`
- Custom guardrails: subclass `Guardrail` and implement `check()`
- Actions on failure: block (raise), rewrite (modify content), warn (log + continue)
- Integration with agent loop: input guardrails before provider call, output guardrails after

**Touches**: New `guardrails/` subpackage, `agent/core.py`, `AgentConfig`.

### Audit Logging

**Problem**: No built-in audit trail. Compliance, debugging, and cost analysis require knowing what tools were called, when, and whether they succeeded.

**What it does**: JSONL append-only audit log with privacy controls and optional daily rotation.

**Scope**:

- `AuditLogger` class with configurable privacy (hash inputs, log arg keys only)
- JSONL format with timestamps, tool name, duration, success/failure
- Daily file rotation
- Queryable via existing `AgentAnalytics` or standalone

**Touches**: New `audit.py`, hooks into `agent/core.py`.

### Tool Output Screening

**Problem**: Tools returning untrusted content (web scraping, email, file parsing) can contain prompt injection payloads.

**What it does**: Optional screening of tool outputs before feeding them back to the LLM. Pattern-based detection with configurable actions.

**Scope**:

- `@tool(screen_output=True)` flag
- Screening strategies: regex patterns, keyword blocklist
- Blocked outputs replaced with safe message
- Integrates with guardrails engine

**Touches**: `agent/core.py`, `Tool` class, new `security.py`.

### Coherence Checking

**Problem**: Prompt injection in tool outputs can cause the agent to call unrelated tools. User asks "summarize my emails" but injected content causes `send_email` instead.

**What it does**: Optional LLM-based check that verifies each tool call matches the user's original intent.

**Scope**:

- `AgentConfig(coherence_check=True)`
- Lightweight check using the agent's own provider
- Compare proposed tool call against original user intent
- Block incoherent calls with explanation to LLM

**Touches**: `agent/core.py`, `AgentConfig`.

| Feature                   | Status    | Impact | Effort |
| ------------------------- | --------- | ------ | ------ |
| **Guardrails Engine**     | 🔵 High   | High   | Large  |
| **Audit Logging**         | 🔵 Medium | Medium | Small  |
| **Tool Output Screening** | 🔵 Medium | Medium | Medium |
| **Coherence Checking**    | 🔵 Medium | Medium | Medium |

---

## v0.16.0: Memory & Persistence

Focus: Durable conversation state, cross-session knowledge, and advanced memory strategies.

### Persistent Conversation Sessions

**Problem**: `ConversationMemory` is in-memory only. Process restarts lose all history. Chat applications need sessions that survive restarts.

**What it does**: `SessionStore` protocol with pluggable backends. Sessions auto-persist after each turn with TTL-based expiry.

**API**:

```python
from selectools.memory import JsonFileSessionStore

store = JsonFileSessionStore(directory="./sessions")
agent = Agent(
    tools=[...], provider=provider,
    config=AgentConfig(session_store=store, session_id="user-123"),
)
result = agent.ask("What was my last question?")  # auto-persisted
```

**Scope**:

- `SessionStore` protocol: `save()`, `load()`, `list()`, `delete()`
- Three backends: JSON file, SQLite, Redis
- Auto-save after each turn
- TTL-based expiry
- Tool-pair-preserving trim on load

**Touches**: New `sessions.py`, `AgentConfig`, `agent/core.py`.

### Summarize-on-Trim

**Problem**: Old messages are silently dropped when history exceeds limits. Important early context is lost.

**What it does**: Before trimming, summarize the messages being removed and inject the summary as a system-level context message.

**API**:

```python
memory = ConversationMemory(
    max_messages=30,
    summarize_on_trim=True,
    summarize_provider=provider,
)
```

**Scope**:

- LLM-generated 2-3 sentence summary of trimmed messages
- Summary injected as system message at conversation start
- Configurable summary model (use a cheap model like Haiku)

**Touches**: `memory.py`, provider integration.

### Entity Memory

**Problem**: The agent can't track entities (people, orgs, projects) mentioned across turns. Each turn starts with no entity context.

**What it does**: Automatically extract named entities from conversation, maintain an entity registry, and inject relevant entity context into prompts.

**API**:

```python
from selectools.memory import EntityMemory

memory = EntityMemory(provider=provider)
agent = Agent(tools=[...], provider=provider, memory=memory)

agent.ask("I'm working with Alice from Acme Corp on Project Alpha")
agent.ask("What project am I working on?")
# Agent knows: Alice (person, Acme Corp), Acme Corp (org), Project Alpha (project)
```

**Scope**:

- LLM-based entity extraction after each turn
- Entity types: person, organization, project, location, date, custom
- Entity registry: name → type, attributes, last mentioned
- System prompt injection of relevant entities
- Configurable: extraction model, max entities, relevance window

**Touches**: New `entity_memory.py`, `PromptBuilder` integration.

### Knowledge Graph Memory

**Problem**: Entity memory tracks individual entities but not relationships between them. "Alice manages Project Alpha" is lost.

**What it does**: Build a graph of (subject, relation, object) triples from conversations. Query the graph to inject relevant relationship context into prompts.

**API**:

```python
from selectools.memory import KnowledgeGraphMemory

memory = KnowledgeGraphMemory(provider=provider, storage="sqlite")
agent = Agent(tools=[...], provider=provider, memory=memory)

agent.ask("Alice manages Project Alpha and reports to Bob")
# Graph: (Alice, manages, Project Alpha), (Alice, reports_to, Bob)

agent.ask("Who manages Project Alpha?")
# Relevant triples injected: (Alice, manages, Project Alpha)
```

**Scope**:

- LLM-based triple extraction
- Storage: in-memory dict (default), SQLite (persistent)
- Query: retrieve triples relevant to current query via keyword + embedding match
- System prompt injection of relevant triples
- Graph operations: add, query, merge, prune

**Touches**: New `knowledge_graph.py`, storage backend, `PromptBuilder`.

### Cross-Session Knowledge Memory

**Problem**: Even with persistent sessions, each session is isolated. There's no way for an agent to "remember" facts across conversations (e.g., user preferences, prior decisions).

**What it does**: A file-based or DB-backed knowledge memory with two layers: a daily log (append-only entries from the current day) and a long-term store (curated facts that persist indefinitely). A built-in `remember` tool lets the agent save facts explicitly. Relevant memories are auto-injected into the system prompt.

**API**:

```python
from selectools.memory import KnowledgeMemory

knowledge = KnowledgeMemory(
    directory="./workspace",
    recent_days=2,           # inject last 2 days into system prompt
    max_context_chars=5000,  # cap memory injection size
)

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(knowledge_memory=knowledge),
)
```

**Scope**:

- Daily log files (`memory/YYYY-MM-DD.md`) + persistent `MEMORY.md`
- Built-in `remember` tool: agent can save categorized facts
- System prompt auto-injection of recent and long-term memories
- Configurable retention and context window

**Touches**: New `knowledge.py` module, `PromptBuilder` integration, built-in tool in `toolbox/`.

| Feature                              | Priority  | Impact | Effort |
| ------------------------------------ | --------- | ------ | ------ |
| **Persistent Conversation Sessions** | 🟡 High   | High   | Medium |
| **Summarize-on-Trim**                | 🟡 Medium | Medium | Small  |
| **Cross-Session Knowledge Memory**   | 🟡 Medium | Medium | Medium |
| **Entity Memory**                    | 🟡 Medium | High   | Medium |
| **Knowledge Graph Memory**           | 🟡 Low    | High   | Large  |

---

## Implementation Order

```
v0.13.0  ✅ Structured Output + Safety Foundation (Complete)
         Tool-pair trimming → Structured output → Execution traces → Reasoning
         → Fallback providers → Batch → Tool policy → Human-in-the-loop

v0.14.0  ✅ AgentObserver Protocol + Production Hardening (Complete)
         AgentObserver (15 events) → LoggingObserver → OTel export
         → Model registry (145 models) → max_completion_tokens fix → 11 bug fixes

v0.14.1  ✅ Streaming & Provider Fixes (Complete)
         13 streaming bug fixes → 141 new tests → Unit tests for 6 untested modules

v0.15.0  🔵 Enterprise Reliability (In Progress)
         Guardrails engine → Audit logging → Tool output screening → Coherence checking

v0.16.0  Memory & Persistence
         Sessions → Summarize-on-trim → Knowledge memory → Entity memory → KG memory
```

---

## Backlog (Unscheduled)

| Feature                    | Notes                                                            |
| -------------------------- | ---------------------------------------------------------------- |
| Multi-Agent Graphs         | `AgentGraph` with conditional edges, cycle detection, handoffs   |
| Supervisor Agent           | Meta-agent for task decomposition and delegation                 |
| Shared State & Blackboard  | Thread-safe key-value store for agent graph nodes                |
| MCP Client                 | Discover and call tools from MCP-compliant servers               |
| MCP Server                 | Expose `@tool` functions as MCP-compliant server                 |
| Framework Integrations     | FastAPI `AgentRouter`, Flask `AgentBlueprint` adapters           |
| Tool Composition           | `@compose` decorator for chaining tools                          |
| Tool Marketplace/Registry  | Community tool sharing                                           |
| Universal Vision Support   | Unified vision API across providers                              |
| AWS Bedrock Provider       | VPC-native model access (Claude, Llama, Mistral)                 |
| Rate Limiting & Quotas     | Per-tool and per-user quotas                                     |
| Visual Agent Builder       | Web UI for agent design                                          |
| Enhanced Testing Framework | Snapshot testing, load tests                                     |
| Documentation Generation   | Auto-generate docs from tool definitions                         |
| Prompt Optimization        | Automatic prompt compression                                     |
| YAML Tool Configuration    | Declarative tool definitions without code                        |
| CRM & Business Tools       | HubSpot, Salesforce integrations                                 |
| Data Source Connectors     | SQL, vector DBs, cloud storage                                   |

---

## Release History

### v0.14.1 - Streaming & Provider Fixes

- ✅ **13 streaming bug fixes**: All providers' `stream()`/`astream()` now pass `tools` and yield `ToolCall` objects
- ✅ **Agent core fixes**: `_streaming_call`/`_astreaming_call` pass tools and don't stringify `ToolCall` objects
- ✅ **Ollama `_format_messages`**: Correct `TOOL` role mapping and `ASSISTANT` tool_calls inclusion
- ✅ **FallbackProvider `astream()`**: Error handling, failover, and circuit breaker support
- ✅ **141 new tests** (total: 1100): Regression tests, recording-provider tests, unit tests for 6 previously untested modules

### v0.14.0 - AgentObserver Protocol & Production Hardening

- ✅ **AgentObserver Protocol**: 15 lifecycle events with `run_id`/`call_id` correlation
- ✅ **LoggingObserver**: Structured JSON logs for ELK/Datadog
- ✅ **OTel Span Export**: `AgentTrace.to_otel_spans()` for OpenTelemetry
- ✅ **Model Registry Update**: 145 models with March 2026 pricing (GPT-5.4, Claude Sonnet 4.6, Gemini 3.1 Pro)
- ✅ **OpenAI `max_completion_tokens`**: Auto-detection for GPT-5.x, GPT-4.1, o-series models
- ✅ **11 bug fixes**: Structured output parser bypass, policy bypass in parallel execution, memory trim observer gap, infinite recursion in batch+fallback, async policy timeout, None content handling, and more

### v0.13.0 - Structured Output, Observability & Safety

- ✅ **Structured Output Parsers**: Pydantic / JSON Schema `response_format` on `run()` / `arun()` / `ask()` with auto-retry
- ✅ **Execution Traces**: `result.trace` with `TraceStep` timeline (`llm_call`, `tool_selection`, `tool_execution`, `error`)
- ✅ **Reasoning Visibility**: `result.reasoning` and `result.reasoning_history` extracted from LLM responses
- ✅ **Provider Fallback Chain**: `FallbackProvider` with circuit breaker and `on_fallback` callback
- ✅ **Batch Processing**: `agent.batch()` / `agent.abatch()` with `max_concurrency` and per-request error isolation
- ✅ **Tool-Pair-Aware Trimming**: `ConversationMemory` preserves tool_use/tool_result pairs during sliding window trim
- ✅ **Tool Policy Engine**: `ToolPolicy` with glob-based allow/review/deny rules and argument-level conditions
- ✅ **Human-in-the-Loop Approval**: `confirm_action` callback for `review` tools with `approval_timeout`

### v0.12.x - Hybrid Search, Reranking, Advanced Chunking & Dynamic Tools

- ✅ **BM25**: Pure-Python Okapi BM25 keyword search; configurable k1/b; stop word removal; zero dependencies
- ✅ **HybridSearcher**: Vector + BM25 fusion via RRF or weighted linear combination
- ✅ **HybridSearchTool**: Agent-ready `@tool` with source attribution and score thresholds
- ✅ **FusionMethod**: `RRF` (rank-based) and `WEIGHTED` (normalised score) strategies
- ✅ **Reranker ABC**: Protocol for cross-encoder reranking with `rerank(query, results, top_k)`
- ✅ **CohereReranker**: Cohere Rerank API v2 (`rerank-v3.5` default)
- ✅ **JinaReranker**: Jina AI Rerank API (`jina-reranker-v2-base-multilingual` default)
- ✅ **HybridSearcher integration**: Optional `reranker=` param for post-fusion re-scoring
- ✅ **SemanticChunker**: Embedding-based topic-boundary splitting; cosine similarity threshold
- ✅ **ContextualChunker**: LLM-generated context prepended to each chunk (Anthropic-style contextual retrieval)
- ✅ **ToolLoader**: Discover `@tool` functions from modules, files, and directories; hot-reload support
- ✅ **Agent dynamic tools**: `add_tool`, `add_tools`, `remove_tool`, `replace_tool` with prompt rebuild

### v0.12.0 - Response Caching

- ✅ **InMemoryCache**: Thread-safe LRU + TTL cache with `OrderedDict`; zero dependencies
- ✅ **RedisCache**: Distributed TTL cache for multi-process deployments (optional `redis` dep)
- ✅ **CacheKeyBuilder**: Deterministic SHA-256 keys from (model, prompt, messages, tools, temperature)
- ✅ **Agent Integration**: `AgentConfig(cache=...)` checks cache before every provider call

### v0.11.0 - Streaming & Parallel Execution

- ✅ **E2E Streaming**: Native tool streaming via `Agent.astream` with `Union[str, ToolCall]` provider protocol
- ✅ **Parallel Tool Execution**: `asyncio.gather` for async, `ThreadPoolExecutor` for sync; enabled by default
- ✅ **Full Type Safety**: 0 mypy errors across 80+ source and test files

### v0.10.0 - Critical Architecture

- ✅ **Native Function Calling**: OpenAI, Anthropic, and Gemini native tool APIs
- ✅ **Context Propagation**: `contextvars.copy_context()` for async tool execution
- ✅ **Routing Mode**: `AgentConfig(routing_only=True)` for classification without execution

### v0.9.0 - Core Capabilities & Reliability

- ✅ **Custom System Prompt**: `AgentConfig(system_prompt=...)` for domain instructions
- ✅ **Structured AgentResult**: `run()` returns `AgentResult` with tool calls, args, and iterations
- ✅ **Reusable Agent Instances**: `Agent.reset()` clears history/memory for clean reuse

### v0.8.0 - Embeddings & RAG

- ✅ **Full RAG Stack**: VectorStore (Memory/SQLite/Chroma), Embeddings (OpenAI/Gemini), Document Loaders
- ✅ **RAG Tools**: `RAGTool` and `SemanticSearchTool` for knowledge base queries

### v0.6.0 - High-Impact Features

- ✅ **Observability Hooks**: `on_agent_start`, `on_tool_end` lifecycle events
- ✅ **Streaming Tools**: Generators yield results progressively

### v0.5.0 - Production Readiness

- ✅ **Cost Tracking**: Token counting and USD estimation
- ✅ **Better Errors**: PyTorch-style error messages with suggestions
