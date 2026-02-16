# Selectools Development Roadmap

> **Status Legend**
>
> - ✅ **Implemented** - Merged and available in latest release
> - 🔵 **In Progress** - Actively being worked on
> - 🟡 **Planned** - Scheduled for implementation
> - ⏸️ **Deferred** - Postponed to later release
> - ❌ **Cancelled** - No longer planned

---

## v0.9.0: Core Capabilities & Reliability (Available Now)

Recent major improvements focusing on agent control and reliability (Agent v0.9) and high-impact RAG features (Embeddings v0.9).

| Feature                        | Status    | Notes                                                                  |
| ------------------------------ | --------- | ---------------------------------------------------------------------- |
| **Custom System Prompt**       | ✅ v0.9.0 | Inject domain instructions via `AgentConfig(system_prompt=...)`        |
| **Structured Decision Result** | ✅ v0.9.0 | `run()` returns `AgentResult` with tool calls, args, and usage stats   |
| **Reusable Agent Instances**   | ✅ v0.9.0 | `Agent.reset()` clears history/memory for clean reuse between requests |
| **Embeddings & RAG**           | ✅ v0.8.0 | Vector stores, document loaders, semantic search, RAG tools            |

---

## v0.10.0: Critical Architecture (Complete)

Focus: Fixing architectural limitations and enabling production scaling.

| Feature                         | Status     | Impact | Description                                                              |
| ------------------------------- | ---------- | ------ | ------------------------------------------------------------------------ |
| **Native Function Calling**     | ✅ v0.10.0 | High   | Use OpenAI/Anthropic/Gemini native tool APIs instead of regex parsing    |
| **Context Propagation (Async)** | ✅ v0.10.0 | High   | `contextvars.copy_context()` for tracing/auth in async tools             |
| **Select-Only / Routing Mode**  | ✅ v0.10.0 | High   | Run agent for classification/routing without executing the selected tool |

---

## v0.11.0: Streaming & Performance (Complete)

Focus: E2E streaming, parallel execution, and type safety.

| Feature                     | Status     | Notes                                                                                                  |
| --------------------------- | ---------- | ------------------------------------------------------------------------------------------------------ |
| **E2E Streaming Response**  | ✅ v0.11.0 | Native tool streaming via `Agent.astream` with `Union[str, ToolCall]` provider protocol                |
| **Parallel Tool Execution** | ✅ v0.11.0 | `asyncio.gather` for async, `ThreadPoolExecutor` for sync; `AgentConfig(parallel_tool_execution=True)` |
| **Full Type Safety**        | ✅ v0.11.0 | 0 mypy errors across all source and test files; `disallow_untyped_defs` enforced                       |

---

## v0.12.0: Caching & Data (Complete)

Focus: Response caching and advanced RAG capabilities.

| Feature                  | Status     | Notes                                                                      |
| ------------------------ | ---------- | -------------------------------------------------------------------------- |
| **Response Caching**     | ✅ v0.12.0 | `InMemoryCache` (LRU+TTL) and `RedisCache`; `AgentConfig(cache=...)`       |
| **Hybrid Search**        | ✅ v0.12.x | `BM25` + `HybridSearcher` with RRF/weighted fusion; `HybridSearchTool`     |
| **Reranking Models**     | ✅ v0.12.x | `CohereReranker` + `JinaReranker`; `HybridSearcher(reranker=...)`          |
| **Advanced Chunking**    | ✅ v0.12.x | `SemanticChunker` (embedding similarity) + `ContextualChunker` (LLM context) |
| **Dynamic Tool Loading** | ✅ v0.12.x | `ToolLoader` + `Agent.add_tool/remove_tool/replace_tool`; hot-reload       |

---

## v0.13.0: Safety & Agent Control (Next)

Focus: Tool execution safety, policy enforcement, and human oversight.
See [FEATURE_PROPOSALS.md](./FEATURE_PROPOSALS.md) for detailed designs.

| Feature                               | Priority  | Impact | Description                                                                     |
| ------------------------------------- | --------- | ------ | ------------------------------------------------------------------------------- |
| **Tool-Pair-Aware Trimming**          | 🟡 High   | High   | Never split `tool_use`/`tool_result` pairs during conversation trimming         |
| **Tool Policy Engine**                | 🟡 High   | High   | Declarative allow/review/deny rules with glob patterns and arg-level conditions |
| **Human-in-the-Loop Approval**        | 🟡 High   | High   | Confirmation callback for `review`-flagged tools; async support + timeout       |

---

## v0.14.0: Memory & Persistence

Focus: Durable conversation state and cross-session knowledge.
See [FEATURE_PROPOSALS.md](./FEATURE_PROPOSALS.md) for detailed designs.

| Feature                            | Priority  | Impact | Description                                                                  |
| ---------------------------------- | --------- | ------ | ---------------------------------------------------------------------------- |
| **Persistent Conversation Sessions** | 🟡 High | High   | `SessionStore` protocol with JSON file, SQLite, and Redis backends; auto-save + TTL |
| **Summarize-on-Trim**               | 🟡 Medium | Medium | LLM-generated summary replaces trimmed messages instead of silent drop       |
| **Cross-Session Knowledge Memory**   | 🟡 Medium | Medium | Daily log + long-term `MEMORY.md`; built-in `remember` tool; system prompt injection |

---

## v1.0.0: Enterprise Reliability (Future)

Focus: Stability, observability, security hardening, and advanced orchestration.

| Feature                     | Priority  | Impact | Description                                                                  |
| --------------------------- | --------- | ------ | ---------------------------------------------------------------------------- |
| **Retry Policies**          | 🟡 Medium | Medium | Declarative retries (exponential backoff) on tool definitions                |
| **Provider Fallback Chain** | 🟡 Medium | High   | Auto-switch providers on failure (OpenAI → Anthropic → Local)                |
| **Tool Middleware**         | 🟡 Medium | Medium | Cross-cutting concerns (auth, rate limiting) via middleware pipeline          |
| **Circuit Breakers**        | 🟡 Medium | High   | Stop cascading failures when downstream services are down                    |
| **Audit Logging**           | 🟡 Medium | Medium | JSONL append-only log with privacy controls (hashed inputs, arg keys only)   |
| **Tool Output Screening**   | 🟡 Medium | Medium | Detect prompt injection in tool results before feeding back to LLM           |
| **Coherence Checking**      | 🟡 Medium | Medium | Verify tool calls match user's original intent to prevent injection hijacking |

---

## Future Enhancements & Detailed Backlog

### High-Impact Complex Features

| Feature                 | Status         | Notes                                   |
| ----------------------- | -------------- | --------------------------------------- |
| Parallel Tool Execution | ✅ Implemented | `asyncio.gather` / `ThreadPoolExecutor` |
| Tool Composition        | 🟡 Planned     | `@compose` decorator                    |

### Tool Capabilities

| Feature                   | Status         | Notes                                                       |
| ------------------------- | -------------- | ----------------------------------------------------------- |
| Dynamic Tool Loading      | ✅ Implemented | `ToolLoader` + `Agent.add_tool/remove_tool/replace_tool`    |
| Tool Usage Analytics      | ✅ v0.6.0      | Track performance metrics                                   |
| Tool Marketplace/Registry | 🟡 Planned     | Community tool sharing                                      |

### Provider Enhancements

| Feature                  | Status         | Notes                                                   |
| ------------------------ | -------------- | ------------------------------------------------------- |
| Universal Vision Support | 🟡 Planned     | Unified vision API                                      |
| Provider Auto-Selection  | 🟡 Planned     | Automatic fallback chains                               |
| Streaming Improvements   | ✅ Implemented | Native tool streaming via `astream`                     |
| Local Model Support      | ✅ v0.6.0      | Ollama, LM Studio integration                           |
| AWS Bedrock Provider     | 🟡 Planned     | Secure VPC-native model access (Claude, Llama, Mistral) |

### Production Reliability

| Feature                   | Status     | Notes                                 |
| ------------------------- | ---------- | ------------------------------------- |
| Advanced Error Recovery   | 🟡 Planned | Circuit breaker, graceful degradation |
| Observability & Debugging | 🟡 Planned | OpenTelemetry, execution replay       |
| Rate Limiting & Quotas    | 🟡 Planned | Per-tool and user quotas              |

### Developer Experience

| Feature                    | Status         | Notes                               |
| -------------------------- | -------------- | ----------------------------------- |
| Interactive Debug Mode     | 🟡 Planned     | Step-through agent execution        |
| Visual Agent Builder       | 🟡 Planned     | Web UI for agent design             |
| Enhanced Testing Framework | 🟡 Planned     | Snapshot testing, load tests        |
| Documentation Generation   | 🟡 Planned     | Auto-generate from tool definitions |
| Type Safety Improvements   | ✅ Implemented | Full mypy coverage, all annotations |

### Ecosystem Integration

| Feature                | Status     | Notes                             |
| ---------------------- | ---------- | --------------------------------- |
| Framework Integrations | 🟡 Planned | FastAPI, Flask, LangChain adapter |
| CRM & Business Tools   | 🟡 Planned | HubSpot, Salesforce, etc          |
| Data Source Connectors | 🟡 Planned | SQL, vector DBs, cloud storage    |

### Performance Optimizations (Backend)

| Feature             | Status         | Notes                                    |
| ------------------- | -------------- | ---------------------------------------- |
| Caching Layer       | ✅ Implemented | `InMemoryCache` (LRU+TTL) + `RedisCache` |
| Batch Processing    | 🟡 Planned     | Efficient multi-request handling         |
| Prompt Optimization | 🟡 Planned     | Automatic prompt compression             |

---

## Release History

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
