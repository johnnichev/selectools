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

| Feature                         | Status     | Impact | Description                                                    |
| ------------------------------- | ---------- | ------ | -------------------------------------------------------------- |
| **Native Function Calling**     | ✅ v0.10.0 | High   | Use OpenAI/Anthropic/Gemini native tool APIs instead of regex parsing |
| **Context Propagation (Async)** | ✅ v0.10.0 | High   | `contextvars.copy_context()` for tracing/auth in async tools   |
| **Select-Only / Routing Mode**  | ✅ v0.10.0 | High   | Run agent for classification/routing without executing the selected tool |

---

## v0.11.0: Streaming & Performance (Complete)

Focus: E2E streaming, parallel execution, and type safety.

| Feature                         | Status     | Notes                                                                  |
| ------------------------------- | ---------- | ---------------------------------------------------------------------- |
| **E2E Streaming Response**      | ✅ v0.11.0 | Native tool streaming via `Agent.astream` with `Union[str, ToolCall]` provider protocol |
| **Parallel Tool Execution**     | ✅ v0.11.0 | `asyncio.gather` for async, `ThreadPoolExecutor` for sync; `AgentConfig(parallel_tool_execution=True)` |
| **Full Type Safety**            | ✅ v0.11.0 | 0 mypy errors across all source and test files; `disallow_untyped_defs` enforced |

---

## v0.12.0: Advanced Data & RAG (Planned)

Focus: Advanced RAG capabilities and caching.

| Feature                      | Priority  | Notes                                                                  |
| ---------------------------- | --------- | ---------------------------------------------------------------------- |
| **Hybrid Search**            | 🟠 High   | Vector + BM25 keyword search                                           |
| **Reranking Models**         | 🟡 Medium | Cohere/Jina rerankers for better search relevance                      |
| **Advanced Chunking**        | 🟡 Medium | Agentic/Contextual document chunking                                   |
| **Response Caching**         | 🟡 Medium | Built-in TTL/LRU caching for identical queries (Redis + In-Memory)     |
| **Dynamic Tool Loading**     | 🟡 Medium | Hot-reload tools without restart                                       |

---

## v1.0.0: Enterprise Reliability (Future)

Focus: Stability, observability, and advanced orchestration.

| Feature                     | Priority  | Impact | Description                                                          |
| --------------------------- | --------- | ------ | -------------------------------------------------------------------- |
| **Retry Policies**          | 🟡 Medium | Medium | Declarative retries (exponential backoff) on tool definitions        |
| **Provider Fallback Chain** | 🟡 Medium | High   | Auto-switch providers on failure (OpenAI → Anthropic → Local)        |
| **Tool Middleware**         | 🟡 Medium | Medium | Cross-cutting concerns (auth, rate limiting) via middleware pipeline |
| **Azure OpenAI Provider**   | 🟡 Medium | Medium | Enterprise-grade Azure deployment integration                        |
| **Circuit Breakers**        | 🟡 Medium | High   | Stop cascading failures when downstream services are down            |

---

## Future Enhancements & Detailed Backlog

### High-Impact Complex Features

| Feature                 | Status         | Notes                         |
| ----------------------- | -------------- | ----------------------------- |
| Parallel Tool Execution | ✅ Implemented | `asyncio.gather` / `ThreadPoolExecutor` |
| Tool Composition        | 🟡 Planned     | `@compose` decorator          |

### Context Management

| Feature                              | Status     | Notes                           |
| ------------------------------------ | ---------- | ------------------------------- |
| Automatic Conversation Summarization | 🟡 Planned | Handle long conversations       |
| Sliding Window with Smart Retention  | 🟡 Planned | Keep important context          |
| Multi-Turn Memory System             | 🟡 Planned | Persistent cross-session memory |

### Tool Capabilities

| Feature                   | Status     | Notes                     |
| ------------------------- | ---------- | ------------------------- |
| Dynamic Tool Loading      | 🟡 Planned | Hot-reload tools          |
| Tool Usage Analytics      | ✅ v0.6.0  | Track performance metrics |
| Tool Marketplace/Registry | 🟡 Planned | Community tool sharing    |

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
| Security Hardening        | 🟡 Planned | Sandboxing, audit logging             |

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

| Feature             | Status     | Notes                              |
| ------------------- | ---------- | ---------------------------------- |
| Caching Layer       | 🟡 Planned | LRU, semantic, distributed caching |
| Batch Processing    | 🟡 Planned | Efficient multi-request handling   |
| Prompt Optimization | 🟡 Planned | Automatic prompt compression       |

---

## Release History

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
