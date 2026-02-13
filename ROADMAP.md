# Selectools Development Roadmap

This document tracks all planned features and improvements. It consolidates requests from the Traffic Cop and Smart Router projects into a unified release plan.

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

## v0.10.0: Critical Architecture (Planned)

Focus: Fixing architectural limitations and enabling production scaling.

| Feature                         | Priority    | Impact | Description                                                              |
| ------------------------------- | ----------- | ------ | ------------------------------------------------------------------------ |
| **Native Function Calling**     | 🔴 Critical | High   | Use OpenAI/Anthropic native tool APIs instead of regex parsing           |
| **Context Propagation (Async)** | 🔴 Critical | High   | `contextvars.copy_context()` for tracing/auth in async tools             |
| **AWS Bedrock Provider**        | 🟠 High     | High   | Secure VPC-native model access (Claude, Llama, Mistral) via boto3        |
| **Select-Only / Routing Mode**  | 🟠 High     | High   | Run agent for classification/routing without executing the selected tool |

---

## v0.11.0: Advanced Data & Performance (Planned)

Focus: Advanced RAG capabilities, streaming, and caching. (Original v0.9.0 plan)

| Feature                      | Priority  | Notes                                                                  |
| ---------------------------- | --------- | ---------------------------------------------------------------------- |
| **Hybrid Search**            | 🟠 High   | Vector + BM25 keyword search                                           |
| **Reranking Models**         | 🟡 Medium | Cohere/Jina rerankers for better search relevance                      |
| **Advanced Chunking**        | 🟡 Medium | Agentic/Contextual document chunking                                   |
| **Streaming Response (E2E)** | 🟠 High   | Token-level streaming from LLM → Tool → Client (SSE/WebSocket support) |
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

| Feature                 | Status     | Notes                         |
| ----------------------- | ---------- | ----------------------------- |
| Parallel Tool Execution | 🟡 Planned | Auto-detect independent tools |
| Tool Composition        | 🟡 Planned | `@compose` decorator          |

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
| Tool Usage Analytics      | 🟡 Planned | Track performance metrics |
| Tool Marketplace/Registry | 🟡 Planned | Community tool sharing    |

### Provider Enhancements

| Feature                  | Status     | Notes                         |
| ------------------------ | ---------- | ----------------------------- |
| Universal Vision Support | 🟡 Planned | Unified vision API            |
| Provider Auto-Selection  | 🟡 Planned | Automatic fallback chains     |
| Streaming Improvements   | 🟡 Planned | SSE, WebSocket support        |
| Local Model Support      | 🟡 Planned | Ollama, LM Studio integration |

### Production Reliability

| Feature                   | Status     | Notes                                 |
| ------------------------- | ---------- | ------------------------------------- |
| Advanced Error Recovery   | 🟡 Planned | Circuit breaker, graceful degradation |
| Observability & Debugging | 🟡 Planned | OpenTelemetry, execution replay       |
| Rate Limiting & Quotas    | 🟡 Planned | Per-tool and user quotas              |
| Security Hardening        | 🟡 Planned | Sandboxing, audit logging             |

### Developer Experience

| Feature                    | Status     | Notes                               |
| -------------------------- | ---------- | ----------------------------------- |
| Interactive Debug Mode     | 🟡 Planned | Step-through agent execution        |
| Visual Agent Builder       | 🟡 Planned | Web UI for agent design             |
| Enhanced Testing Framework | 🟡 Planned | Snapshot testing, load tests        |
| Documentation Generation   | 🟡 Planned | Auto-generate from tool definitions |
| Type Safety Improvements   | 🟡 Planned | Better type inference               |

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

### v0.8.0 - Embeddings & RAG

- ✅ **Full RAG Stack**: VectorStore (Memory/SQLite/Chroma), Embeddings (OpenAI/Gemini), Document Loaders
- ✅ **RAG Tools**: `RAGTool` and `SemanticSearchTool` for knowledge base queries

### v0.6.0 - High-Impact Features

- ✅ **Observability Hooks**: `on_agent_start`, `on_tool_end` lifecycle events
- ✅ **Streaming Tools**: Generators yield results progressively

### v0.5.0 - Production Readiness

- ✅ **Cost Tracking**: Token counting and USD estimation
- ✅ **Better Errors**: PyTorch-style error messages with suggestions
