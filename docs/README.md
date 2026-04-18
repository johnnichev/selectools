# Selectools Implementation Documentation

**Version:** 0.16.6
**Last Updated:** March 2026

Welcome to the comprehensive technical documentation for selectools - a production-ready Python framework for building AI agents with tool-calling capabilities and RAG support.

---

## 📚 Documentation Structure

### Getting Started

- **[QUICKSTART.md](QUICKSTART.md)** - **Start here.** Build your first agent in 5 minutes, no API key needed.

### Main Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system overview, architecture diagrams, data flows, and design principles.

### Reference

- **[KEYS.md](KEYS.md)** - Environment variables and API keys for all providers
- **[RELEASE_GUIDE.md](RELEASE_GUIDE.md)** - PyPI release process and checklist

### Module Documentation

Detailed technical documentation for each module:

1. **[AGENT.md](modules/AGENT.md)** - Agent loop, structured output, traces, reasoning, batch, policy, observer, caching
2. **[STREAMING.md](modules/STREAMING.md)** - E2E streaming, parallel execution, routing mode, AgentResult, context propagation
3. **[TOOLS.md](modules/TOOLS.md)** - Tool definition, validation, registry, and streaming
4. **[DYNAMIC_TOOLS.md](modules/DYNAMIC_TOOLS.md)** - ToolLoader, dynamic tool loading, hot-reload, plugin systems
5. **[PARSER.md](modules/PARSER.md)** - TOOL_CALL contract and JSON extraction strategies
6. **[PROMPT.md](modules/PROMPT.md)** - System prompt generation and tool schema formatting
7. **[PROVIDERS.md](modules/PROVIDERS.md)** - LLM provider adapters, message formatting, and FallbackProvider
8. **[MEMORY.md](modules/MEMORY.md)** - Conversation memory, sliding windows, and tool-pair-aware trimming
9. **[USAGE.md](modules/USAGE.md)** - Usage tracking, cost calculation, and analytics
10. **[RAG.md](modules/RAG.md)** - Complete RAG pipeline from documents to answers
11. **[HYBRID_SEARCH.md](modules/HYBRID_SEARCH.md)** - BM25, hybrid search, fusion methods, and reranking
12. **[ADVANCED_CHUNKING.md](modules/ADVANCED_CHUNKING.md)** - Semantic and contextual document chunking
13. **[EMBEDDINGS.md](modules/EMBEDDINGS.md)** - Embedding providers and semantic search
14. **[VECTOR_STORES.md](modules/VECTOR_STORES.md)** - Vector database implementations
15. **[MODELS.md](modules/MODELS.md)** - 152 models across 5 providers with March 2026 pricing
16. **[GUARDRAILS.md](modules/GUARDRAILS.md)** - Input/output validation pipeline, PII redaction, topic blocking
17. **[AUDIT.md](modules/AUDIT.md)** - JSONL audit logging with privacy controls
18. **[SECURITY.md](modules/SECURITY.md)** - Tool output screening and coherence checking
19. **[TOOLBOX.md](modules/TOOLBOX.md)** - 24 pre-built tools across 5 categories (file, web, data, datetime, text)
20. **[EXCEPTIONS.md](modules/EXCEPTIONS.md)** - Error hierarchy, exception attributes, catch patterns
21. **[SESSIONS.md](modules/SESSIONS.md)** - Persistent session storage with 4 backends
22. **[ENTITY_MEMORY.md](modules/ENTITY_MEMORY.md)** - Named entity extraction and tracking
23. **[KNOWLEDGE_GRAPH.md](modules/KNOWLEDGE_GRAPH.md)** - Relationship triple extraction and graph memory
24. **[KNOWLEDGE.md](modules/KNOWLEDGE.md)** - Cross-session knowledge with daily logs and persistent facts

---

## 🎯 Quick Navigation

### By Role

**For Developers:**

- Start: [ARCHITECTURE.md](ARCHITECTURE.md)
- Building agents: [AGENT.md](modules/AGENT.md)
- Creating tools: [TOOLS.md](modules/TOOLS.md)
- Pre-built toolbox: [TOOLBOX.md](modules/TOOLBOX.md)
- Dynamic tools & plugins: [DYNAMIC_TOOLS.md](modules/DYNAMIC_TOOLS.md)
- Adding RAG: [RAG.md](modules/RAG.md)
- Error handling: [EXCEPTIONS.md](modules/EXCEPTIONS.md)

**For Contributors:**

- Adding providers: [PROVIDERS.md](modules/PROVIDERS.md)
- Adding vector stores: [VECTOR_STORES.md](modules/VECTOR_STORES.md)
- Understanding parser: [PARSER.md](modules/PARSER.md)

**For DevOps/Production:**

- Cost tracking: [USAGE.md](modules/USAGE.md)
- Model selection: [MODELS.md](modules/MODELS.md)
- Monitoring: [AGENT.md](modules/AGENT.md#agentobserver-protocol) (AgentObserver + AsyncAgentObserver, hooks deprecated), and `result.trace.to_otel_spans()` for OpenTelemetry
- Guardrails & safety: [GUARDRAILS.md](modules/GUARDRAILS.md)
- Audit logging: [AUDIT.md](modules/AUDIT.md)
- Prompt injection defence: [SECURITY.md](modules/SECURITY.md)

### By Feature

**Tool Calling:**

- [AGENT.md](modules/AGENT.md) - Orchestration
- [TOOLS.md](modules/TOOLS.md) - Definition
- [DYNAMIC_TOOLS.md](modules/DYNAMIC_TOOLS.md) - Dynamic loading, plugins, hot-reload
- [PARSER.md](modules/PARSER.md) - Parsing
- [PROMPT.md](modules/PROMPT.md) - Prompting

**RAG System:**

- [RAG.md](modules/RAG.md) - Overview
- [HYBRID_SEARCH.md](modules/HYBRID_SEARCH.md) - Hybrid search & reranking
- [ADVANCED_CHUNKING.md](modules/ADVANCED_CHUNKING.md) - Semantic & contextual chunking
- [EMBEDDINGS.md](modules/EMBEDDINGS.md) - Vector generation
- [VECTOR_STORES.md](modules/VECTOR_STORES.md) - Storage

**Security & Compliance:**

- [GUARDRAILS.md](modules/GUARDRAILS.md) - Input/output validation pipeline
- [SECURITY.md](modules/SECURITY.md) - Tool output screening & coherence checking
- [AUDIT.md](modules/AUDIT.md) - JSONL audit trail with privacy controls

**Memory & Persistence:**

- [SESSIONS.md](modules/SESSIONS.md) - Persistent session storage
- [ENTITY_MEMORY.md](modules/ENTITY_MEMORY.md) - Entity extraction and tracking
- [KNOWLEDGE_GRAPH.md](modules/KNOWLEDGE_GRAPH.md) - Relationship triple graph
- [KNOWLEDGE.md](modules/KNOWLEDGE.md) - Cross-session knowledge memory

**Streaming & Performance:**

- [STREAMING.md](modules/STREAMING.md) - E2E streaming, parallel execution, routing mode

**Cost Management & Caching:**

- [USAGE.md](modules/USAGE.md) - Tracking
- [MODELS.md](modules/MODELS.md) - Pricing
- [AGENT.md](modules/AGENT.md#response-caching) - Response caching

---

## 📊 Documentation Stats

- **Total files:** 25 (1 main + 24 modules)
- **ASCII diagrams:** 30+ diagrams
- **Code examples:** 250+ examples

---

## 🔍 Understanding the Flow

### Standard Tool-Calling Flow

```
1. User Query
   ↓
2. INPUT GUARDRAILS validate/redact user message (PII, topic, toxicity)
   ↓
3. AGENT loads history (MEMORY) and calls PROVIDER (or FALLBACK chain)
   ↓
4. CACHE checked (if configured) → hit? Return cached response
   ↓
5. PROVIDER formats prompt (PROMPT + STRUCTURED schema) and calls LLM → CACHE stores result
   ↓
6. OUTPUT GUARDRAILS validate LLM response (format, length, toxicity)
   ↓
7. PARSER extracts TOOL_CALL from response; REASONING extracted
   ↓
8. POLICY ENGINE evaluates tool call → allow/review/deny
   ↓
8b. If review: HUMAN-IN-THE-LOOP callback → approve/reject
   ↓
9. COHERENCE CHECK verifies tool call matches user intent
   ↓
10. TOOLS validates and executes (parallel if multiple)
   ↓
11. OUTPUT SCREENING checks tool results for prompt injection
   ↓
12. TRACE records each step; AUDIT LOGGER writes JSONL; USAGE tracks costs
   ↓
13. If response_format: STRUCTURED validates → retry on failure
   ↓
14. Loop continues or returns AgentResult (with .parsed, .trace, .reasoning)
```

**Read:**

1. [AGENT.md](modules/AGENT.md) - Main loop
2. [PROVIDERS.md](modules/PROVIDERS.md) - LLM communication
3. [PARSER.md](modules/PARSER.md) - Response parsing
4. [TOOLS.md](modules/TOOLS.md) - Tool execution

### RAG Flow

```
1. Documents
   ↓
2. LOADERS read files/PDFs
   ↓
3. CHUNKING splits into pieces
   (TextSplitter → Recursive → Semantic → Contextual)
   ↓
4. EMBEDDINGS generate vectors
   ↓
5. VECTOR_STORES persist
   ↓
6. Query → Hybrid Search (Vector + BM25) → Fusion → Rerank → Answer
```

**Read:**

1. [RAG.md](modules/RAG.md) - Complete pipeline
2. [ADVANCED_CHUNKING.md](modules/ADVANCED_CHUNKING.md) - Semantic & contextual chunking
3. [EMBEDDINGS.md](modules/EMBEDDINGS.md) - Vector generation
4. [VECTOR_STORES.md](modules/VECTOR_STORES.md) - Storage
5. [HYBRID_SEARCH.md](modules/HYBRID_SEARCH.md) - BM25, hybrid search & reranking

---

## 🎓 Learning Path

### Beginner

1. Follow [QUICKSTART.md](QUICKSTART.md) - Build your first agent in 5 minutes
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) - Get the big picture
3. Read [AGENT.md](modules/AGENT.md) - Understand the core loop
4. Read [TOOLS.md](modules/TOOLS.md) - Learn to create tools

### Intermediate

1. Read [PARSER.md](modules/PARSER.md) - Understand parsing
2. Read [PROVIDERS.md](modules/PROVIDERS.md) - Switch providers
3. Read [MEMORY.md](modules/MEMORY.md) - Add conversations
4. Read [USAGE.md](modules/USAGE.md) - Track costs

### Advanced

1. Read [RAG.md](modules/RAG.md) - Add document search
2. Read [HYBRID_SEARCH.md](modules/HYBRID_SEARCH.md) - Hybrid search & reranking
3. Read [ADVANCED_CHUNKING.md](modules/ADVANCED_CHUNKING.md) - Semantic & contextual chunking
4. Read [STREAMING.md](modules/STREAMING.md) - Streaming, parallel execution, routing
5. Read [DYNAMIC_TOOLS.md](modules/DYNAMIC_TOOLS.md) - Plugin systems & hot-reload

### Production / Enterprise

1. Read [GUARDRAILS.md](modules/GUARDRAILS.md) - Input/output validation pipeline
2. Read [AUDIT.md](modules/AUDIT.md) - Compliance logging
3. Read [SECURITY.md](modules/SECURITY.md) - Prompt injection defence

### Memory & Persistence

1. Read [SESSIONS.md](modules/SESSIONS.md) - Persistent sessions
2. Read [ENTITY_MEMORY.md](modules/ENTITY_MEMORY.md) - Entity tracking
3. Read [KNOWLEDGE_GRAPH.md](modules/KNOWLEDGE_GRAPH.md) - Knowledge graphs
4. Read [KNOWLEDGE.md](modules/KNOWLEDGE.md) - Cross-session knowledge
5. Build production RAG and agent systems!

---

## 💡 Key Concepts

### Design Principles

1. **Provider Agnosticism** - Switch LLMs without code changes
2. **Library-First** - Composable, no framework lock-in
3. **Production Hardened** - Retries, timeouts, validation
4. **Developer Friendly** - Type hints, decorators, clear errors
5. **Observable** - AgentObserver + AsyncAgentObserver protocol, OTel span export, usage tracking (hooks deprecated)
6. **Cost Aware** - Automatic tracking and warnings
7. **Performance Optimized** - Parallel tool execution, response caching, async-first design
8. **Enterprise Secure** - Guardrails, PII redaction, prompt injection screening, coherence checking, audit logging

### Core Patterns

- **Agent Loop** - Iterative tool calling until completion
- **Structured Output** - Pydantic/JSON Schema validation with auto-retry
- **Execution Traces** - Structured timeline on every run (`result.trace`)
- **Reasoning Visibility** - Why the agent chose a tool (`result.reasoning`)
- **Provider Fallback** - Priority-ordered providers with circuit breaker
- **Batch Processing** - Concurrent multi-prompt execution
- **Tool Policy Engine** - Declarative allow/review/deny with HITL approval
- **Native Tool Calling** - Provider-native function calling APIs
- **Parallel Execution** - Concurrent tool calls via asyncio.gather
- **Tool Calling Contract** - TOOL_CALL with JSON payload
- **Schema Generation** - Automatic from type hints
- **Injected Parameters** - Hide secrets from LLM
- **Streaming** - Progressive results via generators
- **Response Caching** - LRU+TTL caching for identical LLM requests
- **RAG Pipeline** - Load → Chunk → Embed → Store → Search
- **Hybrid Search** - BM25 + vector fusion with optional reranking
- **Semantic Chunking** - Embedding-based topic-boundary splitting
- **Contextual Chunking** - LLM-enriched chunks for better retrieval
- **Dynamic Tool Loading** - Plugin discovery, hot-reload, runtime tool management
- **Routing Mode** - Tool selection without execution for intent classification
- **Guardrails Engine** - Input/output content validation with block/rewrite/warn actions
- **Audit Logging** - JSONL audit trail with privacy controls and daily rotation
- **Tool Output Screening** - Pattern-based prompt injection detection
- **Coherence Checking** - LLM-based intent verification for tool calls

---

## 🔧 Common Tasks

### Create an Agent

See: [AGENT.md](modules/AGENT.md)

### Define a Tool

See: [TOOLS.md](modules/TOOLS.md)

### Use Pre-Built Tools

See: [TOOLBOX.md](modules/TOOLBOX.md)

### Handle Errors

See: [EXCEPTIONS.md](modules/EXCEPTIONS.md)

### Look Up Model Pricing

See: [MODELS.md — Programmatic Pricing API](modules/MODELS.md#programmatic-pricing-api)

### Add RAG

See: [RAG.md](modules/RAG.md)

### Get Structured Output

See: [AGENT.md — Structured Output](modules/AGENT.md#structured-output)

### See Execution Traces

See: [AGENT.md — Execution Traces](modules/AGENT.md#execution-traces)

### Add Provider Fallback

See: [PROVIDERS.md — FallbackProvider](modules/PROVIDERS.md#fallbackprovider)

### Batch Processing

See: [AGENT.md — Batch Processing](modules/AGENT.md#batch-processing)

### Add Tool Policies

See: [AGENT.md — Tool Policy](modules/AGENT.md#tool-policy-human-in-the-loop)

### Add Guardrails

See: [GUARDRAILS.md](modules/GUARDRAILS.md)

### Add Audit Logging

See: [AUDIT.md](modules/AUDIT.md)

### Screen Tool Outputs for Injection

See: [SECURITY.md — Tool Output Screening](modules/SECURITY.md#tool-output-screening)

### Enable Coherence Checking

See: [SECURITY.md — Coherence Checking](modules/SECURITY.md#coherence-checking)

### Monitor with AgentObserver

See: [AGENT.md — AgentObserver Protocol](modules/AGENT.md#agentobserver-protocol)

### Export Traces to OpenTelemetry

See: [AGENT.md — AgentObserver Protocol](modules/AGENT.md#agentobserver-protocol) (`result.trace.to_otel_spans()`)

### Switch Providers

See: [PROVIDERS.md](modules/PROVIDERS.md)

### Add Hybrid Search

See: [HYBRID_SEARCH.md](modules/HYBRID_SEARCH.md)

### Use Advanced Chunking

See: [ADVANCED_CHUNKING.md](modules/ADVANCED_CHUNKING.md)

### Stream Responses

See: [STREAMING.md](modules/STREAMING.md)

### Load Tools Dynamically

See: [DYNAMIC_TOOLS.md](modules/DYNAMIC_TOOLS.md)

### Track Costs

See: [USAGE.md](modules/USAGE.md)

### Choose a Model

See: [MODELS.md](modules/MODELS.md)

---

## 🚀 Next Steps

1. **Read the [ARCHITECTURE.md](ARCHITECTURE.md)** for system overview
2. **Explore module docs** based on your needs
3. **Check the main [README](https://github.com/johnnichev/selectools#readme)** for quick start examples
4. **Review the [Roadmap](https://github.com/johnnichev/selectools/blob/main/ROADMAP.md)** for upcoming features

---

## 📝 Contributing

Found an error or want to improve the docs?

1. Check the source code in `src/selectools/`
2. Submit issues or PRs on GitHub
3. Follow the patterns established in existing docs

---

**Built with ❤️ for developers who want to understand their tools.**
