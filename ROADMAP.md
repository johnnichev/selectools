# Selectools Development Roadmap

This document tracks the implementation status of all planned features. See [README.md](README.md#roadmap--future-improvements) for detailed descriptions.

## Priority 1: Quick Wins (v0.4.0 - v0.5.1)

| Feature                   | Status         | Notes                                  |
| ------------------------- | -------------- | -------------------------------------- |
| Conversation Memory       | ✅ Implemented | Simple history management (v0.4)       |
| Async Support             | ✅ Implemented | `Agent.arun()`, async tools (v0.4)     |
| Anthropic Provider (Full) | ✅ Implemented | Real SDK integration with async (v0.4) |
| Gemini Provider (Full)    | ✅ Implemented | Real SDK integration with async (v0.4) |
| Remove Pillow Dependency  | ✅ Implemented | Removed bbox example (v0.4)            |
| Better Error Messages     | ✅ Implemented | PyTorch-style helpful errors (v0.5.0)  |
| Cost Tracking             | ✅ Implemented | Track tokens and API costs (v0.5.0)    |
| Pre-built Tool Library    | ✅ Implemented | 22 tools in 5 categories (v0.5.1)      |

---

## v0.5.x Series: Production Readiness ✅ COMPLETE

| Feature                         | Status    | Notes                                        |
| ------------------------------- | --------- | -------------------------------------------- |
| Conversation Memory             | ✅ v0.4.0 | Multi-turn context management                |
| Async Support                   | ✅ v0.4.0 | `Agent.arun()`, async tools, async providers |
| Anthropic/Gemini Providers      | ✅ v0.4.0 | Full SDK integration                         |
| Better Error Messages           | ✅ v0.5.0 | Custom exceptions with suggestions           |
| Cost Tracking                   | ✅ v0.5.0 | Token counting and cost estimation           |
| Gemini SDK Migration            | ✅ v0.5.0 | Updated to google-genai v1.0+                |
| Pre-built Tool Library          | ✅ v0.5.1 | 22 tools in 5 categories                     |
| Tool Validation at Registration | ✅ v0.5.2 | Validates tools at registration              |
| Observability Hooks             | ✅ v0.5.2 | 10 lifecycle hooks for monitoring            |

---

## v0.6.0: Enhanced Capabilities ✅ COMPLETE

| Feature              | Status    | Notes                                     |
| -------------------- | --------- | ----------------------------------------- |
| Local Model Support  | ✅ v0.6.0 | Ollama provider for local LLM execution   |
| Tool Usage Analytics | ✅ v0.6.0 | Track metrics, success rates, export data |

---

## v0.6.1: Streaming Tools (Completed)

| Feature                | Status    | Notes                                      |
| ---------------------- | --------- | ------------------------------------------ |
| Streaming Tool Results | ✅ v0.6.1 | Tools can yield results progressively      |
| Async Streaming        | ✅ v0.6.1 | Support for AsyncGenerator return types    |
| on_tool_chunk Hook     | ✅ v0.6.1 | Real-time chunk callbacks for streaming    |
| Streaming Analytics    | ✅ v0.6.1 | Track chunk counts and streaming metrics   |
| Toolbox Streaming      | ✅ v0.6.1 | read_file_stream, process_csv_stream tools |

---

## v0.7.0: Model Registry System ✅ COMPLETE

| Feature               | Status    | Notes                                    |
| --------------------- | --------- | ---------------------------------------- |
| Model Registry        | ✅ v0.7.0 | Canonical source of truth for 120 models |
| Typed Model Constants | ✅ v0.7.0 | IDE autocomplete for all models          |
| Model Metadata        | ✅ v0.7.0 | Pricing, context windows, max tokens     |
| Provider Integration  | ✅ v0.7.0 | All providers use model constants        |
| Example Migration     | ✅ v0.7.0 | All examples demonstrate new pattern     |

**Key Improvements:**

- **120 models** with complete metadata (64 OpenAI, 18 Anthropic, 25 Gemini, 13 Ollama)
- **IDE autocomplete** - Type `OpenAI.` and see all models
- **Type safety** - Catch typos at dev time
- **Single source of truth** - Update models.py, propagates everywhere
- **Ready for embeddings** - Structure prepared for v0.8.0 RAG support

---

## v0.8.0: Embeddings & RAG ✅ COMPLETE

| Feature                   | Status    | Notes                                            |
| ------------------------- | --------- | ------------------------------------------------ |
| Embedding Models Registry | ✅ v0.8.0 | 10 embedding models in models.py                 |
| Embedding Providers       | ✅ v0.8.0 | OpenAI, Anthropic/Voyage, Gemini, Cohere         |
| Vector Store Abstraction  | ✅ v0.8.0 | VectorStore interface with Document/SearchResult |
| Built-in Vector Stores    | ✅ v0.8.0 | InMemory, SQLite, Chroma, Pinecone               |
| Document Loaders          | ✅ v0.8.0 | Text, file, directory, PDF support               |
| Text Chunking Strategies  | ✅ v0.8.0 | TextSplitter, RecursiveTextSplitter              |
| RAG Tool                  | ✅ v0.8.0 | RAGTool for document Q&A                         |
| Semantic Search Tool      | ✅ v0.8.0 | SemanticSearchTool with scoring                  |
| Hybrid Search             | ⏸️ v0.9.0 | Deferred (Low priority)                          |

**Key Capabilities:**

- ✅ **Embedding Support**: 10 embedding models across 4 providers (OpenAI, Anthropic/Voyage, Gemini, Cohere)
- ✅ **Vector Databases**: Abstract VectorStore interface + 4 implementations (InMemory, SQLite, Chroma, Pinecone)
- ✅ **Document Processing**: Load from text/files/directories, automatic chunking, PDF support
- ✅ **RAG Tools**: RAGTool and SemanticSearchTool pre-built
- ✅ **Cost Tracking**: Full embedding cost tracking integrated with UsageStats
- ✅ **High-Level API**: RAGAgent.from_documents() and RAGAgent.from_directory()

**Example API:**

```python
from selectools import Agent, OpenAIProvider
from selectools.models import OpenAI
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.rag import VectorStore, DocumentLoader, RAGAgent

# Set up embedding provider
embedder = OpenAIEmbeddingProvider(model=OpenAI.TEXT_EMBEDDING_3_SMALL.id)

# Create vector store and load documents
vector_store = VectorStore.create("chroma", embedder=embedder)
docs = DocumentLoader.from_directory("./docs")

# High-level API - creates agent with RAG tool automatically
agent = RAGAgent.from_documents(
    documents=docs,
    provider=OpenAIProvider(),
    vector_store=vector_store,
    chunk_size=500
)

# Or use RAGTool directly
from selectools.rag import RAGTool
rag_tool = RAGTool(vector_store=vector_store, top_k=3)
agent = Agent(tools=[rag_tool], provider=OpenAIProvider())

response = agent.run("What are the main features of selectools?")
```

---

## v0.9.0: Upcoming (Planned)

| Feature              | Status     | Effort | Priority | Notes                            |
| -------------------- | ---------- | ------ | -------- | -------------------------------- |
| Hybrid Search        | 🟡 Planned | High   | Medium   | Vector + BM25 keyword search     |
| Reranking Models     | 🟡 Planned | Medium | Medium   | Cohere, Jina rerankers           |
| Advanced Chunking    | 🟡 Planned | High   | Low      | Agentic, contextual chunking     |
| Dynamic Tool Loading | 🟡 Planned | Medium | Low      | Hot-reload tools without restart |

---

## v0.9.0+: Future Enhancements

### High-Impact Complex Features

| Feature                 | Status     | Notes                         |
| ----------------------- | ---------- | ----------------------------- |
| Parallel Tool Execution | 🟡 Planned | Auto-detect independent tools |
| Tool Composition        | 🟡 Planned | `@compose` decorator          |

---

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

### Performance Optimizations

| Feature             | Status     | Notes                              |
| ------------------- | ---------- | ---------------------------------- |
| Caching Layer       | 🟡 Planned | LRU, semantic, distributed caching |
| Batch Processing    | 🟡 Planned | Efficient multi-request handling   |
| Prompt Optimization | 🟡 Planned | Automatic prompt compression       |

---

## Status Legend

- ✅ **Implemented** - Feature is complete and merged
- 🔵 **In Progress** - Actively being worked on
- 🟡 **Planned** - Scheduled for implementation
- 🟠 **Blocked** - Waiting on dependencies or decisions
- ⏸️ **Deferred** - Postponed to later release
- ❌ **Cancelled** - No longer planned

---

## How to Contribute

1. **Pick a feature** from Priority 1 or 2 (great for first-time contributors!)
2. **Comment on the issue** or create one if it doesn't exist
3. **Implement the feature** following [CONTRIBUTING.md](CONTRIBUTING.md)
4. **Submit a PR** with clear description
5. **Update this roadmap** to mark feature as ✅ Implemented

---

## Release Schedule

### v0.4.0 - Quick Wins

**Focus:** Developer experience improvements that close gaps with LangChain

**Completed:**

- ✅ Conversation Memory
- ✅ Async Support (Agent.arun(), async tools, async providers)
- ✅ Anthropic Provider (Full SDK integration)
- ✅ Gemini Provider (Full SDK integration)
- ✅ Removed Pillow dependency

**Remaining:**

- Better Error Messages
- Cost Tracking
- Pre-built Tool Library (at least 3 tools)

### v0.6.0 - High-Impact Features

**Focus:** Performance and observability

**Must-have:**

- ✅ Parallel Tool Execution
- ✅ Observability Hooks

**Nice-to-have:**

- Streaming Tool Results
- Tool Composition
- Interactive Debug Mode

### v0.7.0 - Advanced Features

**Focus:** Advanced context management and ecosystem

**Must-have:**

- ✅ Automatic Conversation Summarization
- ✅ Tool Marketplace (basic version)

**Nice-to-have:**

- Provider Auto-Selection
- Local Model Support
- Framework Integrations

### v1.0.0

**Focus:** Enterprise features and stability

**Must-have:**

- ✅ All Priority 1 & 2 features
- ✅ Comprehensive documentation
- ✅ 90%+ test coverage
- ✅ Security hardening
- ✅ Performance benchmarks

## Last Updated

**Date:** 2025-12-10
**By:** John (v0.8.0 completion)
**Next Review:** 2025-12-17

**Recent Changes:**

- ✅ Completed v0.8.0 - Full Embeddings & RAG support
- ✅ Implemented 4 embedding providers (OpenAI, Anthropic/Voyage, Gemini, Cohere)
- ✅ Built 4 vector store implementations (InMemory, SQLite, Chroma, Pinecone)
- ✅ Added document loaders with PDF support
- ✅ Created text chunking strategies (TextSplitter, RecursiveTextSplitter)
- ✅ Implemented RAGTool and SemanticSearchTool
- ✅ Extended cost tracking for embeddings
- ✅ Fixed all 65 failing tests - achieved 100% pass rate (463/463)
- ✅ Created 3 comprehensive RAG examples
