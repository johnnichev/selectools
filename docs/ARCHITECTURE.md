# Selectools Architecture

**Version:** 0.12.0
**Last Updated:** February 2026

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Module Dependencies](#module-dependencies)
6. [Design Principles](#design-principles)
7. [RAG Integration](#rag-integration)

---

## Overview

Selectools is a production-ready Python framework for building AI agents with tool-calling capabilities and Retrieval-Augmented Generation (RAG). The library provides a unified interface across multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama) and handles the complexity of tool execution, conversation management, cost tracking, and semantic search.

### Key Features

- **Provider-Agnostic**: Switch between OpenAI, Anthropic, Gemini, and Ollama with one line
- **Production-Ready**: Robust error handling, retry logic, timeouts, and validation
- **RAG Support**: 4 embedding providers, 4 vector stores, document loaders
- **Developer-Friendly**: Type hints, `@tool` decorator, automatic schema inference
- **Observable**: Built-in hooks, analytics, usage tracking, and cost monitoring
- **Native Tool Calling**: OpenAI, Anthropic, and Gemini native function calling APIs
- **Streaming**: E2E token-level streaming with native tool call support via `Agent.astream`
- **Parallel Execution**: Concurrent tool execution via `asyncio.gather` / `ThreadPoolExecutor`
- **Response Caching**: Built-in LRU+TTL cache (`InMemoryCache`) and distributed `RedisCache`

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            USER APPLICATION                              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              AGENT                                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Agent Loop (agent.py)                                           │  │
│  │  • Iterative execution                                           │  │
│  │  • Tool call detection                                           │  │
│  │  • Error handling & retries                                      │  │
│  │  • Hooks (observability)                                         │  │
│  │  • Parallel tool execution                                       │  │
│  │  • Response caching (LRU+TTL)                                    │  │
│  └─────────┬────────────────────────┬──────────────────┬────────────┘  │
│            │                        │                  │               │
│            ▼                        ▼                  ▼               │
│  ┌─────────────────┐    ┌──────────────────┐  ┌──────────────────┐   │
│  │  PromptBuilder  │    │  ToolCallParser  │  │  ConversationMemory│  │
│  │  (prompt.py)    │    │  (parser.py)     │  │  (memory.py)     │   │
│  │  • System prompt│    │  • JSON parsing  │  │  • History mgmt  │   │
│  │  • Tool schemas │    │  • Error recovery│  │  • Sliding window│   │
│  └─────────────────┘    └──────────────────┘  └──────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROVIDER LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │   OpenAI     │  │  Anthropic   │  │    Gemini    │  │   Ollama  │  │
│  │   Provider   │  │   Provider   │  │   Provider   │  │  Provider │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│         │                 │                  │                │         │
│         └─────────────────┴──────────────────┴────────────────┘         │
│                                 │                                        │
│                     ┌───────────▼───────────┐                           │
│                     │   Provider Protocol   │                           │
│                     │   (base.py)           │                           │
│                     │   • complete()        │                           │
│                     │   • stream()          │                           │
│                     │   • acomplete()       │                           │
│                     │   • astream()         │                           │
│                     └───────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
┌─────────────────────────────┐   ┌─────────────────────────────────────┐
│      TOOL SYSTEM            │   │         RAG SYSTEM                  │
│  ┌──────────────────────┐   │   │  ┌──────────────────────────────┐ │
│  │  Tool (tools.py)     │   │   │  │  DocumentLoader              │ │
│  │  • Definition        │   │   │  │  • from_file()               │ │
│  │  • Validation        │   │   │  │  • from_directory()          │ │
│  │  • Execution         │   │   │  │  • from_pdf()                │ │
│  │  • Streaming support │   │   │  └────────┬─────────────────────┘ │
│  └──────────────────────┘   │   │           ▼                        │
│  ┌──────────────────────┐   │   │  ┌──────────────────────────────┐ │
│  │  @tool decorator     │   │   │  │  TextSplitter / Recursive    │ │
│  │  • Auto schema       │   │   │  │  • Chunking strategies       │ │
│  │  • Type inference    │   │   │  └────────┬─────────────────────┘ │
│  └──────────────────────┘   │   │           ▼                        │
│  ┌──────────────────────┐   │   │  ┌──────────────────────────────┐ │
│  │  ToolRegistry        │   │   │  │  EmbeddingProvider          │ │
│  │  • Organization      │   │   │  │  • OpenAI / Anthropic       │ │
│  │  • Discovery         │   │   │  │  • Gemini / Cohere          │ │
│  └──────────────────────┘   │   │  └────────┬─────────────────────┘ │
└─────────────────────────────┘   │           ▼                        │
                                  │  ┌──────────────────────────────┐ │
                                  │  │  VectorStore                 │ │
                                  │  │  • Memory / SQLite           │ │
                                  │  │  • Chroma / Pinecone         │ │
                                  │  └────────┬─────────────────────┘ │
                                  │           ▼                        │
                                  │  ┌──────────────────────────────┐ │
                                  │  │  RAGTool                     │ │
                                  │  │  • search_knowledge_base()   │ │
                                  │  └──────────────────────────────┘ │
                                  └─────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SUPPORT SYSTEMS                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │   Usage      │  │  Analytics   │  │   Pricing    │  │   Models  │  │
│  │   Tracking   │  │  (analytics) │  │  (pricing)   │  │ (registry)│  │
│  │   (usage.py) │  │  • Metrics   │  │  • Cost calc │  │  • 135+   │  │
│  │   • Tokens   │  │  • Patterns  │  │  • Per model │  │   models  │  │
│  │   • Cost     │  │  • Success   │  │              │  │           │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Agent (`agent.py`)

The **Agent** is the orchestrator that manages the iterative loop of:

1. Sending messages to the LLM provider
2. Parsing responses for tool calls
3. Executing requested tools
4. Feeding results back to the LLM
5. Repeating until task completion or max iterations

**Key Responsibilities:**

- Conversation management with optional memory
- Retry logic with exponential backoff
- Rate limit detection and handling
- Tool timeout enforcement
- Hook invocation for observability
- Async/sync execution support
- Parallel tool execution for concurrent multi-tool calls
- Streaming responses via `astream()`
- Response caching to avoid redundant LLM calls

### 2. Tools (`tools.py`)

**Tools** are Python functions that agents can invoke. The tool system provides:

- Automatic JSON schema generation from type hints
- Runtime parameter validation with helpful error messages
- Support for sync/async and streaming (Generator/AsyncGenerator)
- Injected kwargs for clean separation of concerns
- `@tool` decorator for ergonomic definition
- `ToolRegistry` for organization

### 3. Providers (`providers/`)

**Providers** are adapters that translate between the library's unified interface and specific LLM APIs:

- `OpenAIProvider` - OpenAI Chat Completions
- `AnthropicProvider` - Claude Messages API
- `GeminiProvider` - Google Generative AI
- `OllamaProvider` - Local LLM execution

Each implements the `Provider` protocol with `complete()`, `stream()`, `acomplete()`, and `astream()` methods. Native tool calling is supported via the `tools` parameter.

### 4. Parser (`parser.py`)

**ToolCallParser** robustly extracts `TOOL_CALL` directives from LLM responses:

- Handles fenced code blocks, inline JSON, mixed content
- Balanced bracket parsing for nested JSON
- Lenient JSON parsing with fallbacks
- Supports variations: `tool_name`/`tool`/`name` and `parameters`/`params`

### 5. Prompt Builder (`prompt.py`)

**PromptBuilder** generates system prompts with:

- Tool calling contract specification
- JSON schema for each available tool
- Best practices and constraints
- Customizable base instructions

### 6. Memory (`memory.py`)

**ConversationMemory** maintains multi-turn dialogue history:

- Sliding window with configurable limits (message count, token count)
- Automatic pruning when limits exceeded
- Integrates seamlessly with Agent

### 7. RAG System (`rag/`)

The **RAG module** provides end-to-end document search:

- **DocumentLoader**: Load from files, directories, PDFs
- **TextSplitter**: Chunk documents intelligently
- **EmbeddingProvider**: Generate vector embeddings
- **VectorStore**: Store and search embeddings
- **RAGTool**: Pre-built knowledge base search tool
- **RAGAgent**: High-level API for RAG agents

### 8. Usage Tracking (`usage.py`, `pricing.py`)

Automatic monitoring of:

- Token consumption (prompt, completion, embedding)
- Cost estimation (per model from registry)
- Per-tool attribution
- Iteration-by-iteration breakdown

### 9. Analytics (`analytics.py`)

Tool usage analytics with:

- Call counts and success rates
- Execution timing
- Parameter patterns
- Streaming metrics
- Export to JSON/CSV

### 10. Model Registry (`models.py`)

Single source of truth for 130+ models:

- Pricing per 1M tokens
- Context windows
- Max output tokens
- Typed constants for IDE autocomplete

---

## Data Flow

### Standard Tool-Calling Flow

```
1. User Message
   │
   ├─→ Agent receives message(s)
   │
2. Conversation History
   │
   ├─→ Memory.get_history() [if enabled]
   ├─→ Append new messages
   │
3. Prompt Building
   │
   ├─→ PromptBuilder.build(tools)
   ├─→ System prompt with tool schemas
   │
4. Cache Lookup [if cache configured]
   │
   ├─→ CacheKeyBuilder.build(model, prompt, messages, tools, temperature)
   ├─→ Cache.get(key) → hit? Return cached (Message, UsageStats)
   │
5. LLM Request [on cache miss]
   │
   ├─→ Provider.complete(model, prompt, messages)
   ├─→ [OpenAI / Anthropic / Gemini / Ollama]
   ├─→ Cache.set(key, response) [store for future hits]
   │
6. Response Parsing
   │
   ├─→ ToolCallParser.parse(response_text)
   ├─→ Extract: tool_name, parameters
   │
7. Tool Execution [if tool call detected]
   │
   ├─→ Tool.validate(parameters)
   ├─→ Tool.execute(parameters, injected_kwargs)
   ├─→ Parallel execution if multiple tools (asyncio.gather / ThreadPoolExecutor)
   ├─→ Handle timeout, errors, streaming
   │
8. Feedback Loop [if tool executed]
   │
   ├─→ Append ASSISTANT message (tool call)
   ├─→ Append TOOL message (result)
   ├─→ Return to step 4 (next iteration)
   │
9. Final Response [no tool call]
   │
   ├─→ Memory.add(response) [if enabled]
   ├─→ Return to user
```

### RAG-Enhanced Flow

```
1. User Question
   │
2. [First Time Setup]
   │
   ├─→ DocumentLoader.from_directory("./docs")
   ├─→ TextSplitter.split_documents(docs)
   ├─→ EmbeddingProvider.embed_texts(chunks)
   ├─→ VectorStore.add_documents(chunks, embeddings)
   │
3. Query Processing
   │
   ├─→ Agent receives question
   ├─→ LLM decides to use RAGTool
   │
4. Knowledge Base Search
   │
   ├─→ EmbeddingProvider.embed_query(question)
   ├─→ VectorStore.search(query_embedding, top_k=3)
   ├─→ Return top matches with scores
   │
5. Context Integration
   │
   ├─→ Format results with source citations
   ├─→ Return to Agent as tool result
   │
6. Response Generation
   │
   ├─→ LLM generates answer using retrieved context
   ├─→ Return to user
```

---

## Module Dependencies

```
┌────────────────┐
│   __init__.py  │  (Public API)
└────────┬───────┘
         │
         ├─→ agent.py
         │    ├─→ types.py (Message, Role, ToolCall)
         │    ├─→ tools.py (Tool)
         │    ├─→ prompt.py (PromptBuilder)
         │    ├─→ parser.py (ToolCallParser)
         │    ├─→ providers/base.py (Provider)
         │    ├─→ memory.py (ConversationMemory)
         │    ├─→ usage.py (AgentUsage, UsageStats)
         │    ├─→ analytics.py (AgentAnalytics)
         │    └─→ cache.py (Cache, InMemoryCache, CacheKeyBuilder)
         │
         ├─→ cache.py (core caching)
         │    └─→ types.py, tools.py, usage.py
         │
         ├─→ cache_redis.py (distributed caching, optional)
         │    └─→ cache.py (CacheStats)
         │
         ├─→ tools.py
         │    ├─→ types.py
         │    └─→ exceptions.py
         │
         ├─→ providers/
         │    ├─→ base.py (Provider protocol)
         │    ├─→ openai_provider.py
         │    ├─→ anthropic_provider.py
         │    ├─→ gemini_provider.py
         │    └─→ ollama_provider.py
         │         └─→ types.py, usage.py, pricing.py
         │
         ├─→ rag/
         │    ├─→ vector_store.py (Document, SearchResult, VectorStore)
         │    ├─→ loaders.py (DocumentLoader)
         │    ├─→ chunking.py (TextSplitter, RecursiveTextSplitter)
         │    ├─→ tools.py (RAGTool, SemanticSearchTool)
         │    └─→ __init__.py (RAGAgent)
         │         └─→ agent.py, tools.py
         │
         ├─→ embeddings/
         │    ├─→ provider.py (EmbeddingProvider protocol)
         │    ├─→ openai.py
         │    ├─→ anthropic.py
         │    ├─→ gemini.py
         │    └─→ cohere.py
         │
         ├─→ pricing.py
         │    └─→ models.py
         │
         └─→ models.py (Model registry)
```

### Import Guidelines

- **Core modules** (`types`, `tools`, `agent`) have minimal dependencies
- **Providers** depend only on core modules and their SDK
- **RAG system** is self-contained, depends on `agent` only for `RAGAgent`
- **Optional dependencies** (ChromaDB, Pinecone, etc.) are lazy-loaded

---

## Design Principles

### 1. Provider Agnosticism

**Problem:** Each LLM provider has different APIs, message formats, and capabilities.

**Solution:** The `Provider` protocol defines a unified interface. Providers handle translation:

- Message format conversion
- Role mapping (e.g., `TOOL` → `ASSISTANT` for OpenAI)
- Image encoding (base64 for vision)
- Streaming implementation

**Benefit:** Switch providers with one line change, no refactoring.

### 2. Library-First Design

**Problem:** Frameworks often take over your application with magic globals and hidden state.

**Solution:** Selectools is a library you import and compose:

- No global state
- Explicit dependency injection
- Use as much or as little as needed
- Integrates with existing code

**Benefit:** Full control, no framework lock-in.

### 3. Production Hardening

**Problem:** Real-world LLM applications fail in ways demos don't.

**Solution:** Built-in robustness:

- **Retry logic**: Exponential backoff for rate limits
- **Timeouts**: Request-level and tool-level
- **Validation**: Early parameter checking with helpful errors
- **Error recovery**: Lenient parsing, fallback strategies
- **Iteration caps**: Prevent runaway costs

**Benefit:** Reliable in production environments.

### 4. Developer Ergonomics

**Problem:** Boilerplate code slows development.

**Solution:** Minimal API surface:

- `@tool` decorator with auto schema inference
- Type hints generate JSON schemas
- Default values make parameters optional
- IDE autocomplete for all models
- Clear error messages with suggestions

**Benefit:** Fast prototyping, maintainable code.

### 5. Type Safety

**Problem:** Runtime errors from typos and type mismatches.

**Solution:** Full type hints everywhere:

- `ModelInfo` dataclass for model metadata
- Typed constants (`OpenAI.GPT_4O`)
- Protocol-based interfaces
- MyPy compatibility

**Benefit:** Catch errors at development time.

### 6. Observability

**Problem:** Black box behavior makes debugging hard.

**Solution:** Hook system for lifecycle events:

- `on_agent_start/end`
- `on_iteration_start/end`
- `on_tool_start/end/error/chunk`
- `on_llm_start/end`
- `on_error`

**Benefit:** Full visibility into agent behavior.

### 7. Cost Awareness

**Problem:** Unpredictable LLM costs.

**Solution:** Automatic tracking:

- Token counting per request
- Cost calculation per model
- Per-tool attribution
- Warning thresholds
- Embedding cost tracking (RAG)

**Benefit:** Budget control and optimization.

### 8. Performance

**Problem:** Sequential tool execution wastes time when tools are independent.

**Solution:** Automatic parallel execution:

- `asyncio.gather()` for async (`arun`, `astream`)
- `ThreadPoolExecutor` for sync (`run`)
- Results preserved in original order
- Enabled by default, configurable via `parallel_tool_execution`

**Benefit:** Faster agent loops when LLM requests multiple independent tools.

### 9. Response Caching

**Problem:** Identical LLM requests are expensive and wasteful.

**Solution:** Pluggable cache layer:

- `Cache` protocol for custom backends
- `InMemoryCache`: LRU + TTL with `OrderedDict`, thread-safe, zero dependencies
- `RedisCache`: Distributed TTL cache for multi-process deployments
- Deterministic key generation via `CacheKeyBuilder` (SHA-256 hash)
- Opt-in via `AgentConfig(cache=InMemoryCache())`

**Benefit:** Eliminate redundant LLM calls, reduce cost and latency.

---

## RAG Integration

### Architecture

The RAG system is designed as a composable pipeline:

```
Documents → Loader → Chunker → Embedder → VectorStore → RAGTool → Agent
```

Each component can be used independently or combined via `RAGAgent` high-level API.

### Document Processing Pipeline

1. **Loading**: `DocumentLoader` supports text, files, directories, PDFs
2. **Chunking**: `TextSplitter` / `RecursiveTextSplitter` with overlap
3. **Embedding**: Provider-agnostic embedding interface
4. **Storage**: VectorStore abstraction (Memory, SQLite, Chroma, Pinecone)
5. **Retrieval**: Semantic search with score thresholds

### Vector Store Abstraction

All vector stores implement the same interface:

- `add_documents(documents, embeddings)` → ids
- `search(query_embedding, top_k, filter)` → SearchResults
- `delete(ids)`
- `clear()`

This allows switching backends without changing agent code.

### RAGAgent High-Level API

Three convenient constructors:

- `RAGAgent.from_documents(docs, provider, store)` - Direct document list
- `RAGAgent.from_directory(path, provider, store)` - Load from folder
- `RAGAgent.from_files(paths, provider, store)` - Load specific files

All handle chunking, embedding, and tool setup automatically.

### Cost Tracking

RAG operations track both:

- **LLM costs**: Standard token counting
- **Embedding costs**: Per-token embedding API costs

Total cost = LLM cost + Embedding cost

---

## Extension Points

### Adding a New Provider

1. Implement the `Provider` protocol in `providers/`
2. Define `complete()` and `stream()` methods
3. Handle message formatting in `_format_messages()`
4. Map roles and content appropriately
5. Extract usage stats and calculate cost

### Adding a New Vector Store

1. Inherit from `VectorStore` abstract base class
2. Implement: `add_documents()`, `search()`, `delete()`, `clear()`
3. Register in `VectorStore.create()` factory
4. Add to `rag/stores/` directory

### Adding a New Tool

```python
from selectools import tool

@tool(description="Your tool description")
def my_tool(param1: str, param2: int = 10) -> str:
    """Tool implementation."""
    return f"Result: {param1}, {param2}"
```

Schema is auto-generated from type hints and defaults.

### Custom Hooks

```python
def my_hook(tool_name, args):
    print(f"Tool: {tool_name}, Args: {args}")

config = AgentConfig(hooks={"on_tool_start": my_hook})
agent = Agent(tools=[...], provider=provider, config=config)
```

---

## Performance Considerations

### Token Efficiency

- Use smaller models (GPT-4o-mini, Haiku) when appropriate
- Limit conversation history with `ConversationMemory`
- Set `max_tokens` to prevent over-generation
- Use `top_k` parameter to limit RAG context

### Async for Concurrency

- Use `Agent.arun()` for non-blocking execution
- Async tools with `async def`
- Concurrent requests with `asyncio.gather()`
- Parallel tool execution via `Agent.astream()` with `asyncio.gather()`
- Better performance in web frameworks (FastAPI)

### Vector Store Selection

- **Memory**: Fast, but not persistent (prototyping)
- **SQLite**: Good balance, local persistence
- **Chroma**: Advanced features, 10k+ documents
- **Pinecone**: Cloud-hosted, production scale

### Response Caching

Built-in caching avoids redundant LLM calls for identical requests:

- **`InMemoryCache`**: Thread-safe LRU + TTL cache, zero dependencies
- **`RedisCache`**: Distributed TTL cache for multi-process / multi-server deployments
- Cache key is a SHA-256 hash of (model, system_prompt, messages, tools, temperature)
- Streaming (`astream`) bypasses cache (non-replayable)
- Cache hits still contribute to usage tracking

```python
from selectools import Agent, AgentConfig, InMemoryCache

cache = InMemoryCache(max_size=500, default_ttl=600)
config = AgentConfig(cache=cache)
agent = Agent(tools=[...], provider=provider, config=config)

# Second identical call returns cached response (no LLM call)
response1 = agent.run([Message(role=Role.USER, content="Hello")])
agent.reset()
response2 = agent.run([Message(role=Role.USER, content="Hello")])

print(cache.stats)  # CacheStats(hits=1, misses=1, ...)
```

### General Caching Tips

- Keep `VectorStore` instance alive between queries
- Reuse `Agent` instance for same tool set
- Batch embedding operations with `embed_texts()`

---

## Testing Strategy

### Unit Tests

- Core modules tested in isolation
- Mock providers for agent logic
- Schema validation edge cases
- Parser robustness tests

### Integration Tests

- Full agent loops with real providers
- RAG pipeline end-to-end
- Multi-turn conversations with memory
- Error scenarios and recovery

### Fixtures

- `LocalProvider` for offline testing
- `SELECTOOLS_BBOX_MOCK_JSON` for deterministic tool calls
- Mock vector stores for RAG tests

---

## Further Reading

- [Agent Module](modules/AGENT.md) - Detailed agent loop documentation
- [Tools Module](modules/TOOLS.md) - Tool system deep dive
- [RAG System](modules/RAG.md) - Complete RAG pipeline
- [Providers](modules/PROVIDERS.md) - Provider implementations
- [Model Registry](modules/MODELS.md) - Model metadata system

---

**Next:** Explore individual module documentation for implementation details.
