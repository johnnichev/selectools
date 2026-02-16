# Selectools

[![PyPI version](https://badge.fury.io/py/selectools.svg)](https://badge.fury.io/py/selectools)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Production-ready AI agents with tool calling, RAG, and hybrid search.** Connect LLMs to your Python functions, embed and search your documents with vector + keyword fusion, stream responses in real time, and dynamically manage tools at runtime. Works with OpenAI, Anthropic, Gemini, and Ollama. Tracks costs automatically.

## Why Selectools

| Capability | What You Get |
|---|---|
| **Provider Agnostic** | Switch between OpenAI, Anthropic, Gemini, Ollama with one line. Your tools stay identical. |
| **Hybrid Search** | BM25 keyword + vector semantic search with RRF/weighted fusion and cross-encoder reranking. |
| **Advanced Chunking** | Fixed, recursive, semantic (embedding-based), and contextual (LLM-enriched) chunking strategies. |
| **E2E Streaming** | Token-level `astream()` with native tool call support. Parallel tool execution via `asyncio.gather`. |
| **Dynamic Tools** | Load tools from files/directories at runtime. Add, remove, replace tools without restarting. |
| **Response Caching** | LRU + TTL in-memory cache and Redis backend. Avoid redundant LLM calls for identical requests. |
| **Routing Mode** | Agent selects a tool without executing it. Use for intent classification and request routing. |
| **Production Hardened** | Retries with backoff, per-tool timeouts, iteration caps, cost warnings, observability hooks. |
| **Library-First** | Not a framework. No magic globals, no hidden state. Use as much or as little as you need. |

## What's Included

- **4 LLM Providers**: OpenAI, Anthropic, Gemini, Ollama with unified interface
- **4 Embedding Providers**: OpenAI, Anthropic/Voyage, Gemini (free!), Cohere
- **4 Vector Stores**: In-memory, SQLite, Chroma, Pinecone
- **Hybrid Search**: BM25 + vector fusion with Cohere/Jina reranking
- **Advanced Chunking**: Semantic + contextual chunking for better retrieval
- **Dynamic Tool Loading**: Plugin system with hot-reload support
- **Response Caching**: InMemoryCache and RedisCache with stats tracking
- **120 Model Registry**: Type-safe constants with pricing and metadata
- **Pre-built Toolbox**: 22 tools for files, data, text, datetime, web
- **18 Examples**: RAG, hybrid search, streaming, caching, routing, and more
- **400+ Tests**: Unit, integration, and E2E with real API calls

## Install

```bash
pip install selectools                    # Core + basic RAG
pip install selectools[rag]               # + Chroma, Pinecone, Voyage, Cohere, PyPDF
pip install selectools[cache]             # + Redis cache
pip install selectools[rag,cache]         # Everything
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Quick Start

### Tool Calling Agent

```python
from selectools import Agent, AgentConfig, Message, Role, tool, OpenAIProvider
from selectools.models import OpenAI

@tool(description="Search the web for information")
def search(query: str) -> str:
    return f"Results for: {query}"

agent = Agent(
    tools=[search],
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
    config=AgentConfig(max_iterations=5)
)

response = agent.run([Message(role=Role.USER, content="Search for Python tutorials")])
print(response.content)
```

### RAG Agent (3 Lines)

```python
from selectools import OpenAIProvider
from selectools.embeddings import OpenAIEmbeddingProvider
from selectools.models import OpenAI
from selectools.rag import RAGAgent, VectorStore

embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
store = VectorStore.create("memory", embedder=embedder)

agent = RAGAgent.from_directory(
    directory="./docs",
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id),
    vector_store=store,
    chunk_size=500, top_k=3
)

response = agent.run("What are the main features?")
print(response.content)
print(agent.get_usage_summary())  # LLM + embedding costs
```

### Hybrid Search (Keyword + Semantic)

```python
from selectools.rag import BM25, HybridSearcher, FusionMethod, HybridSearchTool, VectorStore

store = VectorStore.create("memory", embedder=embedder)
store.add_documents(chunked_docs)

searcher = HybridSearcher(
    vector_store=store,
    vector_weight=0.6,
    keyword_weight=0.4,
    fusion=FusionMethod.RRF,
)
searcher.add_documents(chunked_docs)

# Use with agent
hybrid_tool = HybridSearchTool(searcher=searcher, top_k=5)
agent = Agent(tools=[hybrid_tool.search_knowledge_base], provider=provider)
```

### Streaming with Parallel Tools

```python
import asyncio
from selectools import Agent, AgentConfig, Message, Role
from selectools.types import StreamChunk, AgentResult

agent = Agent(
    tools=[tool_a, tool_b, tool_c],
    provider=provider,
    config=AgentConfig(parallel_tool_execution=True)  # Default: enabled
)

async for item in agent.astream([Message(role=Role.USER, content="Run all tasks")]):
    if isinstance(item, StreamChunk):
        print(item.content, end="", flush=True)
    elif isinstance(item, AgentResult):
        print(f"\nDone in {item.iterations} iterations")
```

## Key Features

### Hybrid Search & Reranking

Combine semantic search with BM25 keyword matching for better recall on exact terms, names, and acronyms:

```python
from selectools.rag import BM25, HybridSearcher, CohereReranker, FusionMethod

searcher = HybridSearcher(
    vector_store=store,
    fusion=FusionMethod.RRF,
    reranker=CohereReranker(),  # Optional cross-encoder reranking
)
results = searcher.search("GDPR compliance", top_k=5)
```

See [docs/modules/HYBRID_SEARCH.md](docs/modules/HYBRID_SEARCH.md) for full documentation.

### Advanced Chunking

Go beyond fixed-size splitting with embedding-aware and LLM-enriched chunking:

```python
from selectools.rag import SemanticChunker, ContextualChunker

# Split at topic boundaries using embedding similarity
semantic = SemanticChunker(embedder=embedder, similarity_threshold=0.75)

# Enrich each chunk with LLM-generated context (Anthropic-style contextual retrieval)
contextual = ContextualChunker(base_chunker=semantic, provider=provider)
enriched_docs = contextual.split_documents(documents)
```

See [docs/modules/ADVANCED_CHUNKING.md](docs/modules/ADVANCED_CHUNKING.md) for full documentation.

### Dynamic Tool Loading

Discover and load `@tool` functions from files and directories at runtime:

```python
from selectools.tools import ToolLoader

# Load tools from a plugin directory
tools = ToolLoader.from_directory("./plugins", recursive=True)
agent.add_tools(tools)

# Hot-reload after editing a plugin
updated = ToolLoader.reload_file("./plugins/search.py")
agent.replace_tool(updated[0])

# Remove tools the agent no longer needs
agent.remove_tool("deprecated_search")
```

See [docs/modules/DYNAMIC_TOOLS.md](docs/modules/DYNAMIC_TOOLS.md) for full documentation.

### Response Caching

Avoid redundant LLM calls with pluggable caching:

```python
from selectools import Agent, AgentConfig, InMemoryCache

cache = InMemoryCache(max_size=1000, default_ttl=300)
agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(cache=cache)
)

# Same question twice -> second call is instant (cache hit)
agent.run([Message(role=Role.USER, content="What is Python?")])
agent.reset()
agent.run([Message(role=Role.USER, content="What is Python?")])

print(cache.stats)  # CacheStats(hits=1, misses=1, hit_rate=50.00%)
```

For distributed setups: `from selectools.cache_redis import RedisCache`

### Routing Mode

Agent selects a tool without executing it -- use for intent classification:

```python
config = AgentConfig(routing_only=True)
agent = Agent(tools=[send_email, schedule_meeting, search_kb], provider=provider, config=config)

result = agent.run([Message(role=Role.USER, content="Book a meeting with Alice tomorrow")])
print(result.tool_name)  # "schedule_meeting"
print(result.tool_args)  # {"attendee": "Alice", "date": "tomorrow"}
```

### E2E Streaming & Parallel Execution

- `agent.astream()` yields `StreamChunk` (text deltas) then `AgentResult` (final)
- Multiple tool calls execute concurrently via `asyncio.gather()` (3 tools @ 0.15s each = ~0.15s total)
- Fallback chain: `astream` -> `acomplete` -> `complete` via executor
- Context propagation with `contextvars` for tracing/auth

See [docs/modules/STREAMING.md](docs/modules/STREAMING.md) for full documentation.

## Providers

| Provider | Streaming | Vision | Native Tools | Cost |
|---|---|---|---|---|
| **OpenAI** | Yes | Yes | Yes | Paid |
| **Anthropic** | Yes | Yes | Yes | Paid |
| **Gemini** | Yes | Yes | Yes | Free tier |
| **Ollama** | Yes | No | No | Free (local) |
| **Local** | No | No | No | Free (testing) |

```python
from selectools.models import OpenAI, Anthropic, Gemini, Ollama

# IDE autocomplete for all 120 models with pricing metadata
model = OpenAI.GPT_4O_MINI
print(f"Cost: ${model.prompt_cost}/${model.completion_cost} per 1M tokens")
print(f"Context: {model.context_window:,} tokens")
```

## Embedding Providers

```python
from selectools.embeddings import (
    OpenAIEmbeddingProvider,     # text-embedding-3-small/large
    AnthropicEmbeddingProvider,  # Voyage AI (voyage-3, voyage-3-lite)
    GeminiEmbeddingProvider,     # FREE (text-embedding-001/004)
    CohereEmbeddingProvider,     # embed-english-v3.0
)
```

## Vector Stores

```python
from selectools.rag import VectorStore

store = VectorStore.create("memory", embedder=embedder)           # Fast, no persistence
store = VectorStore.create("sqlite", embedder=embedder, db_path="docs.db")  # Persistent
store = VectorStore.create("chroma", embedder=embedder, persist_directory="./chroma")
store = VectorStore.create("pinecone", embedder=embedder, index_name="my-index")
```

## Agent Configuration

```python
config = AgentConfig(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=2000,
    max_iterations=6,
    max_retries=3,
    retry_backoff_seconds=2.0,
    request_timeout=60.0,
    tool_timeout_seconds=30.0,
    cost_warning_threshold=0.50,
    parallel_tool_execution=True,
    routing_only=False,
    stream=False,
    cache=None,                  # InMemoryCache or RedisCache
    enable_analytics=True,
    verbose=False,
    hooks={                      # Lifecycle callbacks
        "on_tool_start": lambda name, args: ...,
        "on_tool_end": lambda name, result, duration: ...,
        "on_llm_end": lambda response, usage: ...,
    },
    system_prompt="You are a helpful assistant...",
)
```

## Tool Definition

### `@tool` Decorator (Recommended)

```python
from selectools import tool

@tool(description="Calculate compound interest")
def calculate_interest(principal: float, rate: float, years: int) -> str:
    amount = principal * (1 + rate / 100) ** years
    return f"After {years} years: ${amount:.2f}"
```

### Tool Registry

```python
from selectools import ToolRegistry

registry = ToolRegistry()

@registry.tool(description="Search the knowledge base")
def search_kb(query: str, max_results: int = 5) -> str:
    return f"Results for: {query}"

agent = Agent(tools=registry.all(), provider=provider)
```

### Injected Parameters

Keep secrets out of the LLM's view:

```python
db_tool = Tool(
    name="query_db",
    description="Execute SQL query",
    parameters=[ToolParameter(name="sql", param_type=str, description="SQL query")],
    function=query_database,
    injected_kwargs={"db_connection": db_conn}  # Hidden from LLM
)
```

### Streaming Tools

```python
from typing import Generator

@tool(description="Process large file", streaming=True)
def process_file(filepath: str) -> Generator[str, None, None]:
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            yield f"[Line {i}] {line.strip()}\n"

config = AgentConfig(hooks={"on_tool_chunk": lambda name, chunk: print(chunk, end="")})
```

## Conversation Memory

```python
from selectools import Agent, ConversationMemory

memory = ConversationMemory(max_messages=20)
agent = Agent(tools=[...], provider=provider, memory=memory)

agent.run([Message(role=Role.USER, content="My name is Alice")])
agent.run([Message(role=Role.USER, content="What's my name?")])  # Remembers "Alice"
```

## Cost Tracking

```python
response = agent.run([Message(role=Role.USER, content="Search and summarize")])

print(f"Total cost: ${agent.total_cost:.6f}")
print(f"Total tokens: {agent.total_tokens:,}")
print(agent.get_usage_summary())
# Includes LLM + embedding costs, per-tool breakdown
```

## Examples

| Example | Features Demonstrated |
|---|---|
| `rag_basic_demo.py` | RAG pipeline, document loading, vector search |
| `hybrid_search_demo.py` | BM25 + vector fusion, RRF/weighted, reranking |
| `advanced_chunking_demo.py` | Semantic and contextual chunking comparison |
| `dynamic_tools_demo.py` | ToolLoader, plugin directory, hot-reload |
| `caching_demo.py` | InMemoryCache, cache stats, agent reset |
| `routing_mode_demo.py` | Routing mode, AgentResult, intent classification |
| `streaming_parallel_demo.py` | astream(), parallel execution, StreamChunk |
| `streaming_tools_demo.py` | Generator-based streaming tools |
| `async_agent_demo.py` | arun(), concurrent agents, FastAPI pattern |
| `conversation_memory_demo.py` | Multi-turn conversations with memory |
| `cost_tracking_demo.py` | Token counting, cost warnings |
| `toolbox_demo.py` | Pre-built tools (file, data, text, datetime) |
| `customer_support_bot.py` | Multi-tool customer support workflow |
| `data_analysis_agent.py` | Data exploration and analysis tools |
| `rag_advanced_demo.py` | PDFs, SQLite persistence, metadata filtering |
| `rag_multi_provider_demo.py` | Embedding/store/chunk-size comparisons |
| `semantic_search_demo.py` | Pure semantic search, metadata filtering |
| `ollama_demo.py` | Local LLM execution with Ollama |

Run any example:

```bash
python examples/hybrid_search_demo.py
```

## Documentation

Comprehensive technical documentation is available in [`docs/`](docs/README.md):

| Module | Description |
|---|---|
| [AGENT](docs/modules/AGENT.md) | Agent loop, retry logic, caching, hooks |
| [STREAMING](docs/modules/STREAMING.md) | E2E streaming, parallel execution, routing |
| [TOOLS](docs/modules/TOOLS.md) | Tool definition, validation, registry |
| [DYNAMIC_TOOLS](docs/modules/DYNAMIC_TOOLS.md) | ToolLoader, plugins, hot-reload |
| [HYBRID_SEARCH](docs/modules/HYBRID_SEARCH.md) | BM25, fusion, reranking |
| [ADVANCED_CHUNKING](docs/modules/ADVANCED_CHUNKING.md) | Semantic & contextual chunking |
| [RAG](docs/modules/RAG.md) | Complete RAG pipeline |
| [EMBEDDINGS](docs/modules/EMBEDDINGS.md) | Embedding providers |
| [VECTOR_STORES](docs/modules/VECTOR_STORES.md) | Storage backends |
| [PROVIDERS](docs/modules/PROVIDERS.md) | LLM provider adapters |
| [MEMORY](docs/modules/MEMORY.md) | Conversation memory |
| [USAGE](docs/modules/USAGE.md) | Cost tracking & analytics |
| [MODELS](docs/modules/MODELS.md) | Model registry & pricing |
| [PARSER](docs/modules/PARSER.md) | Tool call parsing |
| [PROMPT](docs/modules/PROMPT.md) | System prompt generation |

## Tests

```bash
pytest tests/ -x -q          # All tests
pytest tests/ -k "not e2e"   # Skip E2E (no API keys needed)
```

400+ tests covering parsing, agent loop, providers, RAG pipeline, hybrid search, advanced chunking, dynamic tools, caching, streaming, and E2E integration.

## License

**LGPL-3.0-or-later** - Use freely in commercial applications. Only modifications to the library itself must be shared. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome contributions for new tools, providers, vector stores, examples, and documentation.

---

[Roadmap](ROADMAP.md) | [Changelog](CHANGELOG.md) | [Documentation](docs/README.md) | [Feature Proposals](FEATURE_PROPOSALS.md)
