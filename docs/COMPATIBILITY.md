# Compatibility Matrix

**Added in:** v0.19.2
**Updated:** v0.25.0

This page documents the tested combinations of Python versions, provider SDKs, and optional
dependencies. All combinations in the CI matrix are validated on every commit.

---

## Python Versions

| Python | Status | CI tested |
| ------ | ------ | --------- |
| 3.9    | ✅ Supported | Yes |
| 3.10   | ✅ Supported | Yes |
| 3.11   | ✅ Supported | Yes |
| 3.12   | ✅ Supported | Yes |
| 3.13   | ✅ Supported | Yes |
| < 3.9  | ❌ Not supported | — |

!!! warning "Python 3.9 support ends at v1.0.0"
    Python 3.9 reached upstream end-of-life in October 2025. selectools will drop
    Python 3.9 support at the **v1.0.0** release; the minimum will become Python 3.10.
    All 0.x releases (including the final 0.24.x series) continue to support 3.9.
    See [MIGRATION_1.0](MIGRATION_1.0.md) for details.

---

## Operating Systems

| OS | Status |
|----|--------|
| Linux (Ubuntu latest) | ✅ CI-tested (full 3.9–3.13 matrix) |
| macOS | ✅ Tested (dev environment) |
| Windows | ⚠️ Not CI-tested — likely works, not guaranteed |

---

## Core Dependencies

These are always installed with `pip install selectools`.

| Package | Required version | Notes |
|---------|-----------------|-------|
| `openai` | `>=1.30.0, <2.0.0` | OpenAI provider + all OpenAI-compatible endpoints (Ollama, Azure) |
| `anthropic` | `>=0.28.0, <1.0.0` | Anthropic provider |
| `google-genai` | `>=1.0.0` | Gemini provider |
| `numpy` | `>=1.24.0, <3.0.0` | Vector operations in RAG |

---

## Optional Dependencies

Install extras as needed: `pip install "selectools[rag,evals,mcp,litellm,postgres,supabase,toolbox,observe,serve]"`

### `[rag]` — Vector stores, embeddings, and loaders

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `chromadb` | `>=0.4.0` | ChromaVectorStore |
| `pinecone-client` | `>=3.0.0` | PineconeVectorStore |
| `qdrant-client` | `>=1.7.0` | QdrantVectorStore |
| `faiss-cpu` | `>=1.7.0` | FAISSVectorStore |
| `voyageai` | `>=0.2.0` | Voyage embeddings (Anthropic's recommended embedding partner) |
| `cohere` | `>=5.0.0` | Cohere embeddings + reranker |
| `pypdf` | `>=4.0.0` | PDFLoader |
| `beautifulsoup4` | `>=4.12.0` | HTML loading/extraction in document loaders |

### `[evals]` — Eval framework

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `pyyaml` | `>=6.0.0` | YAML dataset loading in EvalSuite |

### `[mcp]` — Model Context Protocol

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `mcp` | `>=1.0.0, <2.0.0` | MCPClient, MCPServer, mcp_tools() |

### `[litellm]` — LiteLLM meta-provider

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `litellm` | `>=1.0.0` | LiteLLMProvider (100+ models through one interface) |

### `[postgres]` — PostgreSQL backends

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `psycopg2-binary` | `>=2.9.0` | PostgresCheckpointStore, PgVectorStore |

### `[supabase]` — Supabase backends

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `supabase` | `>=2.0.0` | SupabaseSessionStore, SupabaseKnowledgeStore |

### `[toolbox]` — Built-in tool integrations

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `requests` | `>=2.28.0` | HTTP tools (web requests) |
| `slack-sdk` | `>=3.27.0` | Slack tools (send/read/search messages) |
| `pdfplumber` | `>=0.11.0` | PDF extraction tools |

### `[observe]` — Observability integrations

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `opentelemetry-api` | `>=1.20.0` | OTelObserver (OpenTelemetry spans) |
| `langfuse` | `>=2.0.0` | LangfuseObserver (Langfuse tracing) |

### `[serve]` — Agent serving

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `pyyaml` | `>=6.0.0` | YAML agent config (`selectools serve agent.yaml`) |
| `starlette` | `>=0.27.0` | AgentAPI / A2AServer ASGI apps |
| `uvicorn[standard]` | `>=0.24.0` | Built-in ASGI server for `selectools serve` |
| `httpx` | `>=0.24.0` | A2AClient and HTTP transport |

---

## Providers

Nine providers ship with the core package (no extra needed unless noted):

| Provider | Class | Requires |
|----------|-------|----------|
| OpenAI | `OpenAIProvider` | core |
| Azure OpenAI | `AzureOpenAIProvider` | core |
| Anthropic | `AnthropicProvider` | core |
| Google Gemini | `GeminiProvider` | core |
| Ollama | `OllamaProvider` | core (Ollama server running locally) |
| LiteLLM | `LiteLLMProvider` | `[litellm]` extra |
| Fallback chain | `FallbackProvider` | core |
| Router | `RouterProvider` | core |
| Local stub | `LocalProvider` | core (testing/offline) |

## Vector Stores

Seven vector store backends:

| Store | Requires |
|-------|----------|
| `InMemoryVectorStore` | core (stdlib + numpy) |
| `SQLiteVectorStore` | core (stdlib) |
| `ChromaVectorStore` | `[rag]` (`chromadb`) |
| `PineconeVectorStore` | `[rag]` (`pinecone-client`) |
| `QdrantVectorStore` | `[rag]` (`qdrant-client`) |
| `FAISSVectorStore` | `[rag]` (`faiss-cpu`) |
| `PgVectorStore` | `[postgres]` (`psycopg2-binary` + pgvector extension) |

## Session and Checkpoint Backends

| Backend | Type | Requires |
|---------|------|----------|
| `JsonFileSessionStore` | sessions | core (stdlib) |
| `SQLiteSessionStore` | sessions | core (stdlib) |
| `RedisSessionStore` | sessions | `redis>=4.0.0` (not a declared extra — install manually) |
| `SupabaseSessionStore` | sessions | `[supabase]` |
| `InMemoryCheckpointStore` | checkpoints | core |
| `FileCheckpointStore` | checkpoints | core (stdlib) |
| `SQLiteCheckpointStore` | checkpoints | core (stdlib) |
| `PostgresCheckpointStore` | checkpoints | `[postgres]` |

---

## Provider SDK Compatibility

### OpenAI

| `openai` SDK | `selectools` | Notes |
|-------------|-------------|-------|
| `1.30.x` – `1.x` | ✅ | Full support |
| `2.0+` | ❌ | Breaking API changes expected |

### Anthropic

| `anthropic` SDK | `selectools` | Notes |
|----------------|-------------|-------|
| `0.28.x` – `0.x` | ✅ | Full support |
| `1.0+` | ❌ | Not yet validated |

### Google Gemini

| `google-genai` SDK | `selectools` | Notes |
|-------------------|-------------|-------|
| `1.0.0+` | ✅ | Full support |

#### Model note: function calling on `gemini-2.5-flash-lite`

`gemini-2.5-flash-lite` supports function calling, but it is markedly less reliable at it
than `gemini-2.5-flash`. It is known to intermittently return an **empty candidate**
(often `finish_reason=MALFORMED_FUNCTION_CALL` or `UNEXPECTED_TOOL_CALL`) instead of a
function call — see [issue #66](https://github.com/johnnichev/selectools/issues/66) and
upstream reports (e.g. [litellm#16651](https://github.com/BerriAI/litellm/issues/16651),
[deepagents#417](https://github.com/langchain-ai/deepagents/issues/417)). When this
happens inside an `Agent` loop, the iteration produces neither text nor a tool call.

`GeminiProvider` logs a `WARNING` (logger `selectools.providers.gemini_provider`)
whenever a tool-equipped response contains neither text nor tool calls, including the
`finish_reason`. If you see this warning repeatedly with `gemini-2.5-flash-lite`,
switch to `gemini-2.5-flash`, simplify your tool schemas, or reduce the number of tools
per request.

`GeminiProvider` also sanitizes tool schemas for Gemini's API, which rejects two shapes
that other providers accept (both were hard 400 errors before sanitization):

- bare `list` parameters (`{"type": "array"}` without `items`) — a permissive
  `items: {"type": "string"}` is injected
- `Dict[K, V]` parameters (`additionalProperties`) — stripped; the parameter degrades
  to a plain `object`

### Ollama

Ollama is accessed via the OpenAI-compatible API (`OllamaProvider` inherits `_OpenAICompatProvider`).
No extra package required. Tested with Ollama server `0.3.x+`.

### LiteLLM

`LiteLLMProvider` routes through the `litellm` package (`[litellm]` extra) and inherits
its model coverage. Compatibility with individual upstream models follows LiteLLM's own
support matrix.

---

## Optional Backend Compatibility

| Backend | Package | Min version | Notes |
|---------|---------|------------|-------|
| Redis (cache/sessions/knowledge) | `redis` | `>=4.0.0` | Not a declared extra — install manually |
| Supabase (sessions/knowledge) | `supabase` | `>=2.0.0` | Declared extra: `selectools[supabase]` |
| PostgreSQL (checkpoints/pgvector) | `psycopg2-binary` | `>=2.9.0` | Declared extra: `selectools[postgres]` |
| SQLite (sessions/checkpoints/vectors) | stdlib | — | Always available |
| In-memory (cache/vectors/checkpoints) | stdlib | — | Always available |

---

## What "Supported" Means

- **✅ Supported**: Tested in CI on every commit. Bugs in this combination are treated as P0.
- **⚠️ Best-effort**: Not in CI but expected to work. Bug reports accepted; fixes on a best-effort basis.
- **❌ Not supported**: Known to be incompatible or untested. No guarantees.

---

## Reporting Incompatibilities

If you discover a combination that should work but doesn't, please open an issue at
[github.com/johnnichev/selectools/issues](https://github.com/johnnichev/selectools/issues)
with your Python version, OS, and the output of `selectools doctor`.
