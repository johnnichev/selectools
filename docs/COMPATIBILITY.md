# Compatibility Matrix

**Added in:** v0.19.2
**Updated:** v0.19.2

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

---

## Operating Systems

| OS | Status |
|----|--------|
| Linux (Ubuntu latest) | ✅ CI-tested |
| macOS | ✅ Tested (dev environment) |
| Windows | ⚠️ Not CI-tested — likely works, not guaranteed |

---

## Core Dependencies

These are always installed with `pip install selectools`.

| Package | Required version | Notes |
|---------|-----------------|-------|
| `openai` | `>=1.30.0, <2.0.0` | OpenAI provider + all OpenAI-compatible endpoints |
| `anthropic` | `>=0.28.0, <1.0.0` | Anthropic provider |
| `google-genai` | `>=1.0.0` | Gemini provider |
| `numpy` | `>=1.24.0, <3.0.0` | Vector operations in RAG |

---

## Optional Dependencies

Install extras as needed: `pip install selectools[rag,evals,mcp,postgres,serve]`

### `[rag]` — Vector stores and embeddings

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `chromadb` | `>=0.4.0` | ChromaVectorStore |
| `pinecone-client` | `>=3.0.0` | PineconeVectorStore |
| `voyageai` | `>=0.2.0` | VoyageEmbedding provider |
| `cohere` | `>=5.0.0` | CohereEmbedding provider + reranker |
| `pypdf` | `>=4.0.0` | PDFLoader |

### `[evals]` — Eval framework

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `pyyaml` | `>=6.0.0` | YAML dataset loading in EvalSuite |

### `[mcp]` — Model Context Protocol

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `mcp` | `>=1.0.0, <2.0.0` | MCPClient, MCPServer, mcp_tools() |

### `[postgres]` — PostgreSQL backends

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `psycopg2-binary` | `>=2.9.0` | PostgresCheckpointStore, PostgresSessionStore |

### `[serve]` — Agent serving

| Package | Required version | What it enables |
|---------|-----------------|-----------------|
| `pyyaml` | `>=6.0.0` | YAML agent config (`selectools serve agent.yaml`) |

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

---

## Optional Backend Compatibility

| Backend | Package | Min version | Notes |
|---------|---------|------------|-------|
| Redis (cache/sessions/knowledge) | `redis` | `>=4.0.0` | Not a declared extra — install manually |
| Supabase (knowledge store) | `supabase` | `>=2.0.0` | Not a declared extra — install manually |
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
