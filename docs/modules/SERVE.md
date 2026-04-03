---
description: "selectools serve: HTTP API + playground + builder deployment"
tags:
  - deployment
  - serve
---

# Serve Module

**Import:** `from selectools.serve.app import create_app`
**Stability:** beta

```python title="serve_quick.py"
from selectools import Agent, tool
from selectools.providers.stubs import LocalProvider
from selectools.serve.app import create_app

@tool(description="Greet a user by name")
def greet(name: str) -> str:
    return f"Hello, {name}!"

agent = Agent(tools=[greet], provider=LocalProvider())
app = create_app(agent, playground=True)
app.serve(port=8000)
```

!!! tip "See Also"
    - [Visual Agent Builder](builder.md) -- drag-drop graph editor served at `/builder`
    - [Templates Module](TEMPLATES.md) -- YAML config and pre-built agent templates
    - [Agent Module](AGENT.md) -- the `Agent` class that powers the server

**Added in:** v0.19.0
**Package:** `src/selectools/serve/`
**Classes:** `AgentRouter`, `AgentServer`
**Functions:** `create_app()`

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [CLI Commands](#cli-commands)
4. [Endpoints](#endpoints)
5. [Streaming (SSE)](#streaming-sse)
6. [Playground UI](#playground-ui)
7. [Python API](#python-api)
8. [FastAPI Integration](#fastapi-integration)
9. [Flask Integration](#flask-integration)
10. [Configuration Options](#configuration-options)
11. [Request / Response Models](#request-response-models)
12. [API Reference](#api-reference)
13. [Examples](#examples)

---

## Overview

The **serve** module turns any selectools `Agent` into an HTTP API with one command. No framework boilerplate, no config files, no Docker -- just `selectools serve agent.yaml` and you have a live endpoint with streaming, a health check, tool schema introspection, and an interactive playground UI.

### Why Serve?

| | selectools serve | Manual FastAPI setup |
|---|---|---|
| **Lines of code** | 1 CLI command or 3 lines of Python | 40+ lines minimum |
| **Dependencies** | Zero (stdlib `http.server`) | fastapi, uvicorn, pydantic |
| **Streaming** | SSE built-in | Manual SSE wiring |
| **Playground** | Built-in chat UI at `/playground` | Build your own |
| **Schema** | Auto-generated from tools | Manual OpenAPI spec |

### Design Philosophy

- **Zero dependencies.** The built-in server uses Python's stdlib `http.server`. No FastAPI, no Flask, no uvicorn required.
- **Production-ready integrations.** When you outgrow the built-in server, `AgentRouter` drops into FastAPI or Flask with 3 lines of code.
- **Config-driven.** Load agents from YAML files or built-in templates. No Python code required for common configurations.

---

## Quick Start

### One Command

```bash
# Serve from a YAML config
selectools serve agent.yaml

# Serve a built-in template
selectools serve customer_support

# Customize host and port
selectools serve agent.yaml --port 3000 --host 127.0.0.1

# Disable the playground UI
selectools serve agent.yaml --no-playground
```

### Three Lines of Python

```python
from selectools.serve import create_app

app = create_app(agent, playground=True)
app.serve(port=8000)
```

The server prints its endpoints on startup:

```
Selectools agent serving at http://0.0.0.0:8000
  POST /invoke   -- single prompt
  POST /stream   -- SSE streaming
  GET  /health   -- health check
  GET  /schema   -- tool schemas
  GET  /playground -- chat UI

Press Ctrl+C to stop.
```

---

## CLI Commands

### `selectools serve`

Start an agent HTTP server from a YAML config file or template name.

```bash
selectools serve <config> [--port PORT] [--host HOST] [--no-playground]
```

| Argument | Default | Description |
|---|---|---|
| `config` | (required) | Path to YAML config file, or a template name (`customer_support`, `data_analyst`, etc.). |
| `--port` | `8000` | Port number. |
| `--host` | `0.0.0.0` | Bind address. Use `127.0.0.1` for local-only. |
| `--no-playground` | `False` | Disable the playground chat UI. |

When `config` is a template name (e.g. `customer_support`), the CLI auto-detects an API key from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`) and creates the provider automatically.

### `selectools doctor`

Diagnose API keys, optional dependencies, and provider connectivity.

```bash
selectools doctor
```

Output:

```
Selectools Doctor
========================================
Version: 0.19.0
Python: 3.12.0

API Keys:
  OPENAI_API_KEY: OK
  ANTHROPIC_API_KEY: MISSING
  GOOGLE_API_KEY: MISSING
  GEMINI_API_KEY: MISSING

Optional Dependencies:
  fastapi: OK (FastAPI serving)
  flask: not installed (Flask serving)
  redis: OK (Redis cache/sessions)
  chromadb: not installed (Chroma vector store)
  ...

Provider Connectivity:
  OpenAI: OK (connected)
  Anthropic: skipped (no key)
  Gemini: skipped (no key)

Diagnosis complete.
```

---

## Endpoints

### POST /invoke

Send a single prompt and receive a JSON response.

**Request:**

```json
{
  "prompt": "What is the capital of France?"
}
```

**Response:**

```json
{
  "content": "The capital of France is Paris.",
  "tool_calls": [],
  "reasoning": null,
  "iterations": 1,
  "tokens": 42,
  "cost_usd": 0.00012,
  "run_id": "run-abc123"
}
```

**cURL example:**

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

### POST /stream

Send a prompt and receive an SSE (Server-Sent Events) stream. Each event is a JSON object with a `type` field.

**Request:** Same as `/invoke`.

**Response stream:**

```
data: {"type": "chunk", "content": "The capital"}
data: {"type": "chunk", "content": " of France"}
data: {"type": "chunk", "content": " is Paris."}
data: {"type": "result", "content": "The capital of France is Paris.", "iterations": 1}
data: [DONE]
```

### GET /health

Health check endpoint. Returns agent status, version, model, provider, and available tools.

**Response:**

```json
{
  "status": "ok",
  "version": "0.19.0",
  "model": "gpt-4o",
  "provider": "openai",
  "tools": ["read_file", "write_file", "web_search"]
}
```

### GET /schema

Returns JSON schemas for all tools registered with the agent.

**Response:**

```json
{
  "model": "gpt-4o",
  "tools": [
    {
      "name": "read_file",
      "description": "Read a file from disk",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "File path to read"}
        },
        "required": ["path"]
      }
    }
  ]
}
```

### GET /playground

Interactive chat UI served as a single HTML page. See [Playground UI](#playground-ui) below.

### GET /

Redirects to `/playground` when the playground is enabled.

---

## Streaming (SSE)

The `/stream` endpoint uses Server-Sent Events for real-time token streaming. The agent's `astream()` method powers this -- each token chunk is forwarded as an SSE event.

### Event Types

| Type | Description |
|---|---|
| `chunk` | A text fragment from the LLM. Concatenate all chunks for the full response. |
| `result` | Final result with content, iteration count. Sent once at the end. |
| `[DONE]` | Stream termination signal. |

### JavaScript Client

```javascript
const response = await fetch("http://localhost:8000/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt: "Explain quantum computing" }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  for (const line of text.split("\n")) {
    if (line.startsWith("data: ") && line !== "data: [DONE]") {
      const event = JSON.parse(line.slice(6));
      if (event.type === "chunk") {
        process.stdout.write(event.content);
      }
    }
  }
}
```

---

## Playground UI

When enabled (default), the server serves an interactive chat interface at `/playground`. The playground is a single self-contained HTML page with no external dependencies.

### Features

- Real-time streaming responses via SSE
- Conversation history within the session
- Tool call visibility (shows which tools the agent invoked)
- Model and provider info displayed in the header
- Works in any modern browser

The playground is intended for development and testing. For production UIs, build a custom frontend against the `/invoke` and `/stream` endpoints.

### Disabling

```bash
# CLI
selectools serve agent.yaml --no-playground

# Python
app = create_app(agent, playground=False)
```

---

## Python API

### AgentRouter

The `AgentRouter` class handles request routing and is the core building block for all integrations. It works standalone or embedded in any WSGI/ASGI framework.

```python
from selectools.serve import AgentRouter

router = AgentRouter(agent, prefix="/api/v1", enable_playground=True)

# Use handler methods directly
result = router.handle_invoke({"prompt": "Hello"})
health = router.handle_health()
schema = router.handle_schema()
```

### create_app()

Create a standalone HTTP server with zero dependencies:

```python
from selectools.serve import create_app

app = create_app(
    agent,
    prefix="",           # URL prefix for all endpoints
    playground=True,      # Enable /playground UI
    host="0.0.0.0",      # Bind address
    port=8000,            # Port number
)

app.serve()  # Blocking -- starts the server
```

---

## FastAPI Integration

Drop `AgentRouter` into a FastAPI application for production deployments:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from selectools.serve import AgentRouter

app = FastAPI()
router = AgentRouter(agent)

@app.post("/invoke")
async def invoke(request: Request):
    body = await request.json()
    return JSONResponse(router.handle_invoke(body))

@app.post("/stream")
async def stream(request: Request):
    body = await request.json()
    return StreamingResponse(
        router.handle_stream(body),
        media_type="text/event-stream",
    )

@app.get("/health")
async def health():
    return JSONResponse(router.handle_health())
```

Run with uvicorn for production-grade performance:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Flask Integration

```python
from flask import Flask, request, jsonify, Response
from selectools.serve import AgentRouter

app = Flask(__name__)
router = AgentRouter(agent)

@app.route("/invoke", methods=["POST"])
def invoke():
    return jsonify(router.handle_invoke(request.json))

@app.route("/stream", methods=["POST"])
def stream():
    return Response(
        router.handle_stream(request.json),
        content_type="text/event-stream",
    )

@app.route("/health")
def health():
    return jsonify(router.handle_health())
```

---

## Configuration Options

### YAML Config File

The recommended way to configure a served agent. See the [Templates Module](TEMPLATES.md) for full YAML reference.

```yaml
provider: openai
model: gpt-4o
system_prompt: "You are a helpful coding assistant."
tools:
  - selectools.toolbox.file_tools.read_file
  - selectools.toolbox.file_tools.write_file
  - ./my_custom_tool.py
budget:
  max_cost_usd: 1.00
retry:
  max_retries: 3
```

### Environment Variables

The CLI auto-detects providers from environment variables:

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI (checked first) |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Gemini |

---

## Request / Response Models

**File:** `src/selectools/serve/models.py`

### InvokeRequest

| Field | Type | Description |
|---|---|---|
| `prompt` | `str` | The user prompt. |
| `config_overrides` | `Optional[Dict[str, Any]]` | Override agent config for this request. |

### InvokeResponse

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Agent response text. |
| `tool_calls` | `List[Dict]` | Tools invoked during execution. |
| `reasoning` | `Optional[str]` | Reasoning trace (when using CoT/ReAct strategies). |
| `iterations` | `int` | Number of agent loop iterations. |
| `tokens` | `int` | Total tokens consumed. |
| `cost_usd` | `float` | Estimated cost in USD. |
| `run_id` | `str` | Unique run identifier for trace lookup. |

### HealthResponse

| Field | Type | Description |
|---|---|---|
| `status` | `str` | Always `"ok"` when healthy. |
| `version` | `str` | Selectools version. |
| `model` | `str` | Active model name. |
| `provider` | `str` | Active provider name. |
| `tools` | `List[str]` | Names of registered tools. |

---

## API Reference

### AgentRouter.__init__()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent` | `Agent` | (required) | The agent to serve. |
| `prefix` | `str` | `""` | URL prefix for all endpoints (e.g. `"/api/v1"`). |
| `enable_playground` | `bool` | `True` | Enable the `/playground` chat UI. |

### AgentRouter Methods

| Method | Description |
|---|---|
| `handle_invoke(body)` | Process a POST /invoke request. Returns response dict. |
| `handle_stream(body)` | Process a POST /stream request. Yields SSE-formatted strings. |
| `handle_health()` | Process a GET /health request. Returns health dict. |
| `handle_schema()` | Process a GET /schema request. Returns tool schemas dict. |

### create_app()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent` | `Agent` | (required) | The agent to serve. |
| `prefix` | `str` | `""` | URL prefix for all endpoints. |
| `playground` | `bool` | `True` | Enable the `/playground` chat UI. |
| `host` | `str` | `"0.0.0.0"` | Bind address. |
| `port` | `int` | `8000` | Port number. |

Returns an `AgentServer` instance. Call `.serve()` to start (blocking).

### AgentServer Methods

| Method | Description |
|---|---|
| `serve(port=None)` | Start the HTTP server. Blocking. Uses stdlib `http.server`. |

---

## Examples

| Example | File | Description |
|---|---|---|
| 64 | [`64_selectools_serve.py`](https://github.com/johnnichev/selectools/blob/main/examples/64_selectools_serve.py) | Serve an agent with the built-in server |
| 62 | [`62_yaml_config.py`](https://github.com/johnnichev/selectools/blob/main/examples/62_yaml_config.py) | Load an agent from YAML config |

---

## Related Examples

| # | File | Description |
|---|------|-------------|
| 64 | [`64_selectools_serve.py`](https://github.com/johnnichev/selectools/blob/main/examples/64_selectools_serve.py) | Serve an agent with the built-in HTTP server |
| 62 | [`62_yaml_config.py`](https://github.com/johnnichev/selectools/blob/main/examples/62_yaml_config.py) | Load an agent from YAML and serve it |
| 63 | [`63_agent_templates.py`](https://github.com/johnnichev/selectools/blob/main/examples/63_agent_templates.py) | Use built-in templates with the serve module |

---

## Further Reading

- [Templates Module](TEMPLATES.md) -- YAML config format and pre-built templates
- [Trace Store Module](TRACE_STORE.md) -- Persist and query execution traces
- [Agent Module](AGENT.md) -- The Agent class that powers the server
- [Streaming Module](STREAMING.md) -- How streaming works under the hood

---

**Next Steps:** Learn about YAML configuration and pre-built templates in the [Templates Module](TEMPLATES.md).
