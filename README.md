# AI Tool Calling - Selectools

[![PyPI version](https://badge.fury.io/py/selectools.svg)](https://badge.fury.io/py/selectools)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Build AI agents that can call your custom Python functions.** Selectools lets you connect LLMs (OpenAI, Anthropic, Gemini) to your own tools and functions. Define what your agent can do—search databases, call APIs, process images, or anything else—and let the LLM decide when and how to use them. Works with any provider, handles errors gracefully, and includes streaming support.

## Why This Library Stands Out

### 🎯 **True Provider Agnosticism**

Unlike other tool-calling libraries that lock you into a single provider's API, this library provides a unified interface across OpenAI, Anthropic, Gemini, and local providers. Switch providers with a single line change—no refactoring required. Your tool definitions remain identical regardless of the backend.

### 🛡️ **Production-Ready Robustness**

Built for real-world reliability:

- **Hardened parser** that handles malformed JSON, fenced code blocks, and mixed content (not just perfect API responses)
- **Automatic retry logic** with exponential backoff for rate limits and transient failures
- **Per-tool execution timeouts** to prevent runaway operations
- **Request-level timeouts** to avoid hanging on slow providers
- **Iteration caps** to control agent loop costs and prevent infinite loops

### 🔧 **Developer-First Ergonomics**

- **`@tool` decorator** with automatic schema inference from Python type hints
- **`ToolRegistry`** for organizing and discovering tools
- **Injected kwargs** for clean separation of user parameters and configuration (API keys, database connections, etc.)
- **Zero boilerplate**: Define a function, add a decorator, done

### 🎨 **Vision + Streaming Support**

- Native vision support for providers that offer it (OpenAI GPT-4o, etc.)
- Real-time streaming with callback handlers for responsive UIs
- Unified API for both streaming and one-shot responses

### 🧪 **Testing-Friendly Architecture**

- **Local provider** for offline development and testing (no API calls, no costs)
- **Mock injection** for deterministic testing (e.g., `SELECTOOLS_BBOX_MOCK_JSON`)
- **Fake providers** included for unit testing your agent logic
- Clean separation of concerns makes components easy to test in isolation

### 📦 **Library-First Design**

Not a framework that takes over your application—a library that integrates into your existing code. Use as much or as little as you need. No magic globals, no hidden state, no framework lock-in.

## What's Included

- Core package at `src/selectools/` with agent loop, parser, prompt builder, and provider adapters
- Providers: OpenAI plus Anthropic/Gemini/Local sharing the same interface
- Library-first examples (see below) and tests with fake providers for schemas, parsing, agent wiring
- PyPI-ready metadata (`pyproject.toml`) using a src-layout package

## Install

### From PyPI (Recommended)

```bash
pip install selectools
```

With optional provider dependencies:

```bash
pip install selectools[providers]  # Includes Anthropic and Gemini
```

### From Source (Development)

```bash
git clone https://github.com/johnnichev/selectools.git
cd selectools
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
```

### Set API Keys

```bash
export OPENAI_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-api-key-here"  # Optional
export GEMINI_API_KEY="your-api-key-here"     # Optional
```

## Usage (Library)

```python
from selectools import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from selectools.providers.openai_provider import OpenAIProvider

# Define a tool
search_tool = Tool(
    name="search",
    description="Search the web",
    parameters=[ToolParameter(name="query", param_type=str, description="query")],
    function=lambda query: f"Results for {query}",
)

# Use typed model constants for autocomplete
from selectools.models import OpenAI
provider = OpenAIProvider(default_model=OpenAI.GPT_4O.id)
agent = Agent(tools=[search_tool], provider=provider, config=AgentConfig(max_iterations=4))
response = agent.run([Message(role=Role.USER, content="Search for Backtrack")])
print(response.content)
```

## Common ways to use it (library-first)

- Define tools (`Tool` or `@tool`/`ToolRegistry`), pick a provider, run `Agent.run([...])`.
- Add vision by supplying `image_path` on `Message` when the provider supports it.
- For offline/testing: use the Local provider and/or `SELECTOOLS_BBOX_MOCK_JSON=tests/fixtures/bbox_mock.json`.
- Optional dev helpers (not required for library use): `scripts/smoke_cli.py` for quick provider smokes; `scripts/chat.py` for the vision demo.

## Providers (incl. vision & limits)

- **OpenAI**: streaming; vision via Chat Completions `image_url` (e.g., `gpt-4o`); request timeout default 30s; retries/backoff via `AgentConfig`.
- **Anthropic**: streaming; vision model-dependent; set `ANTHROPIC_API_KEY`.
- **Gemini**: streaming; vision model-dependent; set `GEMINI_API_KEY`.
- **Ollama** (v0.6.0): local LLM execution; zero cost; privacy-first; supports llama3.2, mistral, codellama, etc.
- **Local**: no network; echoes latest user text; no vision.
- Rate limits: agent detects `rate limit`/`429` and backs off + retries.
- Timeouts: `AgentConfig.request_timeout` (provider) and `tool_timeout_seconds` (per tool).

## Agent config at a glance

- Core: `model`, `temperature`, `max_tokens`, `max_iterations`.
- Reliability: `max_retries`, `retry_backoff_seconds`, rate-limit backoff, `request_timeout`.
- Execution safety: `tool_timeout_seconds` to bound tool runtime.
- Streaming: `stream=True` to stream provider deltas; optional `stream_handler` callback.
- Analytics (v0.6.0): `enable_analytics=True` to track tool usage metrics, success rates, and performance.
- Observability (v0.5.2): `hooks` dict for lifecycle callbacks (`on_tool_start`, `on_llm_end`, etc.).

## Model Selection with Autocomplete

Use typed model constants for IDE autocomplete and type safety:

```python
from selectools import Agent, OpenAIProvider
from selectools.models import OpenAI, Anthropic, Gemini, Ollama

# IDE suggests all available models when you type OpenAI.
provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)

# Or use in AgentConfig
agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(model=OpenAI.GPT_4O.id)
)

# Access model metadata
model_info = OpenAI.GPT_4O
print(f"Cost: ${model_info.prompt_cost}/${model_info.completion_cost} per 1M tokens")
print(f"Context: {model_info.context_window:,} tokens")
print(f"Max output: {model_info.max_tokens:,} tokens")
```

**Available model classes:**

- `OpenAI` - 65 models (GPT-5, GPT-4o, o-series, GPT-4, GPT-3.5, etc.)
- `Anthropic` - 18 models (Claude 4.5, 4.1, 4, 3.7, 3.5, 3)
- `Gemini` - 26 models (Gemini 3, 2.5, 2.0, 1.5, 1.0, Gemma)
- `Ollama` - 13 models (Llama, Mistral, Phi, etc.)

All 120 models include pricing, context windows, and max token metadata. See `selectools.models` for the complete registry.

> **Coming in v0.7.0:** Embedding models and RAG support

## Real-World Examples

### 1. **Quick Start: Simple Echo Tool**

The simplest possible tool—great for testing your setup:

```python
from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.providers.openai_provider import OpenAIProvider

@tool(description="Echo input back to user")
def echo(text: str) -> str:
    return text

agent = Agent(tools=[echo], provider=OpenAIProvider(), config=AgentConfig(max_iterations=3))
resp = agent.run([Message(role=Role.USER, content="Hello, world!")])
print(resp.content)
```

### 2. **Customer Support: Multi-Tool Workflow**

Build a customer support agent that can search a knowledge base, check order status, and create tickets:

```python
from selectools import Agent, AgentConfig, Message, Role, ToolRegistry
from selectools.providers.openai_provider import OpenAIProvider
import json

registry = ToolRegistry()

@registry.tool(description="Search the knowledge base for help articles")
def search_kb(query: str, max_results: int = 5) -> str:
    # Your knowledge base search logic here
    results = [
        {"title": "How to reset password", "url": "https://help.example.com/reset"},
        {"title": "Shipping information", "url": "https://help.example.com/shipping"}
    ]
    return json.dumps(results)

@registry.tool(description="Look up order status by order ID")
def check_order(order_id: str) -> str:
    # Your order lookup logic here
    return json.dumps({
        "order_id": order_id,
        "status": "shipped",
        "tracking": "1Z999AA10123456784",
        "estimated_delivery": "2025-12-10"
    })

@registry.tool(description="Create a support ticket")
def create_ticket(customer_email: str, subject: str, description: str, priority: str = "normal") -> str:
    # Your ticketing system integration here
    ticket_id = "TKT-12345"
    return json.dumps({
        "ticket_id": ticket_id,
        "status": "created",
        "message": f"Ticket {ticket_id} created successfully"
    })

from selectools.models import OpenAI

agent = Agent(
    tools=registry.all(),
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O.id),
    config=AgentConfig(max_iterations=5, temperature=0.7)
)

# Customer inquiry
response = agent.run([
    Message(role=Role.USER, content="Hi, I ordered something last week (order #12345) and haven't received it yet. Can you help?")
])
print(response.content)
```

### 3. **Vision AI: Bounding Box Detection**

Detect and annotate objects in images using OpenAI Vision:

```python
from selectools import Agent, AgentConfig, Message, Role
from selectools.examples.bbox import create_bounding_box_tool
from selectools.providers.openai_provider import OpenAIProvider

bbox_tool = create_bounding_box_tool()
from selectools.models import OpenAI

agent = Agent(
    tools=[bbox_tool],
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O.id),
    config=AgentConfig(max_iterations=5)
)

# Analyze an image and find specific objects
response = agent.run([
    Message(
        role=Role.USER,
        content="Find the laptop in this image and draw a bounding box around it",
        image_path="assets/office_desk.jpg"
    )
])
print(response.content)
# Output: Detected laptop; output saved to assets/office_desk_with_bbox.png
```

### 4. **Data Pipeline: Research Assistant**

Chain multiple tools to research a topic, summarize findings, and save results:

```python
from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.providers.openai_provider import OpenAIProvider
import json

@tool(description="Search academic papers and articles")
def search_papers(query: str, year_from: int = 2020) -> str:
    # Your search API integration (e.g., Semantic Scholar, arXiv)
    papers = [
        {"title": "Attention Is All You Need", "authors": "Vaswani et al.", "year": 2017},
        {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "authors": "Devlin et al.", "year": 2018}
    ]
    return json.dumps(papers)

@tool(description="Extract key insights from text")
def extract_insights(text: str, num_insights: int = 5) -> str:
    # Your summarization logic
    insights = [
        "Transformers use self-attention mechanisms",
        "BERT uses bidirectional training",
        "Pre-training on large corpora improves performance"
    ]
    return json.dumps(insights)

@tool(description="Save research findings to a file")
def save_findings(filename: str, content: str) -> str:
    with open(filename, 'w') as f:
        f.write(content)
    return f"Saved findings to {filename}"

from selectools.models import OpenAI

agent = Agent(
    tools=[search_papers, extract_insights, save_findings],
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O.id),
    config=AgentConfig(max_iterations=8, temperature=0.3)
)

response = agent.run([
    Message(role=Role.USER, content="Research transformer architectures, extract key insights, and save to research_notes.txt")
])
print(response.content)
```

### 5. **Streaming UI: Real-Time Chat**

Build responsive UIs with streaming responses:

```python
from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.providers.openai_provider import OpenAIProvider
import sys

@tool(description="Get current time in a timezone")
def get_time(timezone: str = "UTC") -> str:
    from datetime import datetime
    import pytz
    tz = pytz.timezone(timezone)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

def stream_to_console(chunk: str):
    """Print chunks as they arrive for responsive UX"""
    print(chunk, end='', flush=True)

from selectools.models import OpenAI

agent = Agent(
    tools=[get_time],
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O.id),
    config=AgentConfig(stream=True, max_iterations=3)
)

response = agent.run(
    [Message(role=Role.USER, content="What time is it in Tokyo and New York?")],
    stream_handler=stream_to_console
)
print("\n")  # Newline after streaming completes
```

### 6. **Secure Tool Injection: Database Access**

Keep sensitive credentials out of tool signatures using `injected_kwargs`:

```python
from selectools import Agent, AgentConfig, Message, Role, Tool
from selectools.tools import ToolParameter
from selectools.providers.openai_provider import OpenAIProvider
import psycopg2

def query_database(sql: str, db_connection) -> str:
    """
    Execute a SQL query. The db_connection is injected, not exposed to the LLM.
    """
    with db_connection.cursor() as cursor:
        cursor.execute(sql)
        results = cursor.fetchall()
    return str(results)

# Create connection (not exposed to LLM)
db_conn = psycopg2.connect(
    host="localhost",
    database="myapp",
    user="readonly_user",
    password="secret"
)

# Tool only exposes 'sql' parameter to LLM
db_tool = Tool(
    name="query_db",
    description="Execute a read-only SQL query against the database",
    parameters=[
        ToolParameter(name="sql", param_type=str, description="SQL SELECT query to execute")
    ],
    function=query_database,
    injected_kwargs={"db_connection": db_conn}  # Injected at runtime
)

agent = Agent(
    tools=[db_tool],
    provider=OpenAIProvider(),
    config=AgentConfig(max_iterations=3)
)

response = agent.run([
    Message(role=Role.USER, content="How many users signed up last week?")
])
print(response.content)
```

### 7. **Testing: Offline Development**

Develop and test without API calls using the Local provider:

```python
from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.providers.stubs import LocalProvider

@tool(description="Format a todo item with priority")
def create_todo(task: str, priority: str = "medium", due_date: str = None) -> str:
    result = f"[{priority.upper()}] {task}"
    if due_date:
        result += f" (due: {due_date})"
    return result

agent = Agent(
    tools=[create_todo],
    provider=LocalProvider(),  # No network calls, no API costs
    config=AgentConfig(max_iterations=2, model="local")
)

response = agent.run([
    Message(role=Role.USER, content="Add 'finish project report' to my todos with high priority")
])
print(response.content)
```

### 8. **Provider Switching: Zero Refactoring**

Switch between providers without changing your tool definitions:

```python
from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.providers.openai_provider import OpenAIProvider
from selectools.providers.anthropic_provider import AnthropicProvider
from selectools.providers.gemini_provider import GeminiProvider
import os

@tool(description="Calculate compound interest")
def calculate_interest(principal: float, rate: float, years: int) -> str:
    amount = principal * (1 + rate/100) ** years
    return f"After {years} years: ${amount:.2f}"

# Choose provider based on environment or preference
from selectools.models import OpenAI, Anthropic, Gemini

provider_name = os.getenv("LLM_PROVIDER", "openai")
providers = {
    "openai": OpenAIProvider(default_model=OpenAI.GPT_4O.id),
    "anthropic": AnthropicProvider(default_model=Anthropic.SONNET_3_5_20241022.id),
    "gemini": GeminiProvider(default_model=Gemini.FLASH_2_0.id)
}

agent = Agent(
    tools=[calculate_interest],
    provider=providers[provider_name],
    config=AgentConfig(max_iterations=3)
)

response = agent.run([
    Message(role=Role.USER, content="If I invest $10,000 at 7% annual interest for 10 years, how much will I have?")
])
print(response.content)
```

### 9. **Cost Tracking: Monitor Token Usage & Costs (v0.5.0)**

Track token usage and estimated costs automatically:

```python
from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.providers.openai_provider import OpenAIProvider

@tool(description="Search for information")
def search(query: str) -> str:
    return f"Results for: {query}"

@tool(description="Summarize text")
def summarize(text: str) -> str:
    return f"Summary: {text[:50]}..."

# Enable cost tracking with optional warning threshold
from selectools.models import OpenAI

agent = Agent(
    tools=[search, summarize],
    provider=OpenAIProvider(default_model=OpenAI.GPT_4O.id),
    config=AgentConfig(
        max_iterations=5,
        cost_warning_threshold=0.10  # Warn if cost exceeds $0.10
    )
)

response = agent.run([
    Message(role=Role.USER, content="Search for Python tutorials and summarize the top result")
])

# Access usage statistics
print(f"Total tokens: {agent.total_tokens:,}")
print(f"Total cost: ${agent.total_cost:.6f}")
print("\nDetailed breakdown:")
print(agent.get_usage_summary())

# Output:
# 📊 Usage Summary
# Total Tokens: 1,234
# Total Cost: $0.012345
#
# Tool Usage:
#   - search: 1 calls, 567 tokens
#   - summarize: 1 calls, 667 tokens
```

**Key Features:**

- Automatic token counting for all providers
- Cost estimation for 15+ models (OpenAI, Anthropic, Gemini)
- Per-tool usage breakdown
- Configurable cost warnings
- Reset usage: `agent.reset_usage()`

### 10. **Pre-built Toolbox: Ready-to-Use Tools (v0.5.1)**

Skip the boilerplate and use production-ready tools from the toolbox:

```python
from selectools import Agent, AgentConfig, Message, Role
from selectools.providers.openai_provider import OpenAIProvider
from selectools.toolbox import get_all_tools, get_tools_by_category

# Use all 22 tools from the toolbox
all_tools = get_all_tools()
agent = Agent(
    tools=all_tools,
    provider=OpenAIProvider(),
    config=AgentConfig(max_iterations=8)
)

# Or get specific categories
file_tools = get_tools_by_category("file")  # read_file, write_file, list_files, file_exists
data_tools = get_tools_by_category("data")  # parse_json, json_to_csv, csv_to_json, etc.
text_tools = get_tools_by_category("text")  # count_text, search_text, extract_emails, etc.
datetime_tools = get_tools_by_category("datetime")  # get_current_time, parse_datetime, etc.
web_tools = get_tools_by_category("web")  # http_get, http_post

# Complex multi-step task using multiple tool categories
response = agent.run([
    Message(
        role=Role.USER,
        content="""
        1. Get the current time in UTC
        2. Parse this JSON: {"users": [{"name": "Alice", "email": "alice@test.com"}]}
        3. Extract all email addresses from the JSON
        4. Write the results to a file called results.txt
        """
    )
])
print(response.content)
```

**Available Tool Categories:**

| Category         | Tools                                                                                                          | Description                 |
| ---------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **File** (4)     | `read_file`, `write_file`, `list_files`, `file_exists`                                                         | File system operations      |
| **Data** (5)     | `parse_json`, `json_to_csv`, `csv_to_json`, `extract_json_field`, `format_table`                               | Data parsing and formatting |
| **Text** (7)     | `count_text`, `search_text`, `replace_text`, `extract_emails`, `extract_urls`, `convert_case`, `truncate_text` | Text processing             |
| **DateTime** (4) | `get_current_time`, `parse_datetime`, `time_difference`, `date_arithmetic`                                     | Date/time utilities         |
| **Web** (2)      | `http_get`, `http_post`                                                                                        | HTTP requests               |

See `examples/toolbox_demo.py` for a complete demonstration.

### 11. **Async Agent: Modern Python with asyncio**

Build high-performance async applications with native async support:

```python
import asyncio
from selectools import Agent, AgentConfig, Message, Role, tool, ConversationMemory
from selectools.providers.openai_provider import OpenAIProvider

# Async tools for I/O-bound operations
@tool(description="Fetch weather data")
async def fetch_weather(city: str) -> str:
    await asyncio.sleep(0.1)  # Simulate async API call
    return f"Weather in {city}: Sunny, 72°F"

# Sync tools work seamlessly alongside async tools
@tool(description="Calculate")
def calculate(a: int, b: int) -> str:
    return f"{a} + {b} = {a + b}"

async def main():
    # Conversation memory works with async
    memory = ConversationMemory(max_messages=20)

    agent = Agent(
        tools=[fetch_weather, calculate],
        provider=OpenAIProvider(),
        config=AgentConfig(max_iterations=5),
        memory=memory
    )

    # Use arun() instead of run()
    response = await agent.arun([
        Message(role=Role.USER, content="What's the weather in Seattle?")
    ])
    print(response.content)

asyncio.run(main())
```

**FastAPI Integration:**

```python
from fastapi import FastAPI
from selectools import Agent, Message, Role, tool, OpenAIProvider

app = FastAPI()

@tool(description="Fetch data")
async def fetch_data(query: str) -> str:
    return f"Data for {query}"

@app.post("/chat")
async def chat(message: str):
    agent = Agent(tools=[fetch_data], provider=OpenAIProvider())
    response = await agent.arun([Message(role=Role.USER, content=message)])
    return {"response": response.content}
```

**Key Async Features:**

- `Agent.arun()` for non-blocking execution
- Async tools with `async def`
- All providers support async (OpenAI, Anthropic, Gemini)
- Concurrent execution with `asyncio.gather()`
- Works with FastAPI, aiohttp, and async frameworks

## Tool ergonomics

- Use `ToolRegistry` or the `@tool` decorator to infer schemas from function signatures and register tools.
- Inject per-tool config or auth using `injected_kwargs` or `config_injector` when constructing a `Tool`.
- Type hints map to JSON schema; defaults make parameters optional.

## Streaming Tools

Tools can stream results progressively using Python generators, providing real-time feedback for long-running operations:

```python
from typing import Generator
from selectools import tool, Agent, AgentConfig

@tool(description="Process large file line by line", streaming=True)
def process_file(filepath: str) -> Generator[str, None, None]:
    """Process file and yield results progressively."""
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            result = process_line(line)
            yield f"[Line {i}] {result}\n"

# Display chunks as they arrive
def on_tool_chunk(tool_name: str, chunk: str):
    print(f"[{tool_name}] {chunk}", end='', flush=True)

config = AgentConfig(hooks={'on_tool_chunk': on_tool_chunk})
agent = Agent(tools=[process_file], provider=provider, config=config)
```

**Features:**

- ✅ Sync generators (`Generator[str, None, None]`)
- ✅ Async generators (`AsyncGenerator[str, None]`)
- ✅ Real-time chunk callbacks via `on_tool_chunk` hook
- ✅ Analytics tracking for chunk counts and streaming metrics
- ✅ Toolbox includes `read_file_stream` and `process_csv_stream`

See `examples/streaming_tools_demo.py` for complete examples.

## Tests

```bash
python tests/test_framework.py
```

- Covers parsing (mixed/fenced), agent loop (retries/streaming), provider mocks (Anthropic/Gemini), CLI streaming, bbox mock path, and tool schema basics.

## Packaging

The project ships a `pyproject.toml` with console scripts and a src layout. Adjust version/metadata before publishing to PyPI.
CI workflow (`.github/workflows/ci.yml`) runs tests, build, and twine check. Tags matching `v*` attempt TestPyPI/PyPI publishes when tokens are provided.

## License

This project is licensed under the **GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)**.

### What This Means for You

✅ **You CAN:**

- Use this library in commercial applications
- Profit from applications that use this library
- Import and use the library without sharing your application code
- Distribute applications that use this library

✅ **You MUST:**

- Preserve copyright notices and license information
- Share any modifications you make to the library itself under LGPL-3.0
- Provide attribution to the original authors

❌ **You CANNOT:**

- Relicense this library under different terms
- Claim this code as your own
- Create proprietary forks (modifications must remain open source)

**In Practice:** You can build and sell proprietary applications using this library via normal import/usage. Only if you modify the library's source code itself must you share those modifications. This is the same license used by popular projects like Qt and GTK.

For the full license text, see the [LICENSE](LICENSE) file.

## More docs

- Single source of truth is this README.
- Examples:
  - `python examples/search_weather.py` - Simple tool with local mock provider
  - `python examples/async_agent_demo.py` - Async/await usage with FastAPI patterns
  - `python examples/conversation_memory_demo.py` - Multi-turn conversation with memory
  - `python examples/cost_tracking_demo.py` - Token counting and cost tracking (v0.5.0)
  - `python examples/toolbox_demo.py` - Using pre-built tools from toolbox (v0.5.1)
  - `python examples/v0_5_2_demo.py` - Tool validation & observability hooks (v0.5.2)
  - `python examples/ollama_demo.py` - Local LLM execution with Ollama (v0.6.0)
  - `python examples/tool_analytics_demo.py` - Track and analyze tool usage (v0.6.0)
  - `python examples/streaming_tools_demo.py` - Progressive tool results with streaming (v0.6.1)
  - `python examples/customer_support_bot.py` - Multi-tool customer support workflow
  - `python examples/data_analysis_agent.py` - Data exploration and analysis tools
- Dev helpers:
  - `python scripts/smoke_cli.py` - Quick provider smoke tests (skips missing keys)
  - `python scripts/test_memory_with_openai.py` - Test memory with real OpenAI API

---

## Roadmap

We're actively developing new features to make Selectools the most production-ready tool-calling library. See **[ROADMAP.md](ROADMAP.md)** for the complete development roadmap, including:

**✅ Completed in v0.4.0:**

- Conversation Memory - Multi-turn context management
- Async Support - `Agent.arun()`, async tools, async providers
- Real Provider Implementations - Full Anthropic & Gemini SDK integration

**✅ Completed in v0.5.0:**

- Better Error Messages - Custom exceptions with helpful context and suggestions
- Cost Tracking - Automatic token counting and cost estimation with warnings
- Gemini SDK Migration - Updated to new google-genai SDK (v1.0+)

**✅ Completed in v0.5.1:**

- Pre-built Tool Library - 22 production-ready tools in 5 categories (file, web, data, datetime, text)

**✅ Completed in v0.5.2:**

- Tool Validation at Registration - Catch tool definition errors during development, not production
- Observability Hooks - 10 lifecycle hooks for monitoring, debugging, and tracking agent behavior

**✅ Completed in v0.6.0:**

- Local Model Support - Ollama provider for privacy-first, zero-cost local LLM execution
- Tool Usage Analytics - Track metrics, success rates, execution times, and parameter patterns

**✅ Completed in v0.6.1:**

- Streaming Tool Results - Tools can yield results progressively with Generator/AsyncGenerator
- Streaming Observability - `on_tool_chunk` hook for real-time chunk callbacks
- Streaming Analytics - Track chunk counts and streaming-specific metrics
- Toolbox Streaming Tools - `read_file_stream` and `process_csv_stream` for large files

**🟡 Coming in v0.6.x:**

- Dynamic Tool Loading - Hot-reload tools without restarting the agent

**🚀 Future (v0.7.0+):**

- Parallel Tool Execution - Run multiple tools concurrently
- Tool Composition - Chain tools together with `@compose` decorator
- Advanced context management
- And much more...

See **[ROADMAP.md](ROADMAP.md)** for detailed feature descriptions, status tracking, and implementation notes.

### 🤝 Contributing

Want to help build these features? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We'd love contributions for:

- Priority 1 features (quick wins)
- Tool implementations for the toolbox
- Examples and tutorials
- Documentation improvements

---

See the full comparison with LangChain in [docs/LANGCHAIN_COMPARISON.md](docs/LANGCHAIN_COMPARISON.md).
