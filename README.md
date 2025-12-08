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

provider = OpenAIProvider(default_model="gpt-4o")
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

- OpenAI: streaming; vision via Chat Completions `image_url` (e.g., `gpt-5`); request timeout default 30s; retries/backoff via `AgentConfig`.
- Anthropic: streaming; vision model-dependent; set `ANTHROPIC_API_KEY`.
- Gemini: streaming; vision model-dependent; set `GEMINI_API_KEY`.
- Local: no network; echoes latest user text; no vision.
- Rate limits: agent detects `rate limit`/`429` and backs off + retries.
- Timeouts: `AgentConfig.request_timeout` (provider) and `tool_timeout_seconds` (per tool).

## Agent config at a glance

- Core: `model`, `temperature`, `max_tokens`, `max_iterations`.
- Reliability: `max_retries`, `retry_backoff_seconds`, rate-limit backoff, `request_timeout`.
- Execution safety: `tool_timeout_seconds` to bound tool runtime.
- Streaming: `stream=True` to stream provider deltas; optional `stream_handler` callback.

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

agent = Agent(
    tools=registry.all(),
    provider=OpenAIProvider(default_model="gpt-4o"),
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
agent = Agent(
    tools=[bbox_tool],
    provider=OpenAIProvider(default_model="gpt-4o"),
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

agent = Agent(
    tools=[search_papers, extract_insights, save_findings],
    provider=OpenAIProvider(default_model="gpt-4o"),
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

agent = Agent(
    tools=[get_time],
    provider=OpenAIProvider(default_model="gpt-4o"),
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
provider_name = os.getenv("LLM_PROVIDER", "openai")
providers = {
    "openai": OpenAIProvider(default_model="gpt-4o"),
    "anthropic": AnthropicProvider(default_model="claude-3-5-sonnet-20241022"),
    "gemini": GeminiProvider(default_model="gemini-2.0-flash-exp")
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

### 9. **Async Agent: Modern Python with asyncio**

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
- Optional dev helpers: `python scripts/smoke_cli.py` (skips providers missing keys), `python scripts/chat.py` (vision demo), `python examples/search_weather.py` (local mock tools).

---

## Roadmap & Future Improvements

We're committed to making Selectools the most production-ready, developer-friendly tool-calling library. Here's our roadmap organized by priority and impact.

### 🎯 **Priority 1: Quick Wins** (Coming in v0.4.0 - Next 2 weeks)

These high-impact features can be implemented quickly and will immediately improve the developer experience:

#### **Conversation Memory** ⏱️ 2 hours ✅ **Completed in v0.4.0**
- Simple `ConversationMemory` class for maintaining context
- Automatic message history management with configurable limits
- Easy integration: `memory = ConversationMemory(); agent = Agent(..., memory=memory)`
- **Status**: ✅ Implemented
- **Why it matters**: Closes a major gap with LangChain, makes multi-turn conversations trivial

#### **Async Support** ⏱️ 6 hours ✅ **Completed in v0.4.0**
- `Agent.arun()` for non-blocking agent execution
- Async tools with `async def` (mixed seamlessly with sync tools)
- Full async provider support (OpenAI, Anthropic, Gemini)
- Compatible with FastAPI, asyncio, and other async frameworks
- **Status**: ✅ Implemented
- **Why it matters**: Required for high-performance web applications and concurrent operations

#### **Better Error Messages** ⏱️ 2 hours
- PyTorch-style helpful error messages with suggestions
- Clear parameter validation errors with "did you mean?" suggestions
- Detailed tool execution failure messages
- **Status**: 🟡 Planned
- **Why it matters**: Dramatically improves debugging and developer experience

#### **Cost Tracking** ⏱️ 2 hours
- Automatic tracking of API costs and token usage
- Per-tool cost breakdown
- `agent.total_cost` and `agent.total_tokens` properties
- **Status**: 🟡 Planned
- **Why it matters**: Essential for production monitoring and budget control

#### **Pre-built Tool Library** ⏱️ 4 hours
- `selectools.toolbox` with 5-10 common tools
- Web search (DuckDuckGo/Brave), HTTP requests, file operations
- Python REPL, shell commands (sandboxed)
- **Status**: 🟡 Planned
- **Why it matters**: Instant productivity, no need to write basic tools from scratch

**Total Priority 1: ~13 hours = 1-2 days of focused work**

---

### 🚀 **Priority 2: High-Impact Features** (v0.5.0 - Next month)

Medium effort, high value features that differentiate us from competitors:

#### **Streaming Tool Results** ⏱️ 3 hours
- Stream tool output as it's generated (not just LLM responses)
- `@tool(streaming=True)` decorator for long-running tools
- Real-time progress updates for better UX
- **Status**: 🟡 Planned

#### **Parallel Tool Execution** ⏱️ 4 hours
- Automatically detect and execute independent tools concurrently
- `AgentConfig(parallel_tools=True)` for 3-5x faster workflows
- Smart dependency detection
- **Status**: 🟡 Planned

#### **Observability Hooks** ⏱️ 3 hours
- `on_tool_start`, `on_tool_end`, `on_error` hooks
- Easy integration with logging, monitoring, and alerting
- OpenTelemetry-compatible
- **Status**: 🟡 Planned

#### **Tool Composition** ⏱️ 4 hours
- `@compose(tool1, tool2)` decorator for chaining tools
- Built-in patterns: pipeline, map-reduce, conditional
- Simplifies complex multi-step operations
- **Status**: 🟡 Planned

#### **Tool Validation at Registration** ⏱️ 2 hours
- Catch type errors when defining tools, not at runtime
- Validate parameter types, descriptions, and schemas
- Clear error messages for invalid tool definitions
- **Status**: 🟡 Planned

#### **Interactive Debug Mode** ⏱️ 6 hours
- `AgentConfig(debug=True)` for step-through debugging
- Inspect state at each tool call
- Modify parameters and retry
- **Status**: 🟡 Planned

**Total Priority 2: ~22 hours = 3-4 days**

---

### 💎 **Priority 3: Advanced Features** (v0.6.0+ - Next quarter)

Longer-term features that require more design and implementation:

These planned enhancements will make this library even more powerful than existing alternatives:

#### **🧠 Advanced Context Management**

**Automatic Conversation Summarization** ⏱️ 8 hours
- Intelligent summarization of long conversations to stay within token limits
- Configurable summarization strategies (extractive, abstractive, hybrid)
- Preserve critical context while compressing historical messages
- **Status**: 🟡 Planned
- **Why it matters**: Most libraries crash or truncate when hitting context limits. We'll handle it gracefully.

**Sliding Window with Smart Retention** ⏱️ 6 hours
- Keep recent messages + important historical context
- Automatic detection of critical information (tool results, user preferences, decisions)
- Configurable window sizes per provider
- **Status**: 🟡 Planned
- **Why it matters**: Better than simple truncation—maintains conversation coherence.

**Multi-Turn Memory System** ⏱️ 12 hours
- Persistent memory across sessions (key-value store, vector DB integration)
- Automatic extraction of facts, preferences, and entities
- Memory retrieval based on relevance to current conversation
- **Status**: 🟡 Planned
- **Why it matters**: Build agents that remember users across sessions, unlike stateless alternatives.

#### **🔧 Enhanced Tool Capabilities**

**Dynamic Tool Loading** ⏱️ 8 hours
- Hot-reload tools without restarting the agent
- Plugin system for third-party tool packages
- Tool versioning and compatibility checking
- **Status**: 🟡 Planned
- **Why it matters**: Deploy new capabilities without downtime.

**Tool Usage Analytics** ⏱️ 6 hours
- Track tool invocation frequency, latency, and success rates
- Automatic performance profiling and bottleneck detection
- Per-tool cost breakdown (API calls, compute time)
- **Status**: 🟡 Planned
- **Why it matters**: Optimize your agent's performance and costs with data.

**Tool Marketplace/Registry** ⏱️ 16 hours
- `selectools.marketplace.install_tool("weather-api")` for community tools
- Searchable registry of pre-built tools
- Version management and dependency resolution
- **Status**: 🟡 Planned
- **Why it matters**: Ecosystem growth and community contributions.

#### **🎯 Provider Enhancements**

**Universal Vision Support** ⏱️ 6 hours
- Unified vision API across all providers (OpenAI, Anthropic, Gemini)
- Automatic image preprocessing (resize, format conversion, optimization)
- Multi-image support with spatial reasoning
- **Status**: 🟡 Planned
- **Why it matters**: Write vision code once, run on any provider.

**Provider Auto-Selection** ⏱️ 8 hours
- Automatic provider selection based on task requirements (vision, speed, cost)
- Fallback chains (try OpenAI, fall back to Anthropic, then Gemini)
- Load balancing across multiple API keys/accounts
- **Status**: 🟡 Planned
- **Why it matters**: Maximum reliability and cost optimization without manual switching.

**Streaming Improvements** ⏱️ 6 hours
- Server-Sent Events (SSE) support for web applications
- WebSocket streaming for real-time bidirectional communication
- Already have partial tool result streaming (see Priority 2)
- **Status**: 🟡 Planned
- **Why it matters**: Build more responsive UIs with richer streaming capabilities.

**Local Model Support** ⏱️ 10 hours
- Integration with Ollama, LM Studio, and other local inference servers
- Quantization-aware provider selection
- GPU utilization monitoring and optimization
- **Status**: 🟡 Planned
- **Why it matters**: Run powerful agents completely offline with local models.

#### **🛡️ Production Reliability**

**Advanced Error Recovery** ⏱️ 8 hours
- Automatic retry with exponential backoff ✅ (already implemented)
- Circuit breaker pattern for failing tools
- Graceful degradation (disable failing tools, continue with others)
- Dead letter queue for failed tool executions
- **Status**: 🟡 Planned (partial ✅)
- **Why it matters**: Keep agents running even when individual components fail.

**Observability & Debugging** ⏱️ 12 hours
- OpenTelemetry integration for distributed tracing
- Structured logging with correlation IDs
- Agent execution replay for debugging
- Performance profiling and flame graphs
- **Status**: 🟡 Planned
- **Why it matters**: Debug production issues quickly with full visibility.

**Rate Limiting & Quotas** ⏱️ 6 hours
- Per-tool rate limiting and quota management
- User-level quotas and fair usage policies
- Automatic throttling and backpressure
- **Status**: 🟡 Planned
- **Why it matters**: Prevent abuse and control costs in multi-tenant environments.

**Security Hardening** ⏱️ 10 hours
- Tool sandboxing (execute in isolated environments)
- Input validation and sanitization framework
- Output filtering for sensitive data (PII, credentials)
- Audit logging for compliance
- **Status**: 🟡 Planned
- **Why it matters**: Deploy agents safely in enterprise environments.

#### **📊 Developer Experience**

**Visual Agent Builder** ⏱️ 24 hours
- Web-based UI for designing agent workflows
- Drag-and-drop tool composition
- Live testing and debugging
- Export to Python code
- **Status**: 🟡 Planned
- **Why it matters**: Faster prototyping and easier onboarding for non-developers.

**Enhanced Testing Framework** ⏱️ 10 hours
- Snapshot testing for agent conversations
- Property-based testing for tool schemas
- Load testing and performance benchmarking
- Mock provider with configurable behaviors (latency, errors, rate limits)
- **Status**: 🟡 Planned
- **Why it matters**: Catch bugs before production with comprehensive testing.

**Documentation Generation** ⏱️ 8 hours
- Auto-generate API docs from tool definitions
- Interactive tool playground (try tools in browser)
- Example generation from tool schemas
- **Status**: 🟡 Planned
- **Why it matters**: Better documentation with zero maintenance overhead.

**Type Safety Improvements** ⏱️ 6 hours
- Full type inference for tool parameters and returns
- Runtime type checking with detailed error messages
- Integration with Pydantic for complex schemas
- **Status**: 🟡 Planned
- **Why it matters**: Catch type errors at development time, not runtime.

#### **🌐 Ecosystem Integration**

**Framework Integrations** ⏱️ 12 hours
- FastAPI/Flask middleware for agent endpoints
- LangChain tool adapter (use LangChain tools in this library)
- LlamaIndex integration for RAG workflows
- **Status**: 🟡 Planned
- **Why it matters**: Seamless integration with popular Python frameworks.

**CRM & Business Tools** ⏱️ 16 hours
- Pre-built tools for HubSpot, Salesforce, Close
- Calendar integrations (Google Calendar, Outlook)
- Communication tools (Slack, Discord, email)
- **Status**: 🟡 Planned
- **Why it matters**: Build business automation agents faster with ready-made integrations.

**Data Source Connectors** ⏱️ 20 hours
- SQL database connectors with query builders
- Vector database integration (Pinecone, Weaviate, Chroma)
- Cloud storage (S3, GCS, Azure Blob)
- APIs (REST, GraphQL) with automatic schema discovery
- **Status**: 🟡 Planned
- **Why it matters**: Connect agents to your data without writing boilerplate.

#### **🚀 Performance Optimizations**

**Caching Layer** ⏱️ 10 hours
- LRU cache for identical tool calls
- Semantic caching (similar queries return cached results)
- Distributed caching (Redis, Memcached)
- Cache invalidation strategies
- **Status**: 🟡 Planned
- **Why it matters**: Reduce API costs and latency by 50-80% for repeated queries.

**Batch Processing** ⏱️ 8 hours
- Batch multiple user requests for efficient processing
- Automatic request coalescing
- Priority queues for urgent requests
- **Status**: 🟡 Planned
- **Why it matters**: Handle high-throughput scenarios efficiently.

**Prompt Optimization** ⏱️ 6 hours
- Automatic prompt compression while preserving meaning
- Token-efficient tool schema serialization
- Dynamic prompt templating based on provider capabilities
- **Status**: 🟡 Planned
- **Why it matters**: Reduce costs and latency with optimized prompts.

---

### 📈 **Roadmap Summary**

**Timeline:**
- **v0.4.0** (Next 2 weeks): Priority 1 features - Conversation Memory, Async, Cost Tracking, Better Errors, Pre-built Tools
- **v0.5.0** (Next month): Priority 2 features - Streaming Tools, Parallel Execution, Observability, Tool Composition
- **v0.6.0+** (Next quarter): Priority 3 features - Advanced context, marketplace, visual builder

**Estimated Effort:**
- Priority 1: ~13 hours (1-2 days)
- Priority 2: ~22 hours (3-4 days)
- Priority 3: ~200+ hours (ongoing)

**Status Legend:**
- ✅ Implemented
- 🟡 Planned
- 🔵 In Progress
- ⏸️ Deferred

### Why These Improvements Matter

While other libraries focus on basic tool calling, these enhancements will make this library the **most production-ready, developer-friendly, and feature-complete** tool-calling framework available:

1. **LangChain**: Great ecosystem but heavy, complex, and opinionated. Our library stays lightweight while adding enterprise features.

2. **OpenAI Function Calling**: Provider-locked and basic. We add provider agnosticism + advanced features.

3. **Anthropic Tool Use**: Same provider lock-in issue. We provide a unified interface.

4. **Haystack**: Focused on RAG/search. We're tool-calling specialists with broader scope.

5. **AutoGPT/BabyAGI**: Autonomous agents but limited tool infrastructure. We provide the robust foundation they need.

Our roadmap focuses on **production reliability**, **developer experience**, and **real-world use cases** that other libraries overlook. We're building the tool-calling library you wish existed.

### 🤝 **Contributing to the Roadmap**

Want to help build these features? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Priority 1 features are great for first-time contributors!

**High-impact contributions we'd love:**
- Implement any Priority 1 feature (2-4 hours each)
- Add tools to the pre-built toolbox
- Write examples and tutorials
- Improve documentation
- Report bugs and suggest improvements

See the full comparison with LangChain in [docs/LANGCHAIN_COMPARISON.md](docs/LANGCHAIN_COMPARISON.md).
