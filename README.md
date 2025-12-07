# AI Tool Calling

Provider-agnostic Python library for TOOL_CALL-style tool execution across multiple LLM/vision backends. The library exposes typed primitives (`Agent`, `Tool`, `Message`, `Role`), pluggable providers (OpenAI adapter + Anthropic/Gemini/Local), streaming support, a hardened TOOL_CALL parser, prompt template, CLI entrypoint, bounding-box demo tool, and helper examples.

## Why this library

- Provider-agnostic tool calling with schema validation and a hardened parser (handles fenced/mixed JSON).
- Robustness controls: retries with rate-limit backoff, request timeouts, per-tool execution timeouts, iteration caps.
- Streaming or one-shot responses; vision supported where providers allow.
- Simple ergonomics: `@tool`/`ToolRegistry`, CLI for one-offs or chat, ready-made examples and tests.

## What's Included

- Core package at `src/toolcalling/` with agent loop, parser, prompt builder, and provider adapters
- Providers: OpenAI plus Anthropic/Gemini/Local sharing the same interface
- Library-first examples (see below) and tests with fake providers for schemas, parsing, agent wiring
- PyPI-ready metadata (`pyproject.toml`) using a src-layout package

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key-here"
```

## Usage (Library)

```python
from toolcalling import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from toolcalling.providers.openai_provider import OpenAIProvider

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
- For offline/testing: use the Local provider and/or `TOOLCALLING_BBOX_MOCK_JSON=tests/fixtures/bbox_mock.json`.
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

## Library examples

- Simple search tool (text):

  ```python
  from toolcalling import Agent, AgentConfig, Message, Role, tool
  from toolcalling.providers.openai_provider import OpenAIProvider

  @tool(description="Echo input")
  def echo(text: str) -> str:
      return text

  agent = Agent(tools=[echo], provider=OpenAIProvider(), config=AgentConfig(max_iterations=3))
  resp = agent.run([Message(role=Role.USER, content="Hello!")])
  print(resp.content)
  ```

- Vision bounding box (uses your image):

  ```python
  from toolcalling import Agent, AgentConfig, Message, Role
  from toolcalling.examples.bbox import create_bounding_box_tool

  bbox_tool = create_bounding_box_tool()
  agent = Agent(tools=[bbox_tool], config=AgentConfig(max_iterations=5, model="gpt-4o"))
  resp = agent.run([
      Message(role=Role.USER, content="Find the object", image_path="path/to/your/image.png")
  ])
  print(resp.content)
  ```

- Multi-tool workflow (search + summarize):

  ```python
  from toolcalling import Agent, AgentConfig, Message, Role, tool
  from toolcalling.providers.gemini_provider import GeminiProvider

  @tool(description="Search the web")
  def search(query: str) -> str:
      return f"results for {query}"

  @tool(description="Summarize text")
  def summarize(text: str) -> str:
      return f"summary: {text[:200]}"

  agent = Agent(
      tools=[search, summarize],
      provider=GeminiProvider(),
      config=AgentConfig(max_iterations=4, stream=False),
  )
  resp = agent.run([Message(role=Role.USER, content="Find and summarize AI tool-calling docs")])
  print(resp.content)
  ```

- Local/offline dry-run (no network):

  ```python
  from toolcalling import Agent, AgentConfig, Message, Role, tool
  from toolcalling.providers.stubs import LocalProvider

  @tool(description="Format a todo item")
  def todo(item: str, priority: str = "medium") -> str:
      return f"[{priority}] {item}"

  agent = Agent(
      tools=[todo],
      provider=LocalProvider(),
      config=AgentConfig(max_iterations=2, model="local"),
  )
  resp = agent.run([Message(role=Role.USER, content="Add buy milk to my list")])
  print(resp.content)
  ```

- Streaming responses:

  ```python
  from toolcalling import Agent, AgentConfig, Message, Role, tool
  from toolcalling.providers.openai_provider import OpenAIProvider

  @tool(description="Generate a short outline")
  def outline(topic: str) -> str:
      return f"Outline for {topic}"

  chunks = []
  def on_chunk(text: str):
      chunks.append(text)

  agent = Agent(
      tools=[outline],
      provider=OpenAIProvider(),
      config=AgentConfig(stream=True, max_iterations=2),
  )
  resp = agent.run([Message(role=Role.USER, content="Draft an outline about LLM safety")], stream_handler=on_chunk)
  print("".join(chunks))
  ```

- Per-tool config injection (e.g., API keys per tool):

  ```python
  from toolcalling import Agent, AgentConfig, Message, Role, Tool
  from toolcalling.tools import ToolParameter
  from toolcalling.providers.openai_provider import OpenAIProvider

  def call_weather_api(city: str, api_key: str) -> str:
      # placeholder for real HTTP call
      return f"Weather for {city} using key {api_key[:4]}***"

  weather_tool = Tool(
      name="weather_lookup",
      description="Gets weather for a city",
      parameters=[ToolParameter(name="city", param_type=str, description="City name")],
      function=call_weather_api,
      injected_kwargs={"api_key": "YOUR_WEATHER_API_KEY"},
  )

  agent = Agent(
      tools=[weather_tool],
      provider=OpenAIProvider(),
      config=AgentConfig(max_iterations=2),
  )
  resp = agent.run([Message(role=Role.USER, content="What's the weather in Paris?")])
  print(resp.content)
  ```

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

## More docs

- Single source of truth is this README.
- Optional dev helpers: `python scripts/smoke_cli.py` (skips providers missing keys), `python scripts/chat.py` (vision demo), `python examples/search_weather.py` (local mock tools).
