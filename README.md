# AI Tool Calling from Scratch

Provider-agnostic Python library for TOOL_CALL-style tool execution across multiple LLM/vision backends. The library exposes typed primitives (`Agent`, `Tool`, `Message`, `Role`), pluggable providers (OpenAI adapter + Anthropic/Gemini/Local), streaming support, a hardened TOOL_CALL parser, prompt template, CLI entrypoint, bounding-box demo tool, and helper examples.

## What's Included

- Core package at `src/toolcalling/` with agent loop, parser, prompt builder, and provider adapters
- OpenAI provider implementation plus Anthropic/Gemini/Local providers sharing the same interface
- CLI (`toolcalling`) to list tools, run one-offs, or chat interactively with streaming
- Bounding-box detection tool (OpenAI Vision) and demo runner in `scripts/chat.py`
- Tests with fake providers to validate schemas, parsing, and agent wiring
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

## CLI

```bash
# List available tools (echo + detect_bounding_box)
toolcalling list-tools

# Run a one-off prompt (supports --stream, --dry-run)
toolcalling run --provider openai --model gpt-4o --prompt "Say hello" --tool echo

# Interactive chat with history (streams tokens if enabled)
toolcalling chat --provider local --stream

# Vision run with an image
toolcalling run --prompt "Find the object" --image assets/environment.png --tool detect_bounding_box
```

## Bounding Box Demo

```
python scripts/chat.py            # processes assets/environment.png
python scripts/chat.py --interactive
```

The demo uses the bounding-box tool backed by OpenAI Vision and writes `*_with_bbox.png` alongside the input image.
For offline tests, set `TOOLCALLING_BBOX_MOCK_JSON=tests/fixtures/bbox_mock.json` to bypass the network.

## Tool ergonomics

- Use `ToolRegistry` or the `@tool` decorator to infer schemas from function signatures and register tools.
- Inject per-tool config or auth using `injected_kwargs` or `config_injector` when constructing a `Tool`.

## Tests

```bash
python tests/test_framework.py
```

## Packaging

The project ships a `pyproject.toml` with console scripts and a src layout. Adjust version/metadata before publishing to PyPI.
CI workflow (`.github/workflows/ci.yml`) runs tests, build, and twine check. Tags matching `v*` attempt TestPyPI/PyPI publishes when tokens are provided.

## More docs

- See `docs/USER_GUIDE.md` for provider/tool/agent/CLI/demo/testing/release details.
- Smoke the CLI across providers (skips if env vars missing): `python scripts/smoke_cli.py`.
- Extra example (no network): `python examples/search_weather.py`.
