# AI Tool Calling from Scratch

Provider-agnostic Python library for TOOL_CALL-style tool execution across multiple LLM/vision backends. The library exposes typed primitives (`Agent`, `Tool`, `Message`, `Role`), pluggable providers (OpenAI adapter + stubs for Anthropic/Gemini/Local), a hardened TOOL_CALL parser, prompt template, CLI entrypoint, and a bounding-box demo tool.

## What's Included

- Core package at `src/toolcalling/` with agent loop, parser, prompt builder, and provider adapters
- OpenAI provider implementation plus Anthropic/Gemini/Local stubs sharing the same interface
- CLI (`toolcalling`) to list tools and run prompts against a chosen provider/model
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

# Run a one-off prompt
toolcalling run --provider openai --model gpt-4o --prompt "Say hello" --tool echo

# Vision run with an image
toolcalling run --prompt "Find the dog" --image assets/dog.png --tool detect_bounding_box
```

## Bounding Box Demo

```
python scripts/chat.py            # processes assets/dog.png
python scripts/chat.py --interactive
```
The demo uses the bounding-box tool backed by OpenAI Vision and writes `*_with_bbox.png` alongside the input image.

## Tests

```bash
python tests/test_framework.py
```

## Packaging

The project ships a `pyproject.toml` with console scripts and a src layout. Adjust version/metadata before publishing to PyPI.
