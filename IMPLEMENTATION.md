# AI Tool Calling Framework - Implementation Notes

## Overview

The project is now a provider-agnostic Python library packaged under `toolcalling/`. It implements a TOOL_CALL contract, pluggable providers, a hardened parser/prompt, a CLI, and an example bounding-box tool that uses OpenAI Vision.

## Package Layout

- `toolcalling/types.py` — `Role`, `Message`, `ToolCall`
- `toolcalling/tools.py` — `Tool`, `ToolParameter`, schema + validation
- `toolcalling/prompt.py` — system prompt builder embedding tool schemas
- `toolcalling/parser.py` — robust TOOL_CALL parser (fenced/mixed blocks, lenient JSON, size limits)
- `toolcalling/agent.py` — iterative loop with configurable limits, retries, streaming, and provider dispatch
- `toolcalling/providers/` — `OpenAIProvider` plus Anthropic/Gemini/Local adapters (streaming-aware)
- `toolcalling/examples/bbox.py` — OpenAI Vision bounding-box tool factory (with offline mock)
- `toolcalling/cli.py` — console entry (`toolcalling`) for listing tools, running prompts, or chatting interactively

## Agent Loop

1. Build system prompt with `PromptBuilder` using tool schemas.
2. Call provider (`Provider.complete`) with system prompt + history.
3. Parse model output via `ToolCallParser`.
4. If no tool call, return final assistant message.
5. If tool call found, validate/execute tool, append tool result, and continue until `max_iterations` is reached.

`AgentConfig` controls `model`, `temperature`, `max_tokens`, `max_iterations`, streaming, retries/backoff, timeouts, and verbosity.

## Providers

`Provider` supports `complete` and `stream`. `OpenAIProvider` formats messages (including vision payloads) for Chat Completions. Anthropic/Gemini/Local adapters share the interface; Local echoes for offline use.

## Bounding-Box Tool

`toolcalling/examples/bbox.py` exposes `create_bounding_box_tool()` and `detect_bounding_box_impl`. The tool:

- Resolves the image path (relative to `assets/` or absolute)
- Supports deterministic offline testing via `TOOLCALLING_BBOX_MOCK_JSON` pointing to a golden JSON payload
- Calls OpenAI Vision with a structured JSON prompt when not mocked
- Validates normalized coordinates
- Draws a labeled red box with Pillow and saves `<name>_with_bbox.png`
- Returns a JSON string with success, coordinates (normalized + pixel), output path, and confidence

## CLI

`toolcalling` console script:

- `list-tools` — show available tools (echo + detect_bounding_box)
- `run` — execute one prompt with a chosen provider/model, optional image, optional single-tool restriction, streaming output, and dry-run (prompt preview)
- `chat` — interactive REPL with history; supports streaming output

## Demo Runner

`scripts/chat.py` wires the bounding-box tool into the agent loop. Default run processes `assets/environment.png`; `--interactive` enables a simple REPL.

## Testing

`tests/test_framework.py` uses a `FakeProvider` to avoid network calls and covers:

- Role/message basics (including optional image encoding)
- Tool schema + validation
- TOOL_CALL parsing in fenced/mixed blocks with size limits
- Agent executing a tool, streaming callbacks, and retry behavior
- Local provider streaming and bounding-box mock path (no network)

## Packaging

`pyproject.toml` defines metadata, dependencies (`openai`, `Pillow`), provider extras, console scripts, and a src layout. Install editable with `pip install -e .`. CI (`.github/workflows/ci.yml`) runs tests, build, and twine check; tags trigger TestPyPI/PyPI publish when tokens are present.

## Future Work

- Add conversation summarization and long-context windowing
- Expand test coverage for error branches and multi-turn flows
- Improve provider-specific formatting (vision for all adapters)
- Publish wheels to PyPI with release automation
