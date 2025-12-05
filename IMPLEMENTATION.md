# AI Tool Calling Framework - Implementation Notes

## Overview

The project is now a provider-agnostic Python library packaged under `toolcalling/`. It implements a TOOL_CALL contract, pluggable providers, a hardened parser/prompt, a CLI, and an example bounding-box tool that uses OpenAI Vision.

## Package Layout

- `toolcalling/types.py` — `Role`, `Message`, `ToolCall`
- `toolcalling/tools.py` — `Tool`, `ToolParameter`, schema + validation
- `toolcalling/prompt.py` — system prompt builder embedding tool schemas
- `toolcalling/parser.py` — robust TOOL_CALL parser (fenced blocks, lenient JSON)
- `toolcalling/agent.py` — iterative loop with configurable limits and provider dispatch
- `toolcalling/providers/` — `OpenAIProvider` plus Anthropic/Gemini/Local stubs
- `toolcalling/examples/bbox.py` — OpenAI Vision bounding-box tool factory
- `toolcalling/cli.py` — console entry (`toolcalling`) for listing tools and running prompts

## Agent Loop

1. Build system prompt with `PromptBuilder` using tool schemas.
2. Call provider (`Provider.complete`) with system prompt + history.
3. Parse model output via `ToolCallParser`.
4. If no tool call, return final assistant message.
5. If tool call found, validate/execute tool, append tool result, and continue until `max_iterations` is reached.

`AgentConfig` controls `model`, `temperature`, `max_tokens`, `max_iterations`, and verbosity.

## Providers

`Provider` is a protocol with a single `complete` method. `OpenAIProvider` formats messages (including vision payloads) for Chat Completions. Anthropic/Gemini/Local stubs share the interface and raise clear errors until implemented.

## Bounding-Box Tool

`toolcalling/examples/bbox.py` exposes `create_bounding_box_tool()` and `detect_bounding_box_impl`. The tool:
- Resolves the image path (relative to `assets/` or absolute)
- Calls OpenAI Vision with a structured JSON prompt
- Validates normalized coordinates
- Draws a labeled red box with Pillow and saves `<name>_with_bbox.png`
- Returns a JSON string with success, coordinates (normalized + pixel), output path, and confidence

## CLI

`toolcalling` console script:
- `list-tools` — show available tools (echo + detect_bounding_box)
- `run` — execute one prompt with a chosen provider/model, optional image, and optional single-tool restriction

## Demo Runner

`scripts/chat.py` wires the bounding-box tool into the agent loop. Default run processes `assets/dog.png`; `--interactive` enables a simple REPL.

## Testing

`tests/test_framework.py` uses a `FakeProvider` to avoid network calls and covers:
- Role/message basics (including optional image encoding)
- Tool schema + validation
- TOOL_CALL parsing in fenced blocks
- Agent executing a tool and returning a final response

## Packaging

`pyproject.toml` defines metadata, dependencies (`openai`, `Pillow`), console scripts, and a src layout. Install editable with `pip install -e .`.

## Future Work

- Implement real Anthropic/Gemini/local providers
- Add streaming responses and conversation summarization
- Expand test coverage for error branches and multi-turn flows
- Publish wheels to PyPI with release automation
