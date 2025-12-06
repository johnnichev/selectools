# Toolcalling User Guide

## Quickstart

- Install: `pip install -e .` (or `pip install .[providers]` to include Anthropic/Gemini SDKs).
- Env: set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` as needed. Local provider requires none.
- First call: define a tool (`@tool` or `Tool`) and run `Agent.run([Message(role=Role.USER, content="...")])`.
- Streaming: set `AgentConfig.stream=True` or CLI `--stream`.
- Dry-run: CLI `toolcalling run --dry-run ...` prints the composed system prompt.

## Providers

- OpenAI: supports streaming; vision via Chat Completions (`image_url`). Timeout default 30s, retries 2, backoff linear.
- Anthropic: streaming supported; vision depends on model availability. Requires `ANTHROPIC_API_KEY`.
- Gemini: streaming supported; vision availability model-dependent. Requires `GEMINI_API_KEY`.
- Local: no network; echoes latest user text, useful for offline/manual tests.
- Rate limits: agent detects `rate limit`/`429` in `ProviderError` and backs off (`rate_limit_cooldown_seconds`, default 5s scaled by attempt) plus retry backoff.
- Timeouts: `AgentConfig.request_timeout` for provider calls; per-tool timeout via `tool_timeout_seconds`.

### Vision support (at a glance)

- OpenAI: Chat Completions with `image_url` (e.g., `gpt-4o`).
- Anthropic: vision depends on model tier/availability; check Claude model docs.
- Gemini: model-dependent; check Gemini model docs for vision-enabled variants.
- Local: no vision (text echo only).

## Tools

- Define via `Tool` or `@tool` decorator with schema inference.
- Use `ToolRegistry` to register and reuse tools, optionally via `registry.tool` decorator.
- Inject config/auth with `injected_kwargs` or `config_injector` when constructing `Tool`.
- Parameter typing: annotations map to JSON schema types; defaults make params optional.

## Agent

- Key config: `model`, `temperature`, `max_tokens`, `max_iterations`, `stream`, `request_timeout`, `max_retries`, `retry_backoff_seconds`, `rate_limit_cooldown_seconds`, `tool_timeout_seconds`.
- Loop: build system prompt with tool schemas, call provider, parse TOOL_CALL, execute tool (with timeout), append results, repeat until completion or iteration cap.
- Errors: provider errors return an assistant message; tool errors are appended as tool responses and the loop continues.

## CLI

- `toolcalling list-tools` — list available tools.
- `toolcalling run --provider <openai|anthropic|gemini|local> --model <name> --prompt "..."`
  - Flags: `--image` (vision), `--tool` (restrict), `--stream`, `--dry-run`, `--timeout`, `--retries`, `--backoff`, `--max-iterations`, `--max-tokens`.
- `toolcalling chat` — interactive REPL with history; supports `--stream`.

## Bounding Box Demo

- Vision run: `toolcalling run --prompt "Find the object" --image assets/environment.png --tool detect_bounding_box`.
- Offline/mock: set `TOOLCALLING_BBOX_MOCK_JSON=tests/fixtures/bbox_mock.json` to bypass network.
- Output: writes `<image>_with_bbox.png` with a red box; returns JSON with normalized + pixel coords.

## Examples & Recipes

- `examples/search_weather.py`: search + weather mock tools using `ToolRegistry` and Local provider.
- Recipes: use `@tool` for schema inference, inject auth/config per tool, enable streaming for token-by-token output, use `tool_timeout_seconds` for long-running tools.

## Testing

- Run: `python tests/test_framework.py` (covers parsing, agent loop, streaming, retries, provider mocks, bbox mock path, CLI streaming).
- Add provider tests: mock SDK modules in `sys.modules` and set API key envs; assert complete/stream behavior without network.
- Add tool/CLI tests by invoking `run_agent` with Local provider and capturing stdout.

## Release

- Versioning: bump `pyproject.toml` version and update `CHANGELOG.md`.
- CI: `.github/workflows/ci.yml` runs tests, build, `twine check`. Tags `v*` publish to TestPyPI/PyPI when tokens are configured.
- Required secrets: `TEST_PYPI_API_TOKEN`, `PYPI_API_TOKEN`.
