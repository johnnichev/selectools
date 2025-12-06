# Changelog

## 0.2.0
- Added Anthropic/Gemini Local adapters with streaming-ready interface.
- Agent: streaming callbacks, retries/backoff with rate-limit detection, tool execution timeouts.
- Parser: balanced JSON extraction, size limits, mixed/fenced handling.
- CLI: streaming/dry-run flags, chat mode, improved defaults; local streaming test.
- Examples: search/weather demo using `ToolRegistry` and `@tool`.
- Docs: comprehensive user guide (providers, tools, agent, CLI, demo, testing, release).
- Packaging: pinned provider extras, updated OpenAI minimum version.
- CI: test/build/twine-check workflow; publish on tags when tokens set.

