# Changelog

All notable changes to selectools will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2024-12-10

### Added

- **Model Registry System** - Single source of truth for all 120 models
  - New `models.py` module with `ModelInfo` dataclass containing complete model metadata
  - Typed model constants for IDE autocomplete: `OpenAI.GPT_4O`, `Anthropic.SONNET_4_5`, etc.
  - 64 OpenAI models (GPT-5, GPT-4o, o-series, GPT-4, GPT-3.5)
  - 18 Anthropic models (Claude 4.5, 4.1, 4, 3.7, 3.5, 3)
  - 25 Gemini models (Gemini 3, 2.5, 2.0, 1.5, 1.0, Gemma)
  - 13 Ollama models (Llama, Mistral, Phi, etc.)
- **Rich Model Metadata** - Each model includes:
  - Pricing (prompt/completion costs per 1M tokens)
  - Context window size
  - Maximum output tokens
  - Model type (chat, audio, multimodal)
  - Provider name
- **New Public API** exports:
  - `models` module
  - `ModelInfo` dataclass
  - `ALL_MODELS` list
  - `MODELS_BY_ID` dict
  - `OpenAI`, `Anthropic`, `Gemini`, `Ollama` classes
- **Updated Documentation**
  - New "Model Selection with Autocomplete" section in README
  - All code examples updated to use typed constants
  - 12 example files migrated to demonstrate new pattern

### Changed

- **Pricing Module Refactored** - Now derives from `models.py` instead of hardcoded dict
- **All Provider Defaults** - Use typed constants instead of hardcoded strings
- **Backward Compatible** - Old code using `PRICING` dict still works
- Updated OpenAI pricing with 70+ models including GPT-5, o3-pro, latest GPT-4o variants
- Updated Anthropic pricing with Claude 4.5, 4.1, 4 series
- Updated Gemini pricing with Gemini 3, 2.5, 2.0 series

### Fixed

- Test suite updated to handle frozen dataclass immutability correctly

## [0.6.1] - 2024-12-09

### Added

- **Streaming Tool Results** - Tools can now yield results progressively
  - Support for `Generator[str, None, None]` return types (sync)
  - Support for `AsyncGenerator[str, None]` return types (async)
  - Real-time chunk callbacks via `on_tool_chunk` hook
  - Streaming metrics in analytics (chunk counts, streaming calls)
- **Toolbox Streaming Tools**
  - `read_file_stream` - Stream file content line by line
  - `process_csv_stream` - Stream CSV content row by row
- Examples: `streaming_tools_demo.py` with 5 comprehensive scenarios

### Changed

- Analytics now track `total_chunks` and `streaming_calls` for streaming tools
- Tool execution supports progressive result delivery

## [0.6.0] - 2024-12-08

### Added

- **Local Model Support** - Ollama provider for local LLM execution
  - Zero cost (all Ollama models priced at $0.00)
  - Privacy-preserving (no data sent to cloud)
  - OpenAI-compatible API
  - Support for llama3.2, mistral, codellama, phi, qwen, etc.
- **Tool Usage Analytics** - Comprehensive metrics tracking
  - Call frequency, success/failure rates, execution duration
  - Parameter usage patterns, cost attribution per tool
  - Export to JSON/CSV with `export_to_json()` and `export_to_csv()`
  - Enable with `AgentConfig(enable_analytics=True)`
- Examples: `ollama_demo.py`, `tool_analytics_demo.py`

### Changed

- Pricing module now includes 13 Ollama models (all free)

## [0.5.2] - 2024-12-07

### Added

- **Tool Validation at Registration** - Validates tool definitions when created
  - Name validation (valid Python identifier, 1-64 chars)
  - Description validation (10-1024 chars, required)
  - Parameter validation (names, types, required fields, duplicates)
  - Signature mismatch detection
- **Observability Hooks** - 10 lifecycle callbacks for monitoring
  - `on_agent_start`, `on_agent_end`
  - `on_tool_start`, `on_tool_end`, `on_tool_error`
  - `on_llm_start`, `on_llm_end`, `on_llm_error`
  - `on_error`, `on_max_iterations`
- Example: `v0_5_2_demo.py` with 8 scenarios

### Changed

- Improved error messages with validation details
- Tools now validate at creation time, not runtime

## [0.5.1] - 2024-12-06

### Added

- **Pre-built Tool Library** - 27 production-ready tools in 5 categories
  - **File Tools** (7): read_file, write_file, list_directory, etc.
  - **Web Tools** (4): fetch_url, search_web, extract_html_text, etc.
  - **Data Tools** (8): parse_json, parse_csv, calculate, etc.
  - **DateTime Tools** (3): get_current_time, parse_datetime, format_datetime
  - **Text Tools** (5): count_words, find_pattern, replace_text, etc.
- **ToolRegistry** - Manage and filter tools by category
- Example: `toolbox_demo.py`

### Changed

- All toolbox tools include comprehensive docstrings and examples

## [0.5.0] - 2024-12-05

### Added

- **Better Error Messages** - PyTorch-style helpful errors
  - Custom exceptions: `ToolValidationError`, `ToolExecutionError`, `ProviderConfigurationError`, `MemoryLimitExceededError`
  - Fuzzy matching for parameter typos with suggestions
  - Context-aware error messages with fix suggestions
- **Cost Tracking** - Automatic token counting and cost estimation
  - `UsageStats` dataclass with token counts and costs
  - `AgentUsage` for aggregated multi-turn usage
  - Configurable cost warnings via `cost_warning_threshold`
  - Pricing for 120+ models across OpenAI, Anthropic, Gemini, Ollama
- **Gemini SDK Migration** - Updated to `google-genai` v1.0+
- Example: `cost_tracking_demo.py`

### Changed

- All providers now return `(content, usage_stats)` tuples from `complete()` methods
- Streaming methods only yield content (no usage stats during streaming)

## [0.4.0] - 2024-11-15

### Added

- **Conversation Memory** - Multi-turn context management
  - `ConversationMemory` class with configurable max_messages
  - Automatic context injection for all turns
  - FIFO eviction when memory limit reached
- **Async Support** - Full async/await support
  - `Agent.arun()` for async execution
  - Async tool functions supported
  - Async providers (`acomplete`, `astream`)
- **Real Provider Integrations**
  - `AnthropicProvider` - Full Anthropic SDK integration
  - `GeminiProvider` - Full Google Gemini SDK integration
- Example: `async_agent_demo.py`, `conversation_memory_demo.py`

### Changed

- All providers support both sync and async operations
- Improved streaming support across all providers

## [0.3.0] - 2024-11-01

### Added

- Initial public release
- OpenAI provider integration
- Basic tool-calling functionality
- Simple agent implementation

---

## Release Links

- [0.7.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.7.0)
- [0.6.1 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.6.1)
- [0.6.0 Release Notes](https://github.com/johnnichev/selectools/releases/tag/v0.6.0)

For detailed migration guides and breaking changes, see the [documentation](https://github.com/johnnichev/selectools).
