# Changelog

## [0.4.0] - 2025-12-08

### 🎉 Major Release: Async Support & Provider Improvements

This release brings full async/await support, real provider implementations, conversation memory, and comprehensive production testing.

### ✨ New Features

#### Async Support
- **`Agent.arun()`** - Async agent execution for high-performance applications
- **Async tools** - Define tools with `async def`, mixed seamlessly with sync tools
- **Async providers** - Full async support for OpenAI, Anthropic, and Gemini
- **Concurrent execution** - Validated with 100+ concurrent users
- **FastAPI integration** - Native async support for web frameworks

#### Conversation Memory
- **`ConversationMemory`** - Manage conversation history across turns
- **Message limits** - Configurable `max_messages` and `max_tokens`
- **Auto-persistence** - Automatically saves messages during agent execution
- **Token-aware** - Estimates and enforces token limits

#### Provider Improvements
- **Real Anthropic Provider** - Full SDK integration with `anthropic` package
- **Real Gemini Provider** - Full SDK integration with `google-generativeai` package
- **Async streaming** - All providers support async streaming
- **Unified interface** - Consistent API across all providers

### 🔧 Improvements
- Removed Pillow dependency (lighter package, only 3 core dependencies)
- Removed bbox example (was the only use of Pillow)
- Moved Anthropic and Gemini to required dependencies
- Updated provider protocol with `supports_async` flag
- ThreadPoolExecutor fallback for sync tools in async context

### 📚 Documentation
- Added async usage examples in README
- Created `examples/async_agent_demo.py` with FastAPI integration
- Updated ROADMAP.md with completed features
- Added comprehensive production readiness report

### 🧪 Testing
- **55 total tests** across 4 test suites (was 27)
- Added 13 edge case tests
- Added 8 integration tests  
- Added 7 high-concurrency stress tests
- **Validated performance**: 10,000+ req/s throughput
- **Stress tested**: 100 concurrent users, 1,000+ sustained requests
- **Zero failures** across all test scenarios

### 🚀 Performance
- Framework overhead: <0.1ms per request
- Throughput: 10,000-15,000 req/s (framework only)
- Concurrency: Validated with 100+ simultaneous users
- Memory: Stable under high load (tested with 250+ conversations)

### 📦 Dependencies
```toml
dependencies = [
    "openai>=1.30.0,<2.0.0",
    "anthropic>=0.28.0,<1.0.0",
    "google-generativeai>=0.8.3,<1.0.0",
]
```

### 🔄 Migration Guide

#### Using Async
```python
# Before (0.3.x)
response = agent.run([Message(content="Hello")])

# After (0.4.0) - async
response = await agent.arun([Message(content="Hello")])

# Old sync code still works!
response = agent.run([Message(content="Hello")])
```

#### Conversation Memory
```python
from selectools import Agent, ConversationMemory

memory = ConversationMemory(max_messages=20)
agent = Agent(tools=[...], memory=memory)

# Memory automatically persists across calls
response = await agent.arun([Message(content="Turn 1")])
response = await agent.arun([Message(content="Turn 2")])
# History is maintained automatically
```

### ⚠️ Breaking Changes
None! This release is 100% backward compatible.

### 📊 Stats
- **New files**: 9
- **Modified files**: 13
- **Deleted files**: 2
- **Lines added**: ~3,000
- **Test coverage**: 55 comprehensive tests

## [0.3.1] - 2025-12-07

### Changed
- Add automated release scripts and tooling


## 0.3.0 (2025-12-07)

### 🚨 Breaking Changes
- **Module renamed**: `toolcalling` → `selectools`
- All imports must be updated: `from selectools import ...`
- CLI command renamed: `toolcalling` → `selectools`
- Environment variable renamed: `TOOLCALLING_BBOX_MOCK_JSON` → `SELECTOOLS_BBOX_MOCK_JSON`

### Migration Guide
```python
# Before (0.2.x)
from toolcalling import Agent, tool
from toolcalling.providers.openai_provider import OpenAIProvider

# After (0.3.0)
from selectools import Agent, tool
from selectools.providers.openai_provider import OpenAIProvider
```

## 0.2.1 (2025-12-07)

### Documentation
- Updated package description to better explain the library's purpose
- Clearer messaging: "Build AI agents that call your custom Python functions"
- Emphasizes LLM integration and custom tool creation

## 0.2.0 (2025-12-07)

### 🎉 Published to PyPI
- **Package name**: `selectools`
- **PyPI URL**: https://pypi.org/project/selectools/
- **License**: LGPL-3.0-or-later (allows commercial use via import, modifications must be shared)

### Features
- Added Anthropic/Gemini Local adapters with streaming-ready interface.
- Agent: streaming callbacks, retries/backoff with rate-limit detection, tool execution timeouts.
- Parser: balanced JSON extraction, size limits, mixed/fenced handling.
- CLI: streaming/dry-run flags, chat mode, improved defaults; local streaming test.
- Examples: search/weather demo using `ToolRegistry` and `@tool`.

### Documentation
- Comprehensive README with real-world examples (customer support, vision AI, research assistant, etc.)
- Detailed "Why This Library Stands Out" section highlighting production-ready features
- Future improvements roadmap with competitive positioning
- License section explaining usage rights and restrictions
- Installation instructions for PyPI and development setup

### Packaging
- Added LGPL-3.0 license with full license text
- Project URLs (homepage, repository, issues, documentation)
- Python version classifiers (3.9, 3.10, 3.11, 3.12)
- Development status: Beta
- Pinned provider extras, updated OpenAI minimum version

