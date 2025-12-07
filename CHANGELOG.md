# Changelog

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

