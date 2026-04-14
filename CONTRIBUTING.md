# Contributing to Selectools

Thank you for your interest in contributing to Selectools! We welcome contributions from the community.

**Current Version:** v0.22.0
**Test Status:** 5343 tests collected (95% coverage)
**Python:** 3.9 – 3.13

## Getting Started

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/johnnichev/selectools.git
cd selectools
```

2. **Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e .
pip install -e ".[dev]"  # Install dev dependencies
```

4. **Set up pre-commit hooks (recommended)**

Pre-commit hooks automatically format code, check for issues, and ensure code quality before each commit:

```bash
# Install pre-commit hooks
pre-commit install

# Optionally run on all files to test
pre-commit run --all-files
```

The hooks will automatically run on staged files when you commit. They include:
- **Ruff** - Formatting + linting + import sorting (replaces Black + isort + flake8)
- **mypy** - Type checking
- **bandit** - Security checks

5. **Set up API keys for testing**

```bash
# LLM Providers
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"  # Optional
export GEMINI_API_KEY="your-key-here"     # Optional

# Embedding Providers (for RAG features)
export VOYAGE_API_KEY="your-key-here"     # Optional (Anthropic embeddings)
export COHERE_API_KEY="your-key-here"     # Optional

# Vector Stores (for RAG features)
export PINECONE_API_KEY="your-key-here"   # Optional
export PINECONE_ENV="your-env-here"       # Optional
```

## Available Scripts

Similar to `npm run` scripts, here are the common commands for this project:

### Testing

```bash
# Run all tests (5343 tests)
pytest tests/ -v

# Run tests quietly (summary only)
pytest tests/ -q

# Run specific test areas
pytest tests/agent/ -v             # Agent tests
pytest tests/rag/ -v               # RAG tests
pytest tests/tools/ -v             # Tool tests
pytest tests/core/ -v              # Core framework tests

# Run tests matching a pattern
pytest tests/ -k "test_tool" -v
pytest tests/ -k "rag" -v

# Run with coverage (if installed)
pytest tests/ --cov=src/selectools --cov-report=html

# Skip end-to-end tests (require real API keys)
pytest tests/ -k "not e2e" -v
```

### Code Quality

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run individual tools
black src/ tests/ examples/          # Format code
isort src/ tests/ examples/          # Sort imports
flake8 src/                          # Lint code
mypy src/                            # Type check
bandit -r src/                       # Security scan
```

### Examples

Examples are numbered by difficulty (see `examples/` for the full list):

```bash
# Beginner — no API key needed
python examples/01_hello_world.py
python examples/02_search_weather.py
python examples/03_toolbox.py

# Intermediate — requires OPENAI_API_KEY
python examples/04_conversation_memory.py
python examples/09_caching.py
python examples/13_dynamic_tools.py

# RAG — requires OPENAI_API_KEY + selectools[rag]
python examples/14_rag_basic.py
python examples/18_hybrid_search.py
python examples/19_advanced_chunking.py
```

### Development Scripts

```bash
# Quick smoke test for providers
python scripts/smoke_cli.py

# Test conversation memory with OpenAI
python scripts/test_memory_with_openai.py
```

### Release

```bash
# Release a new version (recommended)
python scripts/release.py --version 0.20.2

# Dry run (see what would happen)
python scripts/release.py --version 0.20.2 --dry-run

# Or use the bash script
./scripts/release.sh 0.20.2
```

See `scripts/README.md` for detailed release instructions.

### Building

```bash
# Build the package
python -m build

# Check the built package
twine check dist/*
```

## Running Tests

Run the test suite to ensure everything works:

```bash
pytest tests/ -v
```

All tests should pass before submitting a pull request.

## Code Style

We follow standard Python conventions and use automated tools to enforce consistency:

### Style Guidelines

- **PEP 8** style guide (enforced by flake8)
- **Type hints** for function signatures (checked by mypy)
- **Docstrings** for public APIs (Google style preferred)
- **Clear variable names** over abbreviations
- **Line length**: 100 characters (configured in Black and isort)

### Automated Formatting

If you installed pre-commit hooks (recommended), your code will be automatically formatted on commit.
You can also run formatters manually:

```bash
# Format code with Black
black src/ tests/ examples/

# Sort imports with isort
isort src/ tests/ examples/

# Check for linting issues
flake8 src/

# Type check
mypy src/
```

### Example

```python
def execute_tool(tool: Tool, arguments: dict[str, Any]) -> str:
    """
    Execute a tool with the provided arguments.

    Args:
        tool: The tool to execute
        arguments: Dictionary of argument names to values

    Returns:
        The tool's output as a string

    Raises:
        ToolExecutionError: If the tool fails to execute
    """
    # Implementation here
    pass
```

## Project Structure

```
selectools/
├── src/selectools/              # Main package
│   ├── __init__.py             # Public exports
│   ├── agent/                  # Agent loop and orchestration
│   │   ├── core.py             # Agent class
│   │   └── config.py           # AgentConfig
│   ├── structured.py           # Structured output parsing & validation
│   ├── trace.py                # AgentTrace & TraceStep
│   ├── policy.py               # ToolPolicy (allow/review/deny)
│   ├── env.py                  # Environment variable loading
│   ├── exceptions.py           # Custom exception classes
│   ├── memory.py               # ConversationMemory (tool-pair-aware)
│   ├── models.py               # Model registry (152 models)
│   ├── parser.py               # ToolCallParser
│   ├── pricing.py              # LLM pricing data and cost calculation
│   ├── prompt.py               # PromptBuilder
│   ├── tools.py                # Tool, @tool, ToolRegistry
│   ├── types.py                # Message, Role, StreamChunk, AgentResult
│   ├── usage.py                # UsageTracker (tokens, costs, analytics)
│   ├── cache.py                # InMemoryCache (LRU+TTL)
│   ├── cache_redis.py          # RedisCache
│   ├── providers/              # LLM provider adapters
│   │   ├── base.py             # Provider interface
│   │   ├── openai_provider.py  # OpenAI
│   │   ├── anthropic_provider.py # Anthropic
│   │   ├── gemini_provider.py  # Google Gemini
│   │   ├── ollama_provider.py  # Ollama local models
│   │   ├── fallback.py         # FallbackProvider (auto-failover)
│   │   └── stubs.py            # LocalProvider / test stubs
│   ├── embeddings/             # Embedding providers
│   ├── rag/                    # RAG: vector stores, chunking, loaders
│   └── toolbox/                # 33 pre-built tools
├── tests/                      # Test suite (5343 tests, 95% coverage)
│   ├── agent/                  # Agent tests
│   ├── rag/                    # RAG tests
│   ├── tools/                  # Tool tests
│   ├── core/                   # Core framework tests
│   └── integration/            # E2E tests (require API keys)
├── examples/                   # 95 numbered examples
├── docs/                       # Detailed documentation
│   ├── QUICKSTART.md           # 5-minute getting started
│   ├── ARCHITECTURE.md         # Architecture overview
│   └── modules/                # Per-module docs
└── scripts/                    # Release and dev scripts
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/johnnichev/selectools/issues)
2. If not, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Relevant code snippets or error messages

### Suggesting Features

1. Check [existing issues](https://github.com/johnnichev/selectools/issues) for similar suggestions
2. Create a new issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Any alternatives you've considered
   - Examples of how it would be used

### Submitting Pull Requests

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

2. **Make your changes**
   - Write clear, focused commits
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**

```bash
pytest tests/ -x -q                  # All tests
pytest tests/ -k "not e2e" -x -q     # Skip E2E (no API keys needed)
```

4. **Commit with clear messages**

```bash
git commit -m "Add support for streaming tool results"
```

Good commit messages:
- Use present tense ("Add feature" not "Added feature")
- Be specific and descriptive
- Reference issues when applicable (#123)

5. **Push and create a pull request**

```bash
git push origin feature/your-feature-name
```

Then open a PR on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/examples if applicable

## Areas for Contribution

We especially welcome contributions in these areas:

### 🔧 **New Providers**
- Add support for new LLM providers (Cohere, AI21, etc.)
- Improve existing provider implementations
- Add vision support to more providers
- Add new embedding providers

### 🛠️ **New Tools**
- Pre-built tools for common use cases
- Integration with popular APIs and services
- Example tools demonstrating best practices

### 🗄️ **RAG & Vector Stores**
- Add new vector store integrations (Weaviate, Qdrant, Milvus)
- Add new reranker integrations
- Improve agentic chunking strategies
- Performance benchmarks for different vector stores

### 📚 **Documentation**
- Improve README examples
- Add tutorials and guides (especially for RAG features!)
- Fix typos and clarify confusing sections
- Add comparison guides (vs LangChain, LlamaIndex)

### 🧪 **Testing**
- Increase test coverage (currently 5343 tests collected!)
- Add performance benchmarks
- Improve E2E test stability with retry/rate-limit handling

### 🐛 **Bug Fixes**
- Fix reported issues
- Improve error messages
- Handle edge cases

## Adding a New Provider

To add support for a new LLM provider:

1. **Create a new provider file**

```python
# src/selectools/providers/your_provider.py

import os
from typing import Iterator

from .base import Provider
from ..exceptions import ProviderConfigurationError
from ..types import Message
from ..usage import UsageStats
from ..pricing import calculate_cost


class YourProvider(Provider):
    def __init__(self, api_key: str = None, default_model: str = "model-name"):
        self.api_key = api_key or os.getenv("YOUR_PROVIDER_API_KEY")
        if not self.api_key:
            raise ProviderConfigurationError(
                "API key is required",
                details={"env_var": "YOUR_PROVIDER_API_KEY"}
            )
        self.default_model = default_model

    def complete(
        self, messages: list[Message], model: str = None, **kwargs
    ) -> tuple[str, UsageStats]:
        model = model or self.default_model
        # Call your provider's API here
        # response = your_api_call(...)

        # Extract usage stats
        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=model,
            provider="your_provider",
            cost_usd=calculate_cost(model, prompt_tokens, completion_tokens),
        )
        return response.text, usage

    def stream(
        self, messages: list[Message], model: str = None, **kwargs
    ) -> Iterator[str]:
        # Streaming only yields text chunks (no usage stats)
        for chunk in your_streaming_api_call(...):
            yield chunk.text
```

2. **Add tests**

```python
# tests/providers/test_your_provider.py

def test_your_provider_complete():
    provider = YourProvider(api_key="fake-key")
    # Add test cases — see existing tests/providers/ for patterns
    pass
```

3. **Update documentation**
   - Add to README provider list
   - Add usage example
   - Update CHANGELOG

## Adding a New Tool

To contribute a new pre-built tool to `src/selectools/toolbox/`:

1. **Create the tool** with the `@tool` decorator

```python
# src/selectools/toolbox/your_tools.py

from selectools import tool


@tool()
def your_tool(param1: str, param2: int = 10) -> str:
    """One-line description of what the tool does.

    Longer multi-line docstring becomes the tool's description in the
    LLM-facing schema. Be specific about what the tool does and when
    to use it.

    Args:
        param1: What this parameter is for.
        param2: Optional. What this parameter is for. Default 10.

    Returns:
        Description of the return value.
    """
    # Your implementation here
    return f"Result: {param1} with {param2}"
```

The `@tool()` decorator (note the parentheses — they're required) introspects
the function signature and docstring to build the JSON schema automatically.
No manual `Tool` / `ToolParameter` construction needed.

2. **Use the tool with an Agent**

```python
from selectools import Agent
from selectools.toolbox.your_tools import your_tool

agent = Agent(tools=[your_tool], provider=OpenAIProvider())
result = agent.run("Use your_tool with param1='hello'")
```

3. **Add tests** in `tests/toolbox/test_your_tools.py`

4. **Add an example** in `examples/NN_your_feature.py` (zero-padded number)

5. **Update documentation**:
   - Add the tool to `docs/modules/TOOLBOX.md`
   - Bump the tool count in `docs/llms.txt`, `landing/index.html`, and `CONTRIBUTING.md`

## Adding RAG Features

### Adding a New Vector Store

To add support for a new vector database:

1. **Create a new vector store implementation**

```python
# src/selectools/rag/stores/your_store.py

from typing import List, Optional
from ..vector_store import VectorStore, Document, SearchResult

class YourVectorStore(VectorStore):
    """Your vector database implementation."""

    def __init__(self, embedder, **config):
        self.embedder = embedder
        # Initialize your vector DB client
        # self.client = your_db.Client(**config)

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """Add documents to the store."""
        # Generate embeddings if not provided
        if embeddings is None:
            texts = [doc.text for doc in documents]
            embeddings = self.embedder.embed_texts(texts)

        # Insert into your vector DB
        # ids = self.client.insert(...)
        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[dict] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        # results = self.client.search(query_embedding, top_k, filter)
        # return [SearchResult(document=..., score=...) for r in results]
        pass

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        # self.client.delete(ids)
        pass
```

2. **Register in the factory**

```python
# src/selectools/rag/vector_store.py

@staticmethod
def create(store_type: str, embedder, **kwargs) -> "VectorStore":
    if store_type == "your_store":
        from .stores.your_store import YourVectorStore
        return YourVectorStore(embedder, **kwargs)
    # ... existing stores
```

3. **Add tests**

```python
# tests/test_vector_stores_crud.py

class TestYourVectorStore:
    def test_add_and_search(self, mock_embedder):
        store = VectorStore.create("your_store", embedder=mock_embedder)
        docs = [Document(text="test", metadata={})]
        ids = store.add_documents(docs)

        query_emb = mock_embedder.embed_query("test")
        results = store.search(query_emb, top_k=1)

        assert len(results) == 1
        assert results[0].document.text == "test"
```

### Adding a New Embedding Provider

1. **Create the provider**

```python
# src/selectools/embeddings/your_provider.py

from typing import List
from .provider import EmbeddingProvider

class YourEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "default-model", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("YOUR_API_KEY")
        # self.client = your_sdk.Client(api_key=self.api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        # response = self.client.embeddings.create(input=texts, model=self.model)
        # return [emb.embedding for emb in response.data]
        pass

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return self.embed_texts([query])[0]
```

2. **Add model definitions to `models.py`**

3. **Add tests and update documentation**

## Code Review Process

1. A maintainer will review your PR
2. They may request changes or ask questions
3. Once approved, your PR will be merged
4. Your contribution will be included in the next release!

## Questions?

- Open an issue for questions about contributing
- Check existing issues and PRs for similar discussions
- Be patient and respectful—we're all volunteers!

## License

By contributing to Selectools, you agree that your contributions will be licensed under the Apache-2.0 license.

---

Thank you for contributing to Selectools! 🎉
