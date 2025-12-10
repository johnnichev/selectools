# Contributing to Selectools

Thank you for your interest in contributing to Selectools! We welcome contributions from the community.

**Current Version:** v0.8.0 (Embeddings & RAG Support)
**Test Status:** ✅ 463 tests passing (100%)
**Next Release:** v0.9.0 (Hybrid Search, Reranking)

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
- **Black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
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
# Run all tests (463 tests)
pytest tests/ -v

# Run tests quietly (summary only)
pytest tests/ -q

# Run specific test file
pytest tests/test_framework.py -v

# Run RAG-specific tests
pytest tests/test_embedding_providers.py -v
pytest tests/test_vector_stores_crud.py -v
pytest tests/test_rag_workflow.py -v

# Run tests matching a pattern
pytest tests/ -k "test_tool" -v
pytest tests/ -k "rag" -v

# Run with coverage (if installed)
pytest tests/ --cov=src/selectools --cov-report=html

# Skip end-to-end tests (require real API keys)
pytest tests/ -m "not e2e" -v
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

```bash
# Basic weather search example
python examples/search_weather.py

# Async agent demo
python examples/async_agent_demo.py

# Conversation memory demo
python examples/conversation_memory_demo.py

# Cost tracking demo
python examples/cost_tracking_demo.py

# Customer support bot example
python examples/customer_support_bot.py

# Data analysis agent example
python examples/data_analysis_agent.py

# RAG (Retrieval-Augmented Generation) examples
python examples/rag_basic_demo.py           # Basic RAG with in-memory store
python examples/rag_advanced_demo.py        # Advanced RAG with PDF + SQLite
python examples/semantic_search_demo.py     # Multi-provider semantic search
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
python scripts/release.py --version 0.5.1

# Dry run (see what would happen)
python scripts/release.py --version 0.5.1 --dry-run

# Or use the bash script
./scripts/release.sh 0.5.1
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
│   ├── agent.py                # Agent loop and orchestration
│   ├── cli.py                  # CLI interface
│   ├── env.py                  # Environment variable loading
│   ├── exceptions.py           # Custom exception classes
│   ├── models.py               # Model registry (120+ models)
│   ├── parser.py               # TOOL_CALL parser
│   ├── pricing.py              # LLM pricing data and cost calculation
│   ├── prompt.py               # Prompt builder
│   ├── tools.py                # Tool definitions and registry
│   ├── types.py                # Core types (Message, Role, etc.)
│   ├── usage.py                # Usage tracking (tokens, costs, analytics)
│   ├── providers/              # LLM provider adapters
│   │   ├── base.py             # Provider interface
│   │   ├── openai_provider.py  # OpenAI implementation
│   │   ├── anthropic_provider.py # Anthropic implementation
│   │   ├── gemini_provider.py  # Google Gemini implementation
│   │   ├── ollama_provider.py  # Ollama local models
│   │   └── stubs.py            # Local/test providers
│   ├── embeddings/             # Embedding providers (v0.8.0)
│   │   ├── provider.py         # EmbeddingProvider interface
│   │   ├── openai.py           # OpenAI embeddings
│   │   ├── anthropic.py        # Anthropic/Voyage embeddings
│   │   ├── gemini.py           # Google Gemini embeddings
│   │   └── cohere.py           # Cohere embeddings
│   ├── rag/                    # RAG components (v0.8.0)
│   │   ├── __init__.py         # RAGAgent high-level API
│   │   ├── vector_store.py     # VectorStore interface
│   │   ├── chunking.py         # Text splitting strategies
│   │   ├── loaders.py          # Document loaders
│   │   ├── tools.py            # RAGTool & SemanticSearchTool
│   │   └── stores/             # Vector store implementations
│   │       ├── memory.py       # In-memory (NumPy)
│   │       ├── sqlite.py       # SQLite persistent storage
│   │       ├── chroma.py       # ChromaDB integration
│   │       └── pinecone.py     # Pinecone cloud storage
│   └── toolbox/                # Pre-built tools (v0.5.1)
│       ├── file.py             # File operations
│       ├── web.py              # Web scraping & HTTP
│       ├── data.py             # Data processing
│       ├── datetime_tools.py   # Date/time utilities
│       └── text.py             # Text manipulation
├── tests/                      # Test suite (463 tests)
│   ├── conftest.py             # Pytest configuration
│   ├── test_framework.py       # Core framework tests
│   ├── test_better_errors.py   # Error handling tests
│   ├── test_cost_tracking.py   # Cost tracking tests
│   ├── test_e2e_providers.py   # End-to-end provider tests
│   ├── test_models.py          # Model registry tests
│   ├── test_embedding_providers.py  # Embedding provider tests
│   ├── test_vector_stores_crud.py   # Vector store CRUD tests
│   ├── test_document_loaders.py     # Document loader tests
│   ├── test_text_chunking.py       # Chunking strategy tests
│   ├── test_rag_workflow.py        # RAG integration tests
│   └── test_sqlite_integration.py  # SQLite persistence tests
├── examples/                   # Usage examples
│   ├── search_weather.py       # Basic tool usage
│   ├── async_agent_demo.py     # Async operations
│   ├── conversation_memory_demo.py
│   ├── cost_tracking_demo.py
│   ├── customer_support_bot.py
│   ├── data_analysis_agent.py
│   ├── rag_basic_demo.py       # Basic RAG with in-memory store
│   ├── rag_advanced_demo.py    # Advanced RAG with PDF + SQLite
│   └── semantic_search_demo.py # Multi-provider semantic search
└── scripts/                    # Development scripts
    ├── README.md               # Release documentation
    ├── release.py              # Python release script
    ├── release.sh              # Bash release script
    ├── smoke_cli.py            # Provider smoke tests
    └── test_memory_with_openai.py
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
python tests/test_framework.py
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

### 🗄️ **RAG & Vector Stores (New in v0.8.0!)**
- Add new vector store integrations (Weaviate, Qdrant, Milvus)
- Implement reranking support (Cohere, Jina)
- Add hybrid search (vector + BM25 keyword)
- Improve chunking strategies (contextual, agentic)
- Create advanced RAG examples and tutorials

### 📚 **Documentation**
- Improve README examples
- Add tutorials and guides (especially for RAG features!)
- Fix typos and clarify confusing sections
- Add comparison guides (vs LangChain, LlamaIndex)

### 🧪 **Testing**
- Increase test coverage (currently 463 tests passing!)
- Add integration tests
- Add performance benchmarks
- Test RAG workflows with different vector stores

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
# tests/test_framework.py

def test_your_provider():
    # Add test cases
    pass
```

3. **Update documentation**
   - Add to README provider list
   - Add usage example
   - Update CHANGELOG

## Adding a New Tool

To contribute a new pre-built tool:

1. **Create the tool**

```python
# src/selectools/tools/your_tool.py

from ..tools import Tool, ToolParameter

def your_tool_implementation(param1: str, param2: int = 10) -> str:
    """Implementation of your tool."""
    # Your logic here
    return result

def create_your_tool() -> Tool:
    """Factory function to create the tool."""
    return Tool(
        name="your_tool",
        description="Clear description of what the tool does",
        parameters=[
            ToolParameter(name="param1", param_type=str, description="Description", required=True),
            ToolParameter(name="param2", param_type=int, description="Description", required=False),
        ],
        function=your_tool_implementation,
    )
```

2. **Add tests and examples**

3. **Update documentation**

## Adding RAG Features (New in v0.8.0!)

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

By contributing to Selectools, you agree that your contributions will be licensed under the LGPL-3.0-or-later license.

---

Thank you for contributing to Selectools! 🎉
