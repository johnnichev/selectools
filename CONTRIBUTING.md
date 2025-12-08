# Contributing to Selectools

Thank you for your interest in contributing to Selectools! We welcome contributions from the community.

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
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"  # Optional
export GEMINI_API_KEY="your-key-here"     # Optional
```

## Running Tests

Run the test suite to ensure everything works:

```bash
python tests/test_framework.py
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
├── src/selectools/          # Main package
│   ├── agent.py            # Agent loop and orchestration
│   ├── parser.py           # TOOL_CALL parser
│   ├── prompt.py           # Prompt builder
│   ├── tools.py            # Tool definitions and registry
│   ├── types.py            # Core types (Message, Role, etc.)
│   ├── cli.py              # CLI interface
│   ├── providers/          # LLM provider adapters
│   │   ├── base.py         # Provider interface
│   │   ├── openai_provider.py
│   │   └── stubs.py        # Anthropic, Gemini, Local
│   └── examples/           # Example tools
│       └── bbox.py         # Bounding box detection
├── tests/                  # Test suite
├── examples/               # Usage examples
└── scripts/                # Development scripts
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

### 🛠️ **New Tools**
- Pre-built tools for common use cases
- Integration with popular APIs and services
- Example tools demonstrating best practices

### 📚 **Documentation**
- Improve README examples
- Add tutorials and guides
- Fix typos and clarify confusing sections

### 🧪 **Testing**
- Increase test coverage
- Add integration tests
- Add performance benchmarks

### 🐛 **Bug Fixes**
- Fix reported issues
- Improve error messages
- Handle edge cases

## Adding a New Provider

To add support for a new LLM provider:

1. **Create a new provider file**

```python
# src/selectools/providers/your_provider.py

from typing import Iterator
from .base import Provider, ProviderError
from ..types import Message

class YourProvider(Provider):
    def __init__(self, api_key: str = None, default_model: str = "model-name"):
        self.api_key = api_key or os.getenv("YOUR_PROVIDER_API_KEY")
        self.default_model = default_model

    def complete(self, messages: list[Message], model: str = None, **kwargs) -> str:
        # Implementation here
        pass

    def stream(self, messages: list[Message], model: str = None, **kwargs) -> Iterator[str]:
        # Implementation here
        pass
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
