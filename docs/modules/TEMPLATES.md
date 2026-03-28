# Templates Module

**Added in:** v0.19.0
**Package:** `src/selectools/templates/`
**Functions:** `from_yaml()`, `from_dict()`, `load_template()`, `list_templates()`

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [from_yaml()](#from_yaml)
4. [from_dict()](#from_dict)
5. [load_template()](#load_template)
6. [Built-in Templates](#built-in-templates)
7. [YAML Config Reference](#yaml-config-reference)
8. [Tool Resolution](#tool-resolution)
9. [Custom Templates](#custom-templates)
10. [Integration with Serve](#integration-with-serve)
11. [API Reference](#api-reference)
12. [Examples](#examples)

---

## Overview

The **templates** module provides two ways to create agents without writing Python:

1. **YAML configuration files** -- define an agent's model, tools, system prompt, and behavior in a YAML file. Load it with `from_yaml()`.
2. **Pre-built templates** -- 5 ready-to-use agent configurations for common use cases. Load them with `load_template()`.

### Why Templates?

| | Python Code | YAML Config | Built-in Template |
|---|---|---|---|
| **Lines** | 10-30 | 5-15 | 1 |
| **Requires Python?** | Yes | No (CLI: `selectools serve agent.yaml`) | No (CLI: `selectools serve customer_support`) |
| **Customizable?** | Full control | Full control | Overrides only |
| **Best for** | Production apps | Config-driven deployments | Demos, prototyping |

### Design Philosophy

- **No magic.** YAML keys map 1:1 to `AgentConfig` fields. If you know the Python API, you know the YAML format.
- **Batteries included.** Five templates cover the most common agent patterns. Each includes purpose-built tools and a tuned system prompt.
- **Composable with serve.** Both YAML configs and template names work directly with `selectools serve`.

---

## Quick Start

### From YAML

```python
from selectools.templates import from_yaml

agent = from_yaml("agent.yaml")
result = agent.run("Hello!")
print(result.content)
```

### From Template

```python
from selectools.templates import load_template
from selectools.providers.openai_provider import OpenAIProvider

agent = load_template("customer_support", provider=OpenAIProvider())
result = agent.run("I can't log into my account")
print(result.content)
```

### From CLI

```bash
# YAML config
selectools serve agent.yaml

# Built-in template (auto-detects API key)
selectools serve research_assistant
```

---

## from_yaml()

Create an `Agent` from a YAML configuration file.

```python
from selectools.templates import from_yaml

# Basic usage -- provider auto-detected from YAML "provider" field
agent = from_yaml("agent.yaml")

# Override provider
from selectools.providers.anthropic_provider import AnthropicProvider
agent = from_yaml("agent.yaml", provider=AnthropicProvider())
```

### Example YAML

```yaml
provider: openai
model: gpt-4o
system_prompt: "You are a helpful coding assistant."
temperature: 0.7
max_iterations: 5

tools:
  - selectools.toolbox.file_tools.read_file
  - selectools.toolbox.file_tools.write_file
  - ./my_custom_tool.py

retry:
  max_retries: 3

budget:
  max_cost_usd: 0.50
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | (required) | Path to the YAML config file. |
| `provider` | `Optional[Provider]` | `None` | Override the provider. If `None`, created from the `provider` field in YAML. |

### Requirements

Requires PyYAML: `pip install pyyaml`. Raises `ImportError` with instructions if not installed.

---

## from_dict()

Create an `Agent` from a Python dictionary. Same format as the YAML config but as a dict -- useful when configs come from a database, API, or environment variables.

```python
from selectools.templates import from_dict

config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "system_prompt": "You are a helpful assistant.",
    "tools": ["selectools.toolbox.file_tools.read_file"],
    "budget": {"max_cost_usd": 0.25},
}

agent = from_dict(config)
result = agent.run("Read the README")
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `Dict[str, Any]` | (required) | Configuration dictionary. |
| `provider` | `Optional[Provider]` | `None` | Override the provider. |

---

## load_template()

Load a pre-built agent template by name. Each template includes purpose-built tools and a tuned system prompt for its use case.

```python
from selectools.templates import load_template
from selectools.providers.openai_provider import OpenAIProvider

provider = OpenAIProvider()

# Load with defaults
agent = load_template("customer_support", provider=provider)

# Override specific config fields
agent = load_template(
    "research_assistant",
    provider=provider,
    model="gpt-4o",           # override default model
    max_iterations=12,         # override default iteration limit
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Template name. See [Built-in Templates](#built-in-templates). |
| `provider` | `Provider` | (required) | LLM provider instance. |
| `**overrides` | `Any` | -- | Override any `AgentConfig` field. |

### Listing Available Templates

```python
from selectools.templates import list_templates

print(list_templates())
# ['code_reviewer', 'customer_support', 'data_analyst', 'rag_chatbot', 'research_assistant']
```

---

## Built-in Templates

### customer_support

A friendly customer support agent with account lookup, knowledge base search, and ticket escalation.

**Tools:** `lookup_customer`, `search_help_articles`, `create_ticket`
**Default model:** `gpt-4o-mini`
**Max iterations:** 5

```python
agent = load_template("customer_support", provider=provider)
result = agent.run("I can't reset my password")
```

The agent will look up the customer's account, search help articles for password reset instructions, and only create a support ticket if it cannot resolve the issue directly.

### research_assistant

A thorough research agent that searches the web, reads sources, and organizes findings with citations.

**Tools:** `web_search`, `read_url`, `save_notes`
**Default model:** `gpt-4o-mini`
**Max iterations:** 8

```python
agent = load_template("research_assistant", provider=provider)
result = agent.run("What are the latest advances in quantum computing?")
```

The agent searches broadly first, dives into relevant sources, and distinguishes facts from opinions.

### data_analyst

An agent for data exploration, SQL queries, and visualization.

**Tools:** Data querying and analysis tools
**Default model:** `gpt-4o-mini`

```python
agent = load_template("data_analyst", provider=provider)
result = agent.run("Show me monthly revenue trends for Q4")
```

### code_reviewer

An agent that reviews code for bugs, style issues, and security vulnerabilities.

**Tools:** Code analysis and review tools
**Default model:** `gpt-4o-mini`

```python
agent = load_template("code_reviewer", provider=provider)
result = agent.run("Review this pull request for security issues")
```

### rag_chatbot

A retrieval-augmented chatbot that searches a knowledge base before answering.

**Tools:** RAG search and retrieval tools
**Default model:** `gpt-4o-mini`

```python
agent = load_template("rag_chatbot", provider=provider)
result = agent.run("How do I configure SSL certificates?")
```

---

## YAML Config Reference

Every field in `AgentConfig` is configurable via YAML. Fields map directly -- no translation layer.

### Top-Level Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | `"openai"` | Provider name: `openai`, `anthropic`, `gemini`, `ollama`, `local`. |
| `model` | `str` | Provider default | Model identifier (e.g. `gpt-4o`, `claude-sonnet-4-20250514`). |
| `temperature` | `float` | Provider default | Sampling temperature. |
| `max_tokens` | `int` | Provider default | Maximum response tokens. |
| `max_iterations` | `int` | `10` | Maximum agent loop iterations. |
| `system_prompt` | `str` | `""` | System prompt for the agent. |
| `verbose` | `bool` | `False` | Enable verbose logging. |
| `stream` | `bool` | `False` | Enable streaming by default. |
| `reasoning_strategy` | `str` | `None` | Reasoning strategy: `react`, `cot`. |

### tools

A list of tool specifications. Each entry can be:

- **Dotted import path:** `selectools.toolbox.file_tools.read_file`
- **Relative file path:** `./my_custom_tool.py` (resolved relative to the YAML file)

```yaml
tools:
  - selectools.toolbox.file_tools.read_file
  - selectools.toolbox.file_tools.write_file
  - selectools.toolbox.web_tools.web_search
  - ./custom_tools/my_tool.py
```

### retry

Retry configuration for LLM API calls.

```yaml
retry:
  max_retries: 3
```

### budget

Token and cost budget limits.

```yaml
budget:
  max_cost_usd: 1.00
  max_tokens: 50000
```

### coherence

Coherence checking configuration.

```yaml
coherence:
  enabled: true
```

### compress

Prompt compression configuration.

```yaml
compress:
  enabled: true
  threshold: 10000
```

### trace

Trace configuration.

```yaml
trace:
  enabled: true
```

### Full Example

```yaml
provider: openai
model: gpt-4o
temperature: 0.3
max_iterations: 8
system_prompt: |
  You are a senior software engineer reviewing code.
  Focus on security, performance, and maintainability.
  Always explain your reasoning.

tools:
  - selectools.toolbox.file_tools.read_file
  - selectools.toolbox.file_tools.list_dir
  - ./project_tools/run_tests.py

retry:
  max_retries: 2

budget:
  max_cost_usd: 2.00

coherence:
  enabled: true

compress:
  enabled: true
  threshold: 8000
```

---

## Tool Resolution

Tools specified in YAML or dicts are resolved at load time through two mechanisms:

### Dotted Import Paths

Reference any tool by its full Python import path. The module is imported and the tool object is extracted.

```yaml
tools:
  - selectools.toolbox.file_tools.read_file     # Built-in tool
  - mypackage.tools.custom_search               # Your own tool
```

### Relative File Paths

Reference a Python file containing `@tool`-decorated functions. The file is loaded via `ToolLoader.from_file()` and all tools discovered in it are registered.

```yaml
tools:
  - ./my_tool.py          # Relative to YAML file location
  - ../shared/utils.py    # Parent directory
```

File paths are resolved relative to the YAML config file's directory. Path traversal outside the config directory is rejected for security.

### No Tools

If no tools are specified, a no-op placeholder tool is registered so the agent can still function (some providers require at least one tool).

---

## Custom Templates

Create your own reusable templates by following the built-in pattern:

### Template Module Structure

```python
# my_templates/sales_agent.py

from selectools.agent.config import AgentConfig
from selectools.agent.core import Agent
from selectools.tools.decorators import tool


@tool(description="Look up product details by SKU or name")
def product_lookup(query: str) -> str:
    """Search the product catalog."""
    return f"Product info for '{query}': ..."


@tool(description="Check inventory levels for a product")
def check_inventory(sku: str) -> str:
    """Check current stock levels."""
    return f"SKU {sku}: 142 units in stock"


SYSTEM_PROMPT = """You are a knowledgeable sales assistant.
Help customers find the right products and check availability."""


def build(provider, **overrides):
    """Build a sales agent."""
    config_kwargs = {
        "model": overrides.pop("model", "gpt-4o-mini"),
        "max_iterations": overrides.pop("max_iterations", 5),
        "system_prompt": overrides.pop("system_prompt", SYSTEM_PROMPT),
        **overrides,
    }
    return Agent(
        provider=provider,
        tools=[product_lookup, check_inventory],
        config=AgentConfig(**config_kwargs),
    )
```

### Using Custom Templates

```python
from my_templates.sales_agent import build

agent = build(provider=provider, model="gpt-4o")
```

The key convention is a `build(provider, **overrides)` function that returns a configured `Agent`.

---

## Integration with Serve

Templates and YAML configs work directly with the serve module:

```bash
# Serve from YAML
selectools serve agent.yaml

# Serve a built-in template by name
selectools serve customer_support
selectools serve research_assistant
selectools serve data_analyst
selectools serve code_reviewer
selectools serve rag_chatbot
```

When serving a template by name, the CLI auto-detects an available API key from environment variables (checking `OPENAI_API_KEY`, then `ANTHROPIC_API_KEY`, then `GOOGLE_API_KEY` in order).

---

## API Reference

### from_yaml()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | (required) | Path to YAML config file. |
| `provider` | `Optional[Provider]` | `None` | Override provider. Auto-resolved from YAML if `None`. |

Returns: `Agent`

### from_dict()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `Dict[str, Any]` | (required) | Configuration dictionary. Same schema as YAML. |
| `provider` | `Optional[Provider]` | `None` | Override provider. Auto-resolved from dict if `None`. |

Returns: `Agent`

### load_template()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | (required) | Template name. |
| `provider` | `Provider` | (required) | LLM provider instance. |
| `**overrides` | `Any` | -- | Override any `AgentConfig` field. |

Returns: `Agent`

### list_templates()

Returns: `List[str]` -- sorted list of available template names.

### Supported Providers (YAML)

| Name | Class | Required Env Var |
|---|---|---|
| `openai` | `OpenAIProvider` | `OPENAI_API_KEY` |
| `anthropic` | `AnthropicProvider` | `ANTHROPIC_API_KEY` |
| `gemini` | `GeminiProvider` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `ollama` | `OllamaProvider` | None (local) |
| `local` | `LocalProvider` | None (stub) |

---

## Examples

| Example | File | Description |
|---|---|---|
| 64 | [`64_yaml_config.py`](https://github.com/johnnichev/selectools/blob/main/examples/64_yaml_config.py) | Load an agent from YAML config |
| 65 | [`65_templates.py`](https://github.com/johnnichev/selectools/blob/main/examples/65_templates.py) | Use all 5 built-in templates |

---

## Further Reading

- [Serve Module](SERVE.md) -- Deploy agents as HTTP APIs
- [Agent Module](AGENT.md) -- The Agent class and AgentConfig
- [Tools Module](TOOLS.md) -- Custom tool creation with `@tool`
- [Dynamic Tools](DYNAMIC_TOOLS.md) -- ToolLoader for runtime tool discovery

---

**Next Steps:** Learn about deploying agents as HTTP APIs in the [Serve Module](SERVE.md).
