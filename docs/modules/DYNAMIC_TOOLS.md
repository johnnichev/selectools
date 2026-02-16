# Dynamic Tools Module

**File:** `src/selectools/tools/loader.py`, `src/selectools/agent/core.py`
**Classes:** `ToolLoader` (loader), `Agent` (dynamic tool methods)
**Imports:** `ToolLoader`, `Tool`, `tool` from `selectools.tools`; `Agent`, `AgentConfig` from `selectools`

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [ToolLoader](#toolloader)
5. [Agent Dynamic Methods](#agent-dynamic-methods)
6. [Plugin System Pattern](#plugin-system-pattern)
7. [Hot-Reload Pattern](#hot-reload-pattern)
8. [Integration with ToolRegistry](#integration-with-toolregistry)
9. [Error Handling](#error-handling)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)
12. [Further Reading](#further-reading)

---

## Overview

Dynamic tool loading enables agents to discover, load, and manage tools at runtime—without restarting the application. This supports:

- **Plugin Systems**: Load tools from third-party or user-provided modules
- **Hot-Reload**: Update tool implementations during development without restarting
- **A/B Testing**: Swap tool sets dynamically to compare behavior
- **Conditional Tool Loading**: Load tools based on environment, permissions, or feature flags

### Why It Matters

| Use Case | Without Dynamic Loading | With Dynamic Loading |
| --- | --- | --- |
| New plugin | Restart app, redeploy | Load module, call `agent.add_tools()` |
| Fix tool bug | Restart app | Call `ToolLoader.reload_file()`, `agent.replace_tool()` |
| Experiment | Deploy different builds | Swap tools at runtime via `replace_tool()` |

### Core Components

```python
ToolLoader    # Discover and load @tool-decorated functions from modules/files/dirs
Agent         # add_tool, add_tools, remove_tool, replace_tool — all rebuild system prompt
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Plugin Directory / File / Module                      │
│                                                                              │
│   plugins/                    my_tools.py              myapp.tools.search     │
│   ├── search.py              @tool def search(...)    @tool def search(...)  │
│   ├── weather.py             @tool def weather(...)                         │
│   └── _internal.py           (skipped)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ ToolLoader.from_directory()
                                      │ ToolLoader.from_file()
                                      │ ToolLoader.from_module()
                                      │ ToolLoader.reload_file()
                                      │ ToolLoader.reload_module()
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ToolLoader                                       │
│                                                                              │
│   Returns: List[Tool]                                                         │
│   - Imports as _selectools_dynamic_.<name> (files)                           │
│   - Skips _*.py by default                                                   │
│   - Collects module-level Tool instances (@tool-decorated functions)        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ agent.add_tools(tools)
                                      │ agent.add_tool(tool)
                                      │ agent.replace_tool(tool)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Agent                                            │
│                                                                              │
│   - Updates self.tools, self._tools_by_name                                  │
│   - Rebuilds system prompt: self._system_prompt = prompt_builder.build()     │
│   - Next LLM call sees new tool schemas immediately                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLM Sees New Tools                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### File-Based Plugin Loading

```python
from selectools.tools import ToolLoader, Tool, tool
from selectools import Agent, AgentConfig
from selectools.providers.openai_provider import OpenAIProvider

# Define a minimal tool in a file (or use existing plugin)
# plugins/greeting.py:
#   from selectools.tools import tool
#   @tool(description="Greet a user by name")
#   def greet(name: str) -> str:
#       return f"Hello, {name}!"

# Load tools from plugin directory
tools = ToolLoader.from_directory("./plugins/")
agent = Agent(
    tools=tools,
    provider=OpenAIProvider(),
    config=AgentConfig(model="gpt-4o-mini"),
)

response = agent.run([Message(role=Role.USER, content="Greet Alice")])
```

### Module-Based Loading

```python
tools = ToolLoader.from_module("myapp.tools.search")
agent = Agent(tools=tools, provider=provider)
```

### Add Tool at Runtime

```python
@tool(description="Get current time")
def get_time() -> str:
    from datetime import datetime
    return datetime.now().isoformat()

agent.add_tool(get_time)
# System prompt rebuilt — LLM can call get_time immediately
```

---

## ToolLoader

`ToolLoader` discovers and loads `Tool` instances (functions decorated with `@tool`) from Python modules, files, or directories.

### Import

```python
from selectools.tools import ToolLoader, Tool, tool
```

### Methods

#### `ToolLoader.from_module(module_path: str) -> List[Tool]`

Import a dotted module path and return all `Tool` objects found.

| Argument | Type | Description |
| --- | --- | --- |
| `module_path` | `str` | Dotted Python module path, e.g. `"myapp.tools"` |

**Returns:** List of `Tool` instances discovered in the module.

**Raises:** `ImportError` if the module cannot be imported.

```python
tools = ToolLoader.from_module("myproject.tools.search")
```

---

#### `ToolLoader.from_file(file_path: str) -> List[Tool]`

Load a single `.py` file and return all `Tool` objects found. The file is imported as a module under the `_selectools_dynamic_` namespace.

| Argument | Type | Description |
| --- | --- | --- |
| `file_path` | `str` | Absolute or relative path to a `.py` file |

**Returns:** List of `Tool` instances discovered in the file.

**Raises:**
- `FileNotFoundError` if the file does not exist
- `ValueError` if the path is not a `.py` file
- `ImportError` if the file cannot be imported

```python
tools = ToolLoader.from_file("/abs/path/to/search_tools.py")
# Module registered as: _selectools_dynamic_.search_tools
```

---

#### `ToolLoader.from_directory(directory, *, recursive=False, exclude=None) -> List[Tool]`

Discover and load `Tool` objects from all `.py` files in a directory.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `directory` | `str` | — | Path to the directory to scan |
| `recursive` | `bool` | `False` | If `True`, also scan subdirectories |
| `exclude` | `Sequence[str]` | `None` | Optional sequence of filenames to skip |

**Behavior:**
- Files whose names start with `_` are **skipped by default**
- Uses `**/*.py` when `recursive=True`, else `*.py`
- Skips files in `exclude`
- On per-file import error: skips file and continues (no exception raised)

**Returns:** List of `Tool` instances discovered across all loaded files.

**Raises:** `FileNotFoundError` if the directory does not exist.

```python
tools = ToolLoader.from_directory("./plugins/")
tools = ToolLoader.from_directory("./plugins/", recursive=True)
tools = ToolLoader.from_directory("./plugins/", exclude=["deprecated.py", "test_tools.py"])
```

---

#### `ToolLoader.reload_module(module_path: str) -> List[Tool]`

Re-import a module and return freshly loaded `Tool` objects. Useful for hot-reloading tools without restarting.

| Argument | Type | Description |
| --- | --- | --- |
| `module_path` | `str` | Dotted Python module path to reload |

**Returns:** List of `Tool` instances from the reloaded module.

**Raises:** `ImportError` if the module cannot be reloaded.

**Behavior:** If the module is not in `sys.modules`, falls back to `from_module()`.

```python
tools = ToolLoader.reload_module("myapp.tools.search")
```

---

#### `ToolLoader.reload_file(file_path: str) -> List[Tool]`

Re-import a Python file and return freshly loaded `Tool` objects. Useful for hot-reloading plugin files after edits.

| Argument | Type | Description |
| --- | --- | --- |
| `file_path` | `str` | Path to the `.py` file to reload |

**Returns:** List of `Tool` instances from the reloaded file.

**Behavior:** Removes the module from `sys.modules` (if present) so the next import is fresh.

```python
tools = ToolLoader.reload_file("/path/to/plugins/search.py")
```

---

### Namespace Convention

Files loaded via `from_file()` or `from_directory()` are imported as:

```
_selectools_dynamic_.<filename_stem>
```

Example: `/path/to/search_tools.py` → `_selectools_dynamic_.search_tools`

This avoids conflicts with application module names.

---

## Agent Dynamic Methods

All dynamic tool methods **rebuild the system prompt** so the LLM immediately sees updated tool schemas on the next call.

### `agent.add_tool(tool: Tool) -> None`

Add a single tool at runtime.

| Argument | Type | Description |
| --- | --- | --- |
| `tool` | `Tool` | Tool instance to add |

**Raises:** `ValueError` if a tool with the same name already exists (suggests `replace_tool()`).

```python
agent.add_tool(new_tool)
```

---

### `agent.add_tools(tools: List[Tool]) -> None`

Add multiple tools at runtime in a batch.

| Argument | Type | Description |
| --- | --- | --- |
| `tools` | `List[Tool]` | List of Tool instances to add |

**Raises:** `ValueError` if any tool name already exists.

```python
agent.add_tools([tool_a, tool_b, tool_c])
```

---

### `agent.remove_tool(tool_name: str) -> Tool`

Remove a tool by name.

| Argument | Type | Description |
| --- | --- | --- |
| `tool_name` | `str` | Name of the tool to remove |

**Returns:** The removed `Tool` instance.

**Raises:**
- `KeyError` if no tool with that name exists
- `ValueError` if removing would leave the agent with zero tools (agent requires at least one tool)

```python
removed = agent.remove_tool("deprecated_search")
```

---

### `agent.replace_tool(tool: Tool) -> Optional[Tool]`

Replace an existing tool with an updated version, or add the tool if no tool with that name exists.

| Argument | Type | Description |
| --- | --- | --- |
| `tool` | `Tool` | The new Tool instance |

**Returns:** The old `Tool` instance that was replaced, or `None` if the tool was newly added.

```python
old_tool = agent.replace_tool(updated_search_tool)
# old_tool is the previous tool, or None if it was new
```

---

### System Prompt Rebuild

Every dynamic method calls:

```python
self._system_prompt = self.prompt_builder.build(self.tools)
```

So the next `provider.complete()` or `provider.acomplete()` call uses the updated tool schemas.

---

## Plugin System Pattern

### Directory-Based Plugin Architecture

```text
myapp/
├── main.py
└── plugins/
    ├── search.py      # @tool def search(...)
    ├── weather.py     # @tool def weather(...)
    ├── calculator.py # @tool def add(...), multiply(...)
    └── _internal.py   # Skipped (starts with _)
```

**main.py:**

```python
from pathlib import Path
from selectools.tools import ToolLoader, tool
from selectools import Agent, AgentConfig
from selectools.providers.openai_provider import OpenAIProvider
from selectools.types import Message, Role

plugins_dir = Path(__file__).parent / "plugins"
tools = ToolLoader.from_directory(str(plugins_dir))

agent = Agent(
    tools=tools,
    provider=OpenAIProvider(),
    config=AgentConfig(model="gpt-4o-mini"),
)

response = agent.run([Message(role=Role.USER, content="Search for Python and add 2+3")])
```

---

## Hot-Reload Pattern

Watch a file for changes, reload it, and replace tools in the agent.

```python
import time
from pathlib import Path
from selectools.tools import ToolLoader, tool
from selectools import Agent, AgentConfig

def watch_and_reload(agent: Agent, file_path: str, tool_names: list[str]) -> None:
    """Reload file and replace tools in agent when file changes."""
    path = Path(file_path)
    last_mtime = 0.0

    while True:
        try:
            mtime = path.stat().st_mtime
            if mtime > last_mtime:
                last_mtime = mtime
                tools = ToolLoader.reload_file(str(path))
                for t in tools:
                    if t.name in tool_names:
                        agent.replace_tool(t)
                        print(f"Reloaded tool: {t.name}")
        except Exception as e:
            print(f"Reload error: {e}")
        time.sleep(1.0)
```

**Usage:**

```python
# In development: run watch_and_reload in a background thread
import threading
watch_thread = threading.Thread(
    target=watch_and_reload,
    args=(agent, "./plugins/search.py", ["search"]),
    daemon=True,
)
watch_thread.start()
```

---

## Integration with ToolRegistry

`ToolLoader` and `ToolRegistry` serve different roles:

| Feature | ToolLoader | ToolRegistry |
| --- | --- | --- |
| Purpose | Load tools from modules/files/dirs | Organize tools defined in code |
| Discovery | File system, import path | Decorator `@registry.tool()` |
| Hot-reload | Yes (`reload_file`, `reload_module`) | No (registry is static) |

### Combined Pattern

Load from plugins, then register in a registry for filtering:

```python
from selectools.tools import ToolLoader, ToolRegistry, tool

# Load from plugin directory
plugin_tools = ToolLoader.from_directory("./plugins/")

# Also use registry for in-code tools
registry = ToolRegistry()
@registry.tool(description="Echo")
def echo(text: str) -> str:
    return text

# Merge and pass to agent
all_tools = plugin_tools + registry.all()
agent = Agent(tools=all_tools, provider=provider)
```

### Using ToolLoader with Registry.all()

```python
registry = ToolRegistry()
# ... register tools ...

# Add plugins on top of registry
plugin_tools = ToolLoader.from_directory("./plugins/")
agent = Agent(
    tools=registry.all() + plugin_tools,
    provider=provider,
)
```

---

## Error Handling

### Duplicate Tool Names

```python
agent.add_tool(search_tool)
agent.add_tool(search_tool)  # ValueError: Tool 'search' already exists. Use replace_tool() to update it.
```

**Fix:** Use `agent.replace_tool(search_tool)` to update.

---

### Missing Files

```python
ToolLoader.from_file("/nonexistent/path.py")  # FileNotFoundError: Tool file not found: ...
ToolLoader.from_directory("/nonexistent/")   # FileNotFoundError: Tool directory not found: ...
```

**Fix:** Ensure paths exist and are correct.

---

### Invalid Modules

```python
ToolLoader.from_module("not.installed.module")  # ImportError
ToolLoader.from_file("syntax_error.py")         # ImportError (or SyntaxError)
```

**Fix:** Install dependencies or fix module syntax.

---

### Removing Last Tool

```python
agent = Agent(tools=[only_tool], provider=provider)
agent.remove_tool("only_tool")  # ValueError: Agent requires at least one tool.
```

**Fix:** Add a replacement tool before removing, or use `replace_tool()`.

---

### Directory Import Failures

`from_directory()` catches per-file import errors and continues:

```python
# plugins/good.py loads OK
# plugins/bad.py raises ImportError → skipped, no exception propagated
tools = ToolLoader.from_directory("./plugins/")  # Returns tools from good.py only
```

---

## Best Practices

### 1. Organize Plugins by Domain

```text
plugins/
├── search/
│   ├── web_search.py
│   └── docs_search.py
├── data/
│   ├── db_query.py
│   └── csv_export.py
└── utils/
    └── calculator.py
```

Load with `recursive=True`:

```python
tools = ToolLoader.from_directory("./plugins/", recursive=True)
```

---

### 2. Use Private Files for Internals

Prefix with `_` to skip during discovery:

```text
plugins/
├── search.py      # Loaded
├── _helpers.py    # Skipped
└── _config.py     # Skipped
```

---

### 3. Use replace_tool for Hot-Reload

Avoid removing then adding; use `replace_tool()` to keep tool order and avoid edge cases:

```python
# Preferred
new_tools = ToolLoader.reload_file("./plugins/search.py")
for t in new_tools:
    agent.replace_tool(t)
```

---

### 4. Validate After Load

```python
tools = ToolLoader.from_directory("./plugins/")
assert len(tools) > 0, "No tools loaded from plugins"
# Optionally check for expected tool names
names = {t.name for t in tools}
assert "search" in names, "Missing expected 'search' tool"
```

---

### 5. Use exclude for Unwanted Files

```python
tools = ToolLoader.from_directory(
    "./plugins/",
    exclude=["legacy_tools.py", "experimental.py"]
)
```

---

## Troubleshooting

### Tools Not Appearing in Agent

**Symptom:** LLM doesn't seem to know about new tools.

**Causes:**
- Tools were added but system prompt wasn't rebuilt (ensure you use `add_tool`/`add_tools`/`replace_tool`, not direct list mutation)
- Cache: if using `AgentConfig(cache=...)`, cached responses won't reflect new tools until cache key changes

**Fix:** Dynamic methods already rebuild the prompt. If using cache, consider invalidating or using a cache that incorporates tool set in the key.

---

### Import Error When Loading File

**Symptom:** `ImportError` when calling `from_file()` or `from_directory()`.

**Causes:**
- Missing dependencies in the plugin file
- Syntax errors in the file
- Circular imports

**Fix:** Run the plugin module directly to reproduce the error:
```bash
python -c "import importlib.util; spec = importlib.util.spec_from_file_location('test', 'plugins/search.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)"
```

---

### Duplicate Tool Names Across Plugins

**Symptom:** `ValueError: Tool 'X' already exists` when calling `add_tools()`.

**Cause:** Multiple plugin files define tools with the same name.

**Fix:**
1. Rename tools to be unique (e.g. `web_search`, `docs_search`)
2. Load files separately and merge manually, resolving duplicates
3. Use `replace_tool()` if the latter definition should override

---

## Further Reading

- [Tools Module](TOOLS.md) — Tool definition, `@tool` decorator, `ToolRegistry`
- [Agent Module](AGENT.md) — Agent loop, configuration, hooks
- [Prompt Module](PROMPT.md) — How tool schemas are formatted in the system prompt

---

**Next Steps:** Define tools with the `@tool` decorator as described in the [Tools Module](TOOLS.md).
