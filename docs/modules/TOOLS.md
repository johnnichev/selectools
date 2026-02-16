# Tools Module

**File:** `src/selectools/tools.py`
**Classes:** `Tool`, `ToolParameter`, `ToolRegistry`
**Decorators:** `@tool`

## Table of Contents

1. [Overview](#overview)
2. [Tool Definition](#tool-definition)
3. [Schema Generation](#schema-generation)
4. [Parameter Validation](#parameter-validation)
5. [Tool Execution](#tool-execution)
6. [Decorator Pattern](#decorator-pattern)
7. [Tool Registry](#tool-registry)
8. [Streaming Tools](#streaming-tools)
9. [Injected Parameters](#injected-parameters)
10. [Implementation Details](#implementation-details)

---

## Overview

The **Tools** module provides the foundation for defining callable functions that AI agents can invoke. It handles:

- **Schema Generation**: Automatic JSON schema from Python type hints
- **Validation**: Runtime parameter checking with helpful errors
- **Execution**: Sync/async function calls with timeout support
- **Streaming**: Progressive results via Generator/AsyncGenerator
- **Injection**: Clean separation of LLM-visible and hidden parameters

### Core Classes

```python
ToolParameter   # Defines a single parameter
Tool            # Encapsulates a callable with metadata
ToolRegistry    # Organizes multiple tools
```

---

## Tool Definition

### Manual Definition

```python
from selectools import Tool, ToolParameter

def get_weather(location: str, units: str = "celsius") -> str:
    return f"Weather in {location}: 72°{units[0].upper()}"

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters=[
        ToolParameter(
            name="location",
            param_type=str,
            description="City name or coordinates",
            required=True
        ),
        ToolParameter(
            name="units",
            param_type=str,
            description="celsius or fahrenheit",
            required=False,
            enum=["celsius", "fahrenheit"]
        ),
    ],
    function=get_weather
)
```

### Using @tool Decorator (Recommended)

```python
from selectools import tool

@tool(
    name="get_weather",  # Optional: defaults to function name
    description="Get current weather for a location",
    param_metadata={
        "location": {"description": "City name or coordinates"},
        "units": {"description": "Temperature units", "enum": ["celsius", "fahrenheit"]}
    }
)
def get_weather(location: str, units: str = "celsius") -> str:
    return f"Weather in {location}: 72°{units[0].upper()}"
```

The decorator:

- Infers parameter names and types from function signature
- Detects required vs optional from default values
- Generates JSON schema automatically

---

## Schema Generation

### Type Mapping

Python types are mapped to JSON schema types:

```python
str    → "string"
int    → "integer"
float  → "number"
bool   → "boolean"
list   → "array"
dict   → "object"
```

### Generated Schema

For the `get_weather` tool:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      },
      "units": {
        "type": "string",
        "description": "Temperature units",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  }
}
```

### Schema Usage

The agent sends this schema to the LLM in the system prompt:

```
Available tools (JSON schema):

{
  "name": "get_weather",
  "description": "Get current weather for a location",
  ...
}
```

The LLM uses this to understand:

- What tools are available
- What parameters each tool needs
- What values are valid
- Which parameters are required

---

## Parameter Validation

### Validation Flow

```
LLM Response → Parser → Tool Call
                            │
                            ▼
                    ┌───────────────┐
                    │ Tool.validate()│
                    └───────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ Check for extra  │      │ Check for missing│
    │ parameters       │      │ required params  │
    └────────┬─────────┘      └────────┬─────────┘
             │                         │
             ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ Suggest typo     │      │ List required    │
    │ corrections      │      │ parameters       │
    └──────────────────┘      └──────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ Validate types   │      │ Check enum       │
    │ (str, int, etc.) │      │ constraints      │
    └────────┬─────────┘      └────────┬─────────┘
             │                         │
             └─────────────┬───────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Valid params?  │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         Yes  ▼                    No   ▼
    ┌──────────────┐       ┌─────────────────────┐
    │   Execute    │       │ Raise Validation    │
    │   Tool       │       │ Error               │
    └──────────────┘       └─────────────────────┘
```

### Implementation

```python
def validate(self, params: Dict[str, ParameterValue]) -> None:
    expected_params = {p.name for p in self.parameters}
    provided_params = set(params.keys())
    extra_params = provided_params - expected_params

    # 1. Check for unexpected parameters
    if extra_params:
        suggestions = []
        for extra in extra_params:
            # Use difflib to find close matches
            matches = difflib.get_close_matches(extra, expected_params, n=1, cutoff=0.6)
            if matches:
                suggestions.append(f"'{extra}' -> Did you mean '{matches[0]}'?")
            else:
                suggestions.append(f"'{extra}' is not a valid parameter")

        raise ToolValidationError(
            tool_name=self.name,
            param_name=", ".join(sorted(extra_params)),
            issue="Unexpected parameter(s)",
            suggestion="; ".join(suggestions)
        )

    # 2. Check for missing required parameters
    for param in self.parameters:
        if param.required and param.name not in params:
            expected_list = ", ".join(f"'{p.name}'" for p in self.parameters if p.required)
            raise ToolValidationError(
                tool_name=self.name,
                param_name=param.name,
                issue="Missing required parameter",
                suggestion=f"Required parameters: {expected_list}"
            )

        if param.name not in params:
            continue

        # 3. Validate parameter type
        error = self._validate_single(param, params[param.name])
        if error:
            # Provide type conversion suggestions
            value = params[param.name]
            type_hint = ""
            if param.param_type is str and not isinstance(value, str):
                type_hint = f"Try: {param.name}=str({repr(value)})"
            elif param.param_type is int and isinstance(value, str):
                type_hint = f"Try: {param.name}=int('{value}')"

            raise ToolValidationError(
                tool_name=self.name,
                param_name=param.name,
                issue=error,
                suggestion=type_hint if type_hint else f"Expected type: {param.param_type.__name__}"
            )
```

### Error Messages

Validation errors are designed to be helpful:

```
============================================================
❌ Tool Validation Error: 'get_weather'
============================================================

Parameter: loction
Issue: Unexpected parameter(s)

💡 Suggestion: 'loction' -> Did you mean 'location'?
Expected parameters: 'location', 'units'

============================================================
```

The LLM sees this error and can correct its mistake in the next iteration.

---

## Tool Execution

### Sync Execution

```python
def execute(self, params: Dict[str, ParameterValue], chunk_callback=None) -> str:
    # 1. Validate parameters
    self.validate(params)

    # 2. Prepare arguments
    call_args = dict(params)
    call_args.update(self.injected_kwargs)
    if self.config_injector:
        call_args.update(self.config_injector() or {})

    # 3. Execute function
    try:
        result = self.function(**call_args)

        # 4. Handle streaming (generators)
        if inspect.isgenerator(result):
            chunks = []
            for chunk in result:
                chunk_str = str(chunk)
                chunks.append(chunk_str)
                if chunk_callback:
                    chunk_callback(chunk_str)
            return "".join(chunks)

        return str(result)

    except Exception as exc:
        raise ToolExecutionError(
            tool_name=self.name,
            error=exc,
            params=params
        ) from exc
```

### Async Execution

```python
async def aexecute(self, params, chunk_callback=None) -> str:
    self.validate(params)

    call_args = dict(params)
    call_args.update(self.injected_kwargs)
    if self.config_injector:
        call_args.update(self.config_injector() or {})

    try:
        if self.is_async:
            # Async function or async generator
            result = self.function(**call_args)

            if inspect.isasyncgen(result):
                # Async generator
                chunks = []
                async for chunk in result:
                    chunk_str = str(chunk)
                    chunks.append(chunk_str)
                    if chunk_callback:
                        chunk_callback(chunk_str)
                return "".join(chunks)

            # Regular async function
            result = await result
            return str(result)
        else:
            # Sync function - run in executor
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    lambda: self.function(**call_args)
                )

            # Handle sync generator in async context
            if inspect.isgenerator(result):
                chunks = []
                for chunk in result:
                    chunk_str = str(chunk)
                    chunks.append(chunk_str)
                    if chunk_callback:
                        chunk_callback(chunk_str)
                return "".join(chunks)

            return str(result)

    except Exception as exc:
        raise ToolExecutionError(...)
```

### Detection of Async Tools

```python
def __init__(self, name, description, parameters, function, ...):
    # ...
    self.is_async = inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function)
```

---

## Decorator Pattern

### Basic Usage

```python
@tool(description="Add two numbers")
def add(a: int, b: int) -> str:
    return str(a + b)
```

This is equivalent to:

```python
def add(a: int, b: int) -> str:
    return str(a + b)

add = tool(description="Add two numbers")(add)
```

### Parameter Metadata

```python
@tool(
    description="Search the web",
    param_metadata={
        "query": {
            "description": "Search terms",
        },
        "limit": {
            "description": "Max results",
        }
    }
)
def search(query: str, limit: int = 10) -> str:
    return f"Found results for: {query}"
```

### Schema Inference

```python
def _infer_parameters_from_callable(func, param_metadata):
    signature = inspect.signature(func)
    parameters = []

    for name, param in signature.parameters.items():
        if name.startswith("_"):
            continue  # Skip private parameters

        # Get type annotation
        annotation = param.annotation if param.annotation is not inspect._empty else str

        # Get metadata
        meta = param_metadata.get(name, {})
        description = meta.get("description", "")
        enum = meta.get("enum")

        # Determine if required
        required = param.default is inspect._empty

        parameters.append(ToolParameter(
            name=name,
            param_type=annotation if isinstance(annotation, type) else str,
            description=description or f"Parameter '{name}'",
            required=required,
            enum=enum,
        ))

    return parameters
```

### Custom Names

```python
@tool(
    name="web_search",  # Override function name
    description="Search the web"
)
def search_google(query: str) -> str:
    return f"Results: {query}"

# Tool is accessible as "web_search", not "search_google"
```

### Docstring as Description

```python
@tool()
def calculate(a: int, b: int, operation: str = "add") -> str:
    """
    Perform arithmetic operations on two numbers.
    Supports add, subtract, multiply, divide.
    """
    # Implementation...
```

If `description` is not provided, the decorator uses the docstring.

---

## Tool Registry

### Purpose

`ToolRegistry` helps organize multiple tools:

```python
from selectools import ToolRegistry

registry = ToolRegistry()

@registry.tool(description="Add numbers")
def add(a: int, b: int) -> str:
    return str(a + b)

@registry.tool(description="Multiply numbers")
def multiply(a: int, b: int) -> str:
    return str(a * b)

@registry.tool(description="Search the web")
def search(query: str) -> str:
    return f"Results for: {query}"
```

### Using Registry with Agent

```python
from selectools import Agent, OpenAIProvider

# Get all registered tools
agent = Agent(
    tools=registry.all(),
    provider=OpenAIProvider()
)

# Or get specific tool
search_tool = registry.get("search")
```

### Benefits

1. **Organization**: Keep related tools together
2. **Discovery**: List all available tools
3. **Reusability**: Share tool sets across agents
4. **Modularity**: Define tools in separate modules

### Pattern

```python
# tools/math_tools.py
math_registry = ToolRegistry()

@math_registry.tool(description="Add")
def add(a: int, b: int) -> str:
    return str(a + b)

# tools/web_tools.py
web_registry = ToolRegistry()

@web_registry.tool(description="Search")
def search(query: str) -> str:
    return f"Results: {query}"

# main.py
from tools.math_tools import math_registry
from tools.web_tools import web_registry

all_tools = math_registry.all() + web_registry.all()
agent = Agent(tools=all_tools, provider=provider)
```

---

## Streaming Tools

### Generator-Based Streaming

```python
from typing import Generator

@tool(description="Process large file", streaming=True)
def process_file(filepath: str) -> Generator[str, None, None]:
    """Process file line by line."""
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            # Process line
            result = process_line(line)

            # Yield result chunk
            yield f"[Line {i}] {result}\n"
```

### Async Generator Streaming

```python
from typing import AsyncGenerator

@tool(description="Stream API responses", streaming=True)
async def stream_api(url: str) -> AsyncGenerator[str, None]:
    """Stream data from API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            async for line in resp.content:
                yield line.decode()
```

### Chunk Callbacks

The agent can register a callback to receive chunks:

```python
def on_chunk(tool_name: str, chunk: str):
    print(f"[{tool_name}] {chunk}", end='', flush=True)

config = AgentConfig(hooks={'on_tool_chunk': on_chunk})
agent = Agent(tools=[process_file], provider=provider, config=config)
```

### Execution Flow

```
Tool.execute() called
    │
    ├─→ Function returns Generator
    │
    ├─→ Iterate over generator
    │
    ├─→ For each chunk:
    │   ├─→ Convert to string
    │   ├─→ Append to accumulator
    │   └─→ Call chunk_callback(chunk)
    │
    └─→ Return accumulated string
```

### Use Cases

- **Large Files**: Process files too big for memory
- **Streaming APIs**: Real-time data from external services
- **Progress Updates**: Show progress for long operations
- **Partial Results**: Return results as they become available

---

## Injected Parameters

### Problem

Some parameters shouldn't be visible to the LLM:

- Database connections
- API keys
- Configuration objects
- Internal state

### Solution: Injected Kwargs

```python
import psycopg2

def query_database(sql: str, db_connection) -> str:
    """Execute SQL query. db_connection is injected."""
    with db_connection.cursor() as cursor:
        cursor.execute(sql)
        results = cursor.fetchall()
    return str(results)

# Create connection (not exposed to LLM)
db_conn = psycopg2.connect(
    host="localhost",
    database="myapp",
    user="readonly_user",
    password="secret"
)

# Tool only exposes 'sql' parameter
db_tool = Tool(
    name="query_db",
    description="Execute a read-only SQL query",
    parameters=[
        ToolParameter(name="sql", param_type=str, description="SQL SELECT query")
    ],
    function=query_database,
    injected_kwargs={"db_connection": db_conn}  # Injected at runtime
)
```

### LLM's View

The LLM only sees:

```json
{
  "name": "query_db",
  "description": "Execute a read-only SQL query",
  "parameters": {
    "type": "object",
    "properties": {
      "sql": { "type": "string", "description": "SQL SELECT query" }
    },
    "required": ["sql"]
  }
}
```

The `db_connection` parameter is completely hidden.

### Config Injector

For dynamic injection:

```python
def get_current_user():
    return {"user_id": 123, "role": "admin"}

@tool(
    description="Check user permissions",
    config_injector=get_current_user  # Called at execution time
)
def check_permissions(resource: str, user_id: int, role: str) -> str:
    return f"User {user_id} ({role}) access to {resource}: granted"
```

The `config_injector` is called during execution to get current values.

---

## Implementation Details

### Tool Validation at Registration

Tools are validated when created, not at runtime:

```python
def _validate_tool_definition(self) -> None:
    # Check for empty name
    if not self.name or not self.name.strip():
        raise ToolValidationError(...)

    # Check for empty description
    if not self.description or not self.description.strip():
        raise ToolValidationError(...)

    # Check for duplicate parameter names
    param_names = [p.name for p in self.parameters]
    duplicates = [name for name in param_names if param_names.count(name) > 1]
    if duplicates:
        raise ToolValidationError(...)

    # Validate parameter types
    supported_types = {str, int, float, bool, list, dict}
    for param in self.parameters:
        if param.param_type not in supported_types:
            raise ToolValidationError(...)

    # Validate function signature matches parameters
    try:
        sig = inspect.signature(self.function)
    except (ValueError, TypeError):
        return  # Can't inspect (built-in function)

    func_params = sig.parameters
    param_names_set = {p.name for p in self.parameters}
    injected_names = set(self.injected_kwargs.keys())

    # Check that all tool parameters exist in function
    for param in self.parameters:
        if param.name not in func_params and param.name not in injected_names:
            raise ToolValidationError(...)
```

This catches errors early, during development.

### ToolParameter Schema Conversion

```python
class ToolParameter:
    def to_schema(self) -> JsonSchema:
        schema = {
            "type": _python_type_to_json(self.param_type),
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema
```

### Tool Schema Generation

```python
class Tool:
    def schema(self) -> JsonSchema:
        properties = {param.name: param.to_schema() for param in self.parameters}
        required = [param.name for param in self.parameters if param.required]

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
```

---

## Best Practices

### 1. Use Type Hints

```python
# ✅ Good
@tool(description="Add numbers")
def add(a: int, b: int) -> str:
    return str(a + b)

# ❌ Bad
@tool(description="Add numbers")
def add(a, b):  # No type hints
    return str(a + b)
```

### 2. Provide Clear Descriptions

```python
# ✅ Good
@tool(description="Search for academic papers by keyword, author, or topic")
def search_papers(query: str) -> str:
    ...

# ❌ Bad
@tool(description="Search")
def search_papers(query: str) -> str:
    ...
```

### 3. Use Enums for Limited Options

```python
@tool(
    description="Convert temperature units",
    param_metadata={
        "units": {"enum": ["celsius", "fahrenheit", "kelvin"]}
    }
)
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    ...
```

### 4. Validate Input Early

```python
@tool(description="Divide two numbers")
def divide(a: float, b: float) -> str:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return str(a / b)
```

### 5. Return Strings

Tools must return strings (or yield strings for streaming):

```python
# ✅ Good
def get_count() -> str:
    return str(42)

# ❌ Bad
def get_count() -> int:
    return 42  # Agent expects string
```

### 6. Use Injected Kwargs for Secrets

```python
# ✅ Good
Tool(
    name="api_call",
    parameters=[ToolParameter(name="endpoint", ...)],
    function=call_api,
    injected_kwargs={"api_key": os.getenv("API_KEY")}
)

# ❌ Bad - exposes API key to LLM
Tool(
    name="api_call",
    parameters=[
        ToolParameter(name="endpoint", ...),
        ToolParameter(name="api_key", ...)  # Don't do this!
    ],
    function=call_api
)
```

---

## Testing

### Unit Testing Tools

```python
def test_add_tool():
    @tool(description="Add numbers")
    def add(a: int, b: int) -> str:
        return str(a + b)

    # Test execution
    result = add.execute({"a": 2, "b": 3})
    assert result == "5"

    # Test validation
    with pytest.raises(ToolValidationError):
        add.execute({"a": 2})  # Missing 'b'
```

### Testing with Agent

```python
def test_tool_with_agent():
    @tool(description="Echo")
    def echo(text: str) -> str:
        return text

    agent = Agent(
        tools=[echo],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=2, model="local")
    )

    response = agent.run([Message(role=Role.USER, content="Hello")])
    assert "Hello" in response.content
```

---

## Common Pitfalls

### 1. Type Mismatches

```python
# LLM might pass "42" as string, but function expects int
@tool(description="Calculate")
def calculate(a: int, b: int) -> str:
    return str(a + b)

# Fix: Validation catches this and suggests conversion
```

### 2. Missing Required Parameters

```python
# Function has required param, but LLM doesn't provide it
@tool(description="Greet user")
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Fix: Validation raises helpful error, LLM corrects on next iteration
```

### 3. Forgetting Return Type

```python
# ❌ Returns None implicitly
@tool(description="Log message")
def log_message(msg: str):
    print(msg)

# ✅ Return string
@tool(description="Log message")
def log_message(msg: str) -> str:
    print(msg)
    return f"Logged: {msg}"
```

---

## Further Reading

- [Agent Module](AGENT.md) - How agents use tools
- [Dynamic Tools Module](DYNAMIC_TOOLS.md) - ToolLoader, hot-reload, plugin systems
- [Parser Module](PARSER.md) - How tool calls are parsed
- [Prompt Module](PROMPT.md) - How tool schemas are formatted

---

**Next Steps:** Understand how the parser extracts tool calls in the [Parser Module](PARSER.md).
