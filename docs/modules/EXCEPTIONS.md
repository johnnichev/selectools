# Error Handling & Exceptions

All selectools exceptions inherit from `SelectoolsError`, so you can catch everything with a single handler or be specific.

---

## Exception Hierarchy

```
SelectoolsError                     # Base — catch-all for any selectools error
├── ToolValidationError             # Bad tool parameters (type mismatch, missing required)
├── ToolExecutionError              # Tool function raised an exception
├── ProviderConfigurationError      # Missing API key or bad provider setup
└── MemoryLimitExceededError        # Message count or token limit hit
```

All exceptions include **PyTorch-style error messages** with clear explanations and fix suggestions.

---

## Catching Errors

### Catch Everything

```python
from selectools import SelectoolsError

try:
    result = agent.ask("Do something")
except SelectoolsError as e:
    print(f"Selectools error: {e}")
```

### Specific Handlers

```python
from selectools import (
    SelectoolsError,
    ToolValidationError,
    ToolExecutionError,
    ProviderConfigurationError,
    MemoryLimitExceededError,
)

try:
    result = agent.ask("Process this data")
except ToolValidationError as e:
    print(f"Bad params for tool '{e.tool_name}': {e.issue}")
    print(f"Param: {e.param_name}")
    if e.suggestion:
        print(f"Fix: {e.suggestion}")
except ToolExecutionError as e:
    print(f"Tool '{e.tool_name}' crashed: {e.error}")
    print(f"Was called with: {e.params}")
except ProviderConfigurationError as e:
    print(f"Provider '{e.provider_name}' misconfigured: {e.missing_config}")
    if e.env_var:
        print(f"Set: export {e.env_var}='your-key'")
except MemoryLimitExceededError as e:
    print(f"Memory {e.limit_type} limit hit: {e.current}/{e.limit}")
except SelectoolsError as e:
    print(f"Other selectools error: {e}")
```

---

## ToolValidationError

**When:** The LLM provides parameters that don't match the tool's schema (wrong type, missing required field).

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `tool_name` | `str` | Name of the tool |
| `param_name` | `str` | Which parameter failed |
| `issue` | `str` | What went wrong |
| `suggestion` | `str` | How to fix it |

**Example output:**

```
============================================================
❌ Tool Validation Error: 'search'
============================================================

Parameter: limit
Issue: Expected int, got str

💡 Suggestion: Pass an integer value for 'limit'

============================================================
```

---

## ToolExecutionError

**When:** The tool function itself raises an exception during execution.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `tool_name` | `str` | Name of the tool |
| `error` | `Exception` | The original exception |
| `params` | `Dict[str, Any]` | Parameters the tool was called with |

**Example output:**

```
============================================================
❌ Tool Execution Failed: 'fetch_data'
============================================================

Error: ConnectionError: Could not reach api.example.com
Parameters: {'url': 'https://api.example.com/data'}

💡 Check that:
  - All required parameters are provided
  - Parameter types match the tool's schema
  - The tool function is correctly implemented

============================================================
```

---

## ProviderConfigurationError

**When:** A provider is created without the required API key or configuration.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `provider_name` | `str` | Provider name (e.g. `"OpenAI"`) |
| `missing_config` | `str` | What's missing |
| `env_var` | `str` | Environment variable to set |

**Example output:**

```
============================================================
❌ Provider Configuration Error: 'OpenAI'
============================================================

Missing: API key

💡 How to fix:
  1. Set the environment variable:
     export OPENAI_API_KEY='your-api-key'
  2. Or pass it directly:
     provider = OpenAIProvider(api_key='your-api-key')

============================================================
```

---

## MemoryLimitExceededError

**When:** Conversation memory exceeds its configured limit. In practice, the sliding window trims automatically, so this is only raised if there's an explicit constraint violation.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `current` | `int` | Current count |
| `limit` | `int` | Configured limit |
| `limit_type` | `str` | `"messages"` or `"tokens"` |

**Example output:**

```
============================================================
⚠️  Memory Limit Exceeded
============================================================

Limit Type: messages
Current: 40
Limit: 20

💡 Suggestions:
  - Increase max_messages: ConversationMemory(max_messages=40)
  - Clear older messages manually: memory.clear()

============================================================
```

---

## Other Errors

These are not selectools-specific but you may encounter them:

| Error | When |
|---|---|
| `GuardrailError` | A guardrail with `action=block` rejected content (see [GUARDRAILS.md](GUARDRAILS.md)) |
| `ProviderError` | An LLM API request failed (raised by providers, caught/retried by agent) |
| `ValueError` | Invalid configuration (e.g. `get_tools_by_category("invalid")`) |

```python
from selectools.guardrails import GuardrailError
from selectools.providers.base import ProviderError

try:
    result = agent.ask("...")
except GuardrailError as e:
    print(f"Content blocked by {e.guardrail_name}: {e.reason}")
except ProviderError:
    print("LLM provider failed after all retries")
```

---

## Best Practice: Production Error Handling

```python
from selectools import Agent, SelectoolsError
from selectools.guardrails import GuardrailError
from selectools.providers.base import ProviderError

def safe_ask(agent: Agent, prompt: str) -> str:
    try:
        result = agent.ask(prompt)
        return result.content
    except GuardrailError as e:
        return f"Your request was blocked: {e.reason}"
    except ProviderError:
        return "The AI service is temporarily unavailable. Please try again."
    except SelectoolsError as e:
        log.error(f"Agent error: {e}")
        return "Something went wrong processing your request."
```
