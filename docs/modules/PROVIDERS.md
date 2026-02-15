# Providers Module

**Directory:** `src/selectools/providers/`
**Files:** `base.py`, `openai_provider.py`, `anthropic_provider.py`, `gemini_provider.py`, `ollama_provider.py`

## Table of Contents

1. [Overview](#overview)
2. [Provider Protocol](#provider-protocol)
3. [Provider Implementations](#provider-implementations)
4. [Message Formatting](#message-formatting)
5. [Native Tool Calling](#native-tool-calling)
6. [Cost Calculation](#cost-calculation)
7. [Implementation Details](#implementation-details)

---

## Overview

**Providers** are adapters that translate between selectools' unified interface and specific LLM APIs. They handle:

- API authentication and configuration
- Message format conversion
- Role mapping
- Image encoding (for vision models)
- Streaming implementation
- Usage statistics extraction
- Error handling

### Design Goal

**Provider Agnosticism**: Switch LLM backends with one line of code, no refactoring required.

---

## Provider Protocol

### Interface Definition

```python
from typing import Protocol, runtime_checkable, List, Optional, Union, AsyncGenerator
from ..types import Message, ToolCall
from ..tools import Tool
from ..usage import UsageStats

@runtime_checkable
class Provider(Protocol):
    """Interface every provider adapter must satisfy."""

    name: str                    # Provider identifier
    supports_streaming: bool     # Can stream responses
    supports_async: bool = False # Has async methods

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,  # Native tool calling
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, UsageStats]:
        """Return assistant Message (with optional tool_calls) and usage stats."""
        ...

    def stream(self, *, model, system_prompt, messages, **kwargs):
        """Yield assistant text chunks (no usage stats)."""
        ...

    async def acomplete(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> tuple[Message, UsageStats]:
        """Async version of complete()."""
        ...

    async def astream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: float | None = None,
    ) -> AsyncGenerator[Union[str, ToolCall], None]:
        """Async streaming with native tool call support.

        Yields:
            str: Text content deltas
            ToolCall: Complete tool call objects when ready
        """
        ...
```

### Key Requirements

1. **Sync Methods**: `complete()` and `stream()` must be implemented
2. **Return Types**: `complete()` returns `(Message, UsageStats)` — Message may contain `tool_calls`
3. **Streaming**: `stream()` yields strings; `astream()` yields `Union[str, ToolCall]`
4. **Native Tool Calling**: Pass `tools` parameter for provider-native function calling
5. **Async**: Recommended for performance; `acomplete()` and `astream()`

---

## Provider Implementations

### OpenAI Provider

```python
from selectools.providers import OpenAIProvider
from selectools.models import OpenAI

provider = OpenAIProvider(
    api_key="sk-...",  # Or set OPENAI_API_KEY env var
    default_model=OpenAI.GPT_4O.id
)

# Features:
# - Streaming support
# - Async support (acomplete/astream)
# - Vision support (image_path in messages)
# - Full usage stats
# - Native tool calling (function calling API)
```

**API:** OpenAI Chat Completions API

### Anthropic Provider

```python
from selectools.providers import AnthropicProvider
from selectools.models import Anthropic

provider = AnthropicProvider(
    api_key="sk-ant-...",  # Or set ANTHROPIC_API_KEY
    default_model=Anthropic.SONNET_4_5.id
)

# Features:
# - Streaming support
# - Async support
# - Vision support (model-dependent)
# - Full usage stats
# - Native tool calling (function calling API)
```

**API:** Anthropic Messages API

### Gemini Provider

```python
from selectools.providers import GeminiProvider
from selectools.models import Gemini

provider = GeminiProvider(
    api_key="...",  # Or set GEMINI_API_KEY or GOOGLE_API_KEY
    default_model=Gemini.FLASH_2_0.id
)

# Features:
# - Streaming support
# - Async support
# - Vision support (model-dependent)
# - Free embeddings
# - Native tool calling (function calling API)
```

**API:** Google Generative AI

### Ollama Provider

```python
from selectools.providers import OllamaProvider
from selectools.models import Ollama

provider = OllamaProvider(
    host="http://localhost:11434",  # Default
    default_model=Ollama.LLAMA_3_2.id
)

# Features:
# - Local execution (privacy-first)
# - Zero cost
# - Streaming support
# - No API key required
```

**API:** Ollama REST API

### Local Provider (Testing)

```python
from selectools.providers.stubs import LocalProvider

provider = LocalProvider()

# Features:
# - No network calls
# - No API costs
# - Returns user's last message
# - Perfect for testing
```

---

## Message Formatting

### Unified Message Format

```python
from selectools.types import Message, Role

Message(role=Role.USER, content="Hello")
Message(role=Role.ASSISTANT, content="Hi there!")
Message(role=Role.TOOL, content="Result", tool_name="search")
Message(role=Role.USER, content="What's in this image?", image_path="./photo.jpg")
```

### Provider-Specific Formatting

#### OpenAI Format

```python
def _format_messages(self, system_prompt: str, messages: List[Message]):
    payload = [{"role": "system", "content": system_prompt}]

    for message in messages:
        role = message.role.value

        # Map TOOL role to ASSISTANT (OpenAI doesn't have TOOL role)
        if role == Role.TOOL.value:
            role = Role.ASSISTANT.value

        payload.append({
            "role": role,
            "content": self._format_content(message),
        })

    return payload

def _format_content(self, message: Message):
    if message.image_base64:
        # Vision: multimodal content
        return [
            {"type": "text", "text": message.content},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{message.image_base64}"},
            },
        ]
    return message.content
```

#### Anthropic Format

```python
def _format_messages(self, messages: List[Message]):
    formatted = []

    for message in messages:
        role = message.role.value

        # Anthropic uses "user" and "assistant" only
        if role == Role.TOOL.value:
            role = "assistant"

        formatted.append({
            "role": role,
            "content": message.content
        })

    return formatted

# System prompt is separate parameter
client.messages.create(
    model=model,
    system=system_prompt,  # Not in messages array
    messages=formatted
)
```

#### Gemini Format

```python
def _format_messages(self, system_prompt: str, messages: List[Message]):
    # Gemini combines system and conversation
    formatted = [{"role": "user", "parts": [system_prompt]}]

    for message in messages:
        role = "user" if message.role == Role.USER else "model"

        formatted.append({
            "role": role,
            "parts": [message.content]
        })

    return formatted
```

---

## Native Tool Calling

### Overview

All providers support native function calling APIs, which provide structured tool calls directly in the response instead of requiring text parsing.

### How It Works

1. Agent passes `tools` parameter to `complete()`/`acomplete()`
2. Provider converts tool schemas to provider-native format
3. LLM returns structured tool calls in `Message.tool_calls`
4. Agent detects `tool_calls` and executes them directly (no regex parsing needed)

### Provider Formats

#### OpenAI
```python
# Tools converted to OpenAI function format
tools=[{"type": "function", "function": {"name": "...", "parameters": {...}}}]

# Response contains tool_calls
response.choices[0].message.tool_calls  # List of tool call objects
```

#### Anthropic
```python
# Tools converted to Anthropic tool format
tools=[{"name": "...", "description": "...", "input_schema": {...}}]

# Response contains tool_use content blocks
response.content  # May contain ToolUse blocks with name and input
```

#### Gemini
```python
# Tools converted to Gemini function declarations
tools=[Tool(function_declarations=[...])]

# Response candidates contain function calls
response.candidates[0].content.parts  # May contain function_call parts
```

### Fallback

If a provider doesn't support native tool calling (e.g., Ollama), or if native calls are not present in the response, the agent falls back to regex-based parsing via `ToolCallParser`.

---

## Cost Calculation

### Usage Stats Extraction

Each provider extracts token counts from API responses:

#### OpenAI

```python
response = client.chat.completions.create(...)

usage_stats = UsageStats(
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens,
    total_tokens=response.usage.total_tokens,
    cost_usd=calculate_cost(model, prompt_tokens, completion_tokens),
    model=model,
    provider="openai"
)
```

#### Anthropic

```python
response = client.messages.create(...)

usage_stats = UsageStats(
    prompt_tokens=response.usage.input_tokens,
    completion_tokens=response.usage.output_tokens,
    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
    cost_usd=calculate_cost(model, input_tokens, output_tokens),
    model=model,
    provider="anthropic"
)
```

#### Gemini

```python
response = model.generate_content(...)

usage_stats = UsageStats(
    prompt_tokens=response.usage_metadata.prompt_token_count,
    completion_tokens=response.usage_metadata.candidates_token_count,
    total_tokens=response.usage_metadata.total_token_count,
    cost_usd=calculate_cost(model, prompt_tokens, completion_tokens),
    model=model,
    provider="gemini"
)
```

### Cost Calculation

```python
from selectools.pricing import calculate_cost

cost = calculate_cost(
    model="gpt-4o",
    prompt_tokens=1000,
    completion_tokens=500
)

# Looks up pricing from models registry:
# OpenAI.GPT_4O: prompt_cost=2.50, completion_cost=10.00 per 1M tokens
# Cost = (1000/1M * 2.50) + (500/1M * 10.00) = $0.0025 + $0.005 = $0.0075
```

---

## Implementation Details

### OpenAI Provider

```python
class OpenAIProvider(Provider):
    name = "openai"
    supports_streaming = True
    supports_async = True

    def __init__(self, api_key: str | None = None, default_model: str = "gpt-4o"):
        from openai import OpenAI, AsyncOpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderConfigurationError(...)

        self._client = OpenAI(api_key=self.api_key)
        self._async_client = AsyncOpenAI(api_key=self.api_key)
        self.default_model = default_model

    def complete(self, *, model, system_prompt, messages, temperature, max_tokens, timeout):
        formatted = self._format_messages(system_prompt, messages)

        response = self._client.chat.completions.create(
            model=model or self.default_model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        content = response.choices[0].message.content
        usage_stats = self._extract_usage(response, model)

        return content or "", usage_stats

    def stream(self, *, model, system_prompt, messages, temperature, max_tokens, timeout):
        formatted = self._format_messages(system_prompt, messages)

        response = self._client.chat.completions.create(
            model=model or self.default_model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=timeout,
        )

        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
```

### Async Streaming (`astream`)

All providers implement `astream()` for E2E streaming with native tool support:

```python
async def astream(self, *, model, system_prompt, messages, tools=None, ...):
    """Yield text deltas and ToolCall objects."""
    # Stream response from provider
    async for chunk in self._async_client.chat.completions.create(stream=True, ...):
        # Yield text deltas
        if delta.content:
            yield delta.content

        # Accumulate tool call deltas
        if delta.tool_calls:
            # ... accumulate until complete ...
            yield ToolCall(tool_name=name, parameters=args, id=tc_id)
```

The agent's `astream()` method consumes these and:
- Yields `StreamChunk` objects for text
- Executes tool calls when received
- Continues the agent loop until completion

### Error Handling

```python
def complete(self, ...):
    try:
        response = self._client.chat.completions.create(...)
        return content, usage_stats
    except Exception as exc:
        raise ProviderError(f"OpenAI completion failed: {exc}") from exc
```

### Async Implementation

```python
async def acomplete(self, *, model, system_prompt, messages, ...):
    formatted = self._format_messages(system_prompt, messages)

    response = await self._async_client.chat.completions.create(
        model=model or self.default_model,
        messages=formatted,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    content = response.choices[0].message.content
    usage_stats = self._extract_usage(response, model)

    return content or "", usage_stats
```

---

## Best Practices

### 1. Set API Keys via Environment

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

```python
# No need to pass api_key
provider = OpenAIProvider()
```

### 2. Use Model Constants

```python
from selectools.models import OpenAI, Anthropic, Gemini

# ✅ Good - Type-safe, autocomplete
provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)

# ❌ Bad - Prone to typos
provider = OpenAIProvider(default_model="gpt-4o-mini")
```

### 3. Handle Provider Errors

```python
from selectools.providers.base import ProviderError

try:
    response, stats = provider.complete(...)
except ProviderError as e:
    logger.error(f"Provider failed: {e}")
    # Fallback logic
```

### 4. Test with Local Provider

```python
from selectools.providers.stubs import LocalProvider

# Development/testing
if os.getenv("ENV") == "test":
    provider = LocalProvider()
else:
    provider = OpenAIProvider()
```

---

## Adding a New Provider

### Steps

1. **Create provider file** in `src/selectools/providers/`
2. **Implement Provider protocol**
3. **Handle message formatting**
4. **Extract usage stats**
5. **Add to exports** in `__init__.py`

### Template

```python
from ..types import Message
from ..usage import UsageStats
from ..pricing import calculate_cost
from .base import Provider, ProviderError

class MyProvider(Provider):
    name = "my_provider"
    supports_streaming = True
    supports_async = False

    def __init__(self, api_key: str, default_model: str = "default-model"):
        self.api_key = api_key
        self.default_model = default_model
        # Initialize client

    def complete(self, *, model, system_prompt, messages, temperature, max_tokens, timeout):
        # Format messages
        formatted = self._format_messages(system_prompt, messages)

        try:
            # Call API
            response = self.client.complete(...)

            # Extract content
            content = response.text

            # Extract usage
            usage_stats = UsageStats(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                cost_usd=calculate_cost(model, ...),
                model=model,
                provider=self.name
            )

            return content, usage_stats

        except Exception as exc:
            raise ProviderError(f"{self.name} failed: {exc}") from exc

    def stream(self, ...):
        # Stream implementation
        for chunk in response:
            yield chunk.text

    def _format_messages(self, system_prompt, messages):
        # Convert to provider's format
        pass
```

---

## Testing

```python
def test_openai_provider():
    provider = OpenAIProvider(api_key="test-key", default_model="gpt-4o-mini")

    messages = [Message(role=Role.USER, content="Hello")]

    response, stats = provider.complete(
        model="gpt-4o-mini",
        system_prompt="You are helpful",
        messages=messages,
        temperature=0.0,
        max_tokens=100
    )

    assert isinstance(response, str)
    assert stats.total_tokens > 0
    assert stats.cost_usd >= 0

def test_provider_switching():
    # Same agent code works with any provider
    for provider in [OpenAIProvider(), AnthropicProvider(), GeminiProvider()]:
        agent = Agent(tools=[...], provider=provider)
        response = agent.run([Message(role=Role.USER, content="Test")])
        assert response.content
```

---

## Further Reading

- [Agent Module](AGENT.md) - How agents use providers
- [Models Module](MODELS.md) - Model registry and pricing
- [Usage Module](USAGE.md) - Usage statistics

---

**Next Steps:** Learn about usage tracking in the [Usage Module](USAGE.md).
