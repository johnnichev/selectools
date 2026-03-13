# Selectools Provider Implementation

Skill for adding or modifying LLM provider adapters in selectools.

## Trigger

Use when adding a new LLM provider, fixing provider bugs, or modifying provider behavior (e.g., adding Bedrock, fixing streaming, updating message formatting).

## Context

- **Provider protocol**: `src/selectools/providers/base.py` defines the `Provider` protocol
- **Existing providers**: `openai_provider.py`, `anthropic_provider.py`, `gemini_provider.py`, `ollama_provider.py`, `fallback.py`, `stubs.py`
- **Model registry**: `src/selectools/models.py` — single source of truth for 146 models with pricing

## Protocol Requirements

Every provider MUST implement all 5 methods:

```python
class NewProvider:
    name: str = "provider_name"
    supports_streaming: bool = True
    supports_async: bool = True

    def complete(self, *, model, system_prompt, messages, tools=None, temperature=None, max_tokens=None, **kwargs) -> Message: ...
    async def acomplete(self, *, model, system_prompt, messages, tools=None, **kwargs) -> Message: ...
    def stream(self, *, model, system_prompt, messages, tools=None, **kwargs) -> Iterable[str]: ...
    async def astream(self, *, model, system_prompt, messages, tools=None, **kwargs) -> AsyncIterable[Union[str, ToolCall]]: ...
    def _format_messages(self, messages: List[Message]) -> list: ...
```

## Critical Rules

1. **ALL methods MUST accept and forward `tools` parameter** — this was a bug across ALL providers
2. **`astream()` must yield `ToolCall` objects** — never stringify them
3. **`_format_messages()` must handle all Role types**:
   - `Role.USER` -> provider user format
   - `Role.ASSISTANT` -> provider assistant format (include `tool_calls` if present)
   - `Role.TOOL` -> provider-specific tool result format (varies per provider)
   - `Role.SYSTEM` -> provider system format

### Provider-Specific Tool Result Formatting

```
OpenAI:    {"role": "tool", "content": ..., "tool_call_id": ...}
Anthropic: {"role": "user", "content": [{"type": "tool_result", ...}]}
Gemini:    {"role": "user", "parts": [{"function_response": ...}]}
Ollama:    {"role": "tool", "content": ..., "tool_call_id": ...}
```

## Tool Mapping

Each provider needs a `_map_tool_to_<provider>(tool: Tool)` helper that converts a selectools `Tool` to the provider's expected tool/function schema format.

## OpenAI-Specific

Use `_uses_max_completion_tokens(model)` to choose between `max_tokens` and `max_completion_tokens`. GPT-5.x, o-series, and GPT-4.1 models require `max_completion_tokens`.

## FallbackProvider Pattern

`astream()` must include:
- try/except with `_is_retriable(error)` check
- `_record_failure(provider_index)` on error
- `_record_success(provider_index)` on success
- `on_fallback` callback invocation
- Circuit breaker state check before attempting provider

## Model Registry Updates

When adding a new provider, add its models to `src/selectools/models.py`:
```python
ModelInfo(
    id="model-name",
    provider="provider_name",
    input_price=X.XX,
    output_price=X.XX,
    context_window=NNNNN,
    max_output=NNNNN,
)
```

Update model count assertions in tests.

## Testing

- Use `RecordingProvider` to verify exact args passed to methods
- Test `_format_messages()` with all Role types including TOOL and ASSISTANT with tool_calls
- Test `astream()` yields `ToolCall` objects, not strings
- Test FallbackProvider failover, circuit breaker, and callback
- Test tool mapping produces correct provider-specific schema
