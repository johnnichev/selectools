# ADR-004: Template Method Base Class for OpenAI-Compatible Providers

**Status**: Accepted
**Date**: 2026-03-15
**Deciders**: Core maintainers

## Context

The 4 provider implementations totaled 1776 lines. OpenAI (421 lines) and Ollama (456 lines) shared ~95% of their code: same SDK (OpenAI Python client), same message formatting, same tool schema mapping, same response parsing. The only differences were:
- Error message strings ("OpenAI" vs "Ollama" + connection hints)
- Pricing (`calculate_cost()` vs `0.0`)
- `max_tokens` vs `max_completion_tokens` (OpenAI-specific for newer models)
- Tool call ID handling (Ollama generates UUIDs for missing IDs)

Gemini and Anthropic use completely different SDKs with different APIs, so they share almost nothing with the OpenAI pair.

## Decision

Create `_OpenAICompatibleBase` (ABC) in `providers/_openai_compat.py` using the Template Method pattern. OpenAI and Ollama inherit from it and override only the varying parts via hook methods:

- `_get_token_key(model)` — token parameter name
- `_calculate_cost(model, prompt_tokens, completion_tokens)` — pricing
- `_get_provider_name()` — for UsageStats
- `_wrap_error(exc, operation)` — error message formatting
- `_parse_tool_call_id(tc)` — ID extraction/generation

Do NOT create a shared base for Gemini or Anthropic.

## Rationale

1. **High duplication, low divergence**: OpenAI and Ollama had 5 small differences in 400+ lines of identical code. The Template Method pattern isolates exactly these differences as overridable hooks.

2. **Gemini/Anthropic excluded deliberately**: They use `google-genai` and `anthropic` SDKs respectively — the message formats, response structures, and streaming APIs are fundamentally different. Forcing them into a shared base would require so many abstract methods that the base class would be hollow.

3. **ABC, not Protocol**: The base class contains real implementation (~400 lines of shared logic). It's an implementation inheritance hierarchy, not an interface contract. `typing.Protocol` (ADR-001) remains the public interface.

4. **Private class**: `_OpenAICompatibleBase` is prefixed with underscore and not exported. It's an internal implementation detail. Users interact with `OpenAIProvider` and `OllamaProvider`.

## Consequences

- **Positive**: OpenAI provider went from 421 to 86 lines (-80%). Ollama from 456 to 126 lines (-72%).
- **Positive**: Bug fixes to response parsing, message formatting, or streaming now happen in one place.
- **Positive**: Adding a new OpenAI-compatible provider (Azure OpenAI, Groq, Together AI) requires only `__init__` + 5 template method overrides.
- **Negative**: Debugging requires following the Template Method dispatch. Stack traces show `_OpenAICompatibleBase.complete()` rather than `OpenAIProvider.complete()`. Mitigated by clear method naming and short override methods.
- **Negative**: Gemini and Anthropic remain standalone. Their duplication (complete/acomplete are similar within each) is accepted as the cost of SDK divergence.
