# ADR-001: Protocol over ABC for Provider Interface

**Status**: Accepted
**Date**: 2026-03-15
**Deciders**: Core maintainers

## Context

The `Provider` interface defines the contract that all LLM adapters (OpenAI, Anthropic, Gemini, Ollama) must satisfy. We needed to choose between `typing.Protocol` (structural subtyping) and `abc.ABC` (nominal subtyping).

At the time of the decision, selectools supported 4 providers and users were building custom adapters for internal LLM endpoints.

## Decision

Use `typing.Protocol` with `@runtime_checkable` for the `Provider` interface in `providers/base.py`.

## Rationale

1. **No forced inheritance**: Third-party providers don't need to import or inherit from our base class. Any object with the right methods works. This matters for users wrapping proprietary APIs behind corporate firewalls where adding a selectools dependency to internal packages is undesirable.

2. **Structural typing aligns with Python idioms**: Duck typing is the Pythonic norm. Protocol formalizes it with static type checking support. `isinstance(obj, Provider)` works at runtime thanks to `@runtime_checkable`.

3. **Avoids diamond inheritance**: With ABC, adding a shared base class (like `_OpenAICompatibleBase` in ADR-004) would create fragile MRO chains. Protocol keeps the inheritance hierarchy flat — concrete providers can inherit from whatever implementation base makes sense without conflicting with the interface.

4. **Better IDE support**: mypy and pyright can verify Protocol conformance without running code, catching missing methods at type-check time rather than at import or first call.

## Consequences

- **Positive**: Users can create providers that satisfy the protocol without any selectools import. The `FallbackProvider` can wrap any protocol-conforming object.
- **Positive**: The `_OpenAICompatibleBase` ABC (ADR-004) coexists cleanly — it's an implementation detail, not part of the public interface.
- **Negative**: Protocol methods don't have enforced implementations at class definition time. A provider missing `acomplete()` only fails when the agent tries to call it. Mitigated by the architecture fitness tests (`tests/test_architecture.py`) that verify all concrete providers satisfy the full protocol.
- **Negative**: `@runtime_checkable` only checks method existence, not signatures. A provider with `complete(self)` (wrong signature) would pass `isinstance` but fail at call time.
