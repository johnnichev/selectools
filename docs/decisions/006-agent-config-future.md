# ADR-006: Structured AgentConfig Deferred to Post-v0.17.0

**Status**: Deferred
**Date**: 2026-03-15
**Deciders**: Core maintainers

## Context

`AgentConfig` is a flat dataclass with 41+ fields mixing:
- **Static config**: `model`, `temperature`, `max_tokens`, `max_iterations`
- **Feature flags**: `coherence_check`, `screen_tool_output`, `summarize_on_trim`, `parallel_tool_execution`
- **Runtime dependencies**: `tool_policy`, `guardrails`, `cache`, `session_store`, `entity_memory`, `knowledge_graph`
- **Retry/timeout tuning**: `max_retries`, `retry_backoff_seconds`, `rate_limit_cooldown_seconds`, `tool_timeout_seconds`, `request_timeout`

The proposed solution was to group fields into nested dataclasses:
```python
@dataclass
class RetryConfig:
    max_retries: int = 2
    backoff_seconds: float = 1.0
    rate_limit_cooldown: float = 5.0

@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    retry: RetryConfig = field(default_factory=RetryConfig)
    ...
```

## Decision

**Defer** to post-v0.17.0. Do not restructure AgentConfig now.

## Rationale

1. **Breaking change surface**: Even with backward-compatible flat kwargs, any config restructuring risks breaking users who access `config.max_retries` directly (which becomes `config.retry.max_retries`). The migration cost is non-trivial for a library with production users.

2. **v0.17.0 will add more fields**: Multi-agent orchestration needs `graph_config`, `delegation_policy`, `agent_registry`, etc. Restructuring now and again in v0.17.0 means two breaking migrations instead of one.

3. **Low bug risk from current state**: The flat config is verbose but not buggy. Fields have sensible defaults, type hints are complete, and mypy catches assignment errors. The anti-pattern is aesthetic, not functional.

4. **Higher-priority work exists**: The agent decomposition (ADR-005), provider base class (ADR-004), and hooks deprecation (ADR-002) had direct impact on bug prevention and code maintainability. Config restructuring would not have prevented any of the bugs found in v0.16.0-v0.16.4.

## Plan for post-v0.17.0

When restructuring eventually happens:
1. Group into nested dataclasses (`RetryConfig`, `CoherenceConfig`, `ScreeningConfig`, `SummarizeConfig`)
2. Support both flat kwargs and nested config for one release cycle
3. Emit `DeprecationWarning` for flat kwargs
4. Remove flat kwargs in the following release

## Consequences

- **Positive**: No migration burden on users during the v0.16.x → v0.17.0 transition.
- **Positive**: v0.17.0's new config fields can be designed alongside the restructuring.
- **Negative**: The 41-field flat config remains until post-v0.17.0. New features (terminal actions, stop_condition, async observers) add more fields to the already-long list.
- **Mitigation**: IDE autocompletion and the docstring on `AgentConfig` make the fields discoverable despite the flat structure.
