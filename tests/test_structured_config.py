"""Tests for structured AgentConfig with nested config groups."""

from __future__ import annotations

import pytest

from selectools import (
    AgentConfig,
    BudgetConfig,
    CoherenceConfig,
    CompressConfig,
    RetryConfig,
    ToolConfig,
    TraceConfig,
)


class TestBackwardCompat:
    """All flat kwargs still work exactly as before."""

    def test_flat_kwargs_work(self):
        config = AgentConfig(max_retries=5, request_timeout=60.0)
        assert config.max_retries == 5
        assert config.request_timeout == 60.0

    def test_defaults_unchanged(self):
        config = AgentConfig()
        assert config.max_retries == 2
        assert config.retry_backoff_seconds == 1.0
        assert config.request_timeout == 30.0
        assert config.tool_timeout_seconds is None
        assert config.coherence_check is False
        assert config.compress_context is False

    def test_flat_config_populates_nested(self):
        """Flat kwargs auto-populate nested config groups."""
        config = AgentConfig(max_retries=5, request_timeout=60.0)
        assert config.retry.max_retries == 5
        assert config.retry.request_timeout == 60.0

    def test_all_flat_fields_accessible(self):
        """Every flat field from the old API still works."""
        config = AgentConfig(
            model="test",
            temperature=0.5,
            max_tokens=2000,
            max_iterations=10,
            verbose=True,
            stream=True,
            max_retries=3,
            tool_timeout_seconds=30.0,
            coherence_check=True,
            screen_tool_output=True,
            compress_context=True,
            max_total_tokens=50000,
            max_cost_usd=1.0,
        )
        assert config.model == "test"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.max_iterations == 10
        assert config.verbose is True
        assert config.stream is True
        assert config.max_retries == 3
        assert config.tool_timeout_seconds == 30.0
        assert config.coherence_check is True
        assert config.screen_tool_output is True
        assert config.compress_context is True
        assert config.max_total_tokens == 50000
        assert config.max_cost_usd == 1.0


class TestNestedConfig:
    """Nested config groups work and take precedence."""

    def test_retry_config(self):
        config = AgentConfig(retry=RetryConfig(max_retries=10, backoff_seconds=5.0))
        assert config.max_retries == 10
        assert config.retry_backoff_seconds == 5.0
        assert config.retry.max_retries == 10

    def test_tool_config(self):
        config = AgentConfig(tool=ToolConfig(timeout_seconds=30.0, parallel_execution=False))
        assert config.tool_timeout_seconds == 30.0
        assert config.parallel_tool_execution is False

    def test_coherence_config(self):
        config = AgentConfig(coherence=CoherenceConfig(enabled=True, fail_closed=True))
        assert config.coherence_check is True
        assert config.coherence_fail_closed is True

    def test_budget_config(self):
        config = AgentConfig(budget=BudgetConfig(max_total_tokens=50000, max_cost_usd=1.0))
        assert config.max_total_tokens == 50000
        assert config.max_cost_usd == 1.0

    def test_compress_config(self):
        config = AgentConfig(compress=CompressConfig(enabled=True, threshold=0.8, keep_recent=6))
        assert config.compress_context is True
        assert config.compress_threshold == 0.8
        assert config.compress_keep_recent == 6

    def test_trace_config(self):
        config = AgentConfig(trace=TraceConfig(tool_result_chars=500, metadata={"env": "prod"}))
        assert config.trace_tool_result_chars == 500
        assert config.trace_metadata == {"env": "prod"}

    def test_nested_takes_precedence_over_flat(self):
        """When both flat and nested are provided, nested wins."""
        config = AgentConfig(
            max_retries=3,  # flat
            retry=RetryConfig(max_retries=10),  # nested — should win
        )
        assert config.max_retries == 10

    def test_nested_config_accessible(self):
        """Nested config objects are accessible."""
        config = AgentConfig()
        assert isinstance(config.retry, RetryConfig)
        assert isinstance(config.tool, ToolConfig)


class TestImports:
    def test_all_config_groups_importable_from_selectools(self):
        from selectools import (
            BudgetConfig,
            CoherenceConfig,
            CompressConfig,
            GuardrailsConfig,
            MemoryConfig,
            RetryConfig,
            SessionConfig,
            SummarizeConfig,
            ToolConfig,
            TraceConfig,
        )

        assert RetryConfig is not None
        assert BudgetConfig is not None
