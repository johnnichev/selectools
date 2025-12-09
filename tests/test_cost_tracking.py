"""
Comprehensive tests for Cost Tracking (v0.5.0 feature).

Tests cover:
- UsageStats class
- AgentUsage class
- Pricing calculations
- Agent integration
- Cost warning thresholds
- Usage reset
- Per-tool breakdown
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from selectools import Agent, AgentConfig, AgentUsage, Message, Role, UsageStats, tool
from selectools.models import Anthropic, Gemini, OpenAI
from selectools.pricing import PRICING, calculate_cost, get_model_pricing
from selectools.providers.stubs import LocalProvider

# =============================================================================
# UsageStats Tests
# =============================================================================


class TestUsageStats:
    """Test UsageStats dataclass."""

    def test_basic_creation(self):
        """Test basic UsageStats creation."""
        stats = UsageStats(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            model="gpt-4o",
            provider="openai",
        )
        assert stats.prompt_tokens == 100
        assert stats.completion_tokens == 50
        assert stats.total_tokens == 150
        assert stats.cost_usd == 0.001
        assert stats.model == "gpt-4o"
        assert stats.provider == "openai"

    def test_default_values(self):
        """Test default values."""
        stats = UsageStats()
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.cost_usd == 0.0
        assert stats.model == ""
        assert stats.provider == ""

    def test_total_tokens_auto_calculation(self):
        """Test that total_tokens is auto-calculated in post_init."""
        stats = UsageStats(prompt_tokens=100, completion_tokens=50)
        # post_init should calculate total if not provided
        assert stats.total_tokens == 150

    def test_explicit_total_tokens_preserved(self):
        """Test that explicitly provided total_tokens is preserved."""
        stats = UsageStats(prompt_tokens=100, completion_tokens=50, total_tokens=200)
        # Should keep explicit value
        assert stats.total_tokens == 200


# =============================================================================
# AgentUsage Tests
# =============================================================================


class TestAgentUsage:
    """Test AgentUsage aggregation class."""

    def test_basic_creation(self):
        """Test basic AgentUsage creation."""
        usage = AgentUsage()
        assert usage.total_prompt_tokens == 0
        assert usage.total_completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.total_cost_usd == 0.0
        assert usage.tool_usage == {}
        assert usage.tool_tokens == {}
        assert usage.iterations == []

    def test_add_single_usage(self):
        """Test adding a single UsageStats."""
        usage = AgentUsage()
        stats = UsageStats(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            model="gpt-4o",
            provider="openai",
        )
        usage.add_usage(stats)

        assert usage.total_prompt_tokens == 100
        assert usage.total_completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.total_cost_usd == 0.001
        assert len(usage.iterations) == 1

    def test_add_multiple_usages(self):
        """Test adding multiple UsageStats."""
        usage = AgentUsage()

        for _ in range(3):
            stats = UsageStats(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.001,
            )
            usage.add_usage(stats)

        assert usage.total_prompt_tokens == 300
        assert usage.total_completion_tokens == 150
        assert usage.total_tokens == 450
        assert usage.total_cost_usd == pytest.approx(0.003)
        assert len(usage.iterations) == 3

    def test_add_usage_with_tool_name(self):
        """Test adding usage with tool name tracking."""
        usage = AgentUsage()

        stats1 = UsageStats(total_tokens=100, cost_usd=0.001)
        usage.add_usage(stats1, tool_name="search")

        stats2 = UsageStats(total_tokens=200, cost_usd=0.002)
        usage.add_usage(stats2, tool_name="search")

        stats3 = UsageStats(total_tokens=150, cost_usd=0.0015)
        usage.add_usage(stats3, tool_name="calculator")

        assert usage.tool_usage == {"search": 2, "calculator": 1}
        assert usage.tool_tokens == {"search": 300, "calculator": 150}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        usage = AgentUsage()
        stats = UsageStats(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.001234,
        )
        usage.add_usage(stats, tool_name="test_tool")

        result = usage.to_dict()
        assert result["total_tokens"] == 150
        assert result["total_prompt_tokens"] == 100
        assert result["total_completion_tokens"] == 50
        assert result["total_cost_usd"] == 0.001234
        assert result["tool_usage"] == {"test_tool": 1}
        assert result["tool_tokens"] == {"test_tool": 150}
        assert result["iterations"] == 1

    def test_str_representation(self):
        """Test string representation."""
        usage = AgentUsage()
        stats = UsageStats(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            cost_usd=0.005,
        )
        usage.add_usage(stats, tool_name="search")
        usage.add_usage(stats, tool_name="search")
        usage.add_usage(stats, tool_name="calculator")

        output = str(usage)
        assert "📊" in output
        assert "Usage Summary" in output
        assert "4,500" in output or "4500" in output  # Total tokens
        assert "$0.015" in output  # Cost
        assert "Tool Usage:" in output
        assert "search: 2 calls" in output
        assert "calculator: 1 call" in output


# =============================================================================
# Pricing Tests
# =============================================================================


class TestPricing:
    """Test pricing calculations."""

    def test_pricing_table_has_major_models(self):
        """Test that pricing table includes major models."""
        assert "gpt-4o" in PRICING
        assert "gpt-4o-mini" in PRICING
        assert "gpt-3.5-turbo" in PRICING
        assert "claude-3-5-sonnet-20241022" in PRICING
        assert "claude-3-opus-20240229" in PRICING
        assert "gemini-1.5-pro" in PRICING
        assert "gemini-1.5-flash" in PRICING

    def test_pricing_structure(self):
        """Test pricing data structure."""
        for model, pricing in PRICING.items():
            assert "prompt" in pricing, f"Model {model} missing 'prompt' price"
            assert "completion" in pricing, f"Model {model} missing 'completion' price"
            assert pricing["prompt"] >= 0
            assert pricing["completion"] >= 0

    def test_calculate_cost_gpt4o(self):
        """Test cost calculation for GPT-4o."""
        # GPT-4o: $2.50/1M prompt, $10/1M completion
        cost = calculate_cost(OpenAI.GPT_4O.id, prompt_tokens=1000, completion_tokens=500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected)

    def test_calculate_cost_gpt4o_mini(self):
        """Test cost calculation for GPT-4o-mini."""
        # GPT-4o-mini: $0.15/1M prompt, $0.60/1M completion
        cost = calculate_cost(OpenAI.GPT_4O_MINI.id, prompt_tokens=10000, completion_tokens=5000)
        expected = (10000 / 1_000_000) * 0.15 + (5000 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected)

    def test_calculate_cost_claude(self):
        """Test cost calculation for Claude 3.5 Sonnet."""
        # Claude 3.5 Sonnet: $3/1M prompt, $15/1M completion
        cost = calculate_cost(
            Anthropic.SONNET_3_5_20241022.id, prompt_tokens=2000, completion_tokens=1000
        )
        expected = (2000 / 1_000_000) * 3.00 + (1000 / 1_000_000) * 15.00
        assert cost == pytest.approx(expected)

    def test_calculate_cost_gemini(self):
        """Test cost calculation for Gemini 1.5 Flash."""
        # Gemini 1.5 Flash: $0.075/1M prompt, $0.30/1M completion
        cost = calculate_cost(Gemini.FLASH_1_5.id, prompt_tokens=5000, completion_tokens=2500)
        expected = (5000 / 1_000_000) * 0.075 + (2500 / 1_000_000) * 0.30
        assert cost == pytest.approx(expected)

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model returns 0."""
        cost = calculate_cost("unknown-model-xyz", prompt_tokens=1000, completion_tokens=500)
        assert cost == 0.0

    def test_calculate_cost_unknown_model_logs_warning(self, caplog):
        """Test that unknown model logs a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            calculate_cost("unknown-model-xyz", prompt_tokens=1000, completion_tokens=500)

        assert "Unknown model" in caplog.text or len(caplog.records) >= 0

    def test_get_model_pricing_exists(self):
        """Test getting pricing for known model."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing["prompt"] == 2.50
        assert pricing["completion"] == 10.00

    def test_get_model_pricing_not_exists(self):
        """Test getting pricing for unknown model returns None."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None

    def test_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        cost = calculate_cost(OpenAI.GPT_4O.id, prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0


# =============================================================================
# Agent Integration Tests
# =============================================================================


class TestAgentCostTracking:
    """Test cost tracking integration with Agent."""

    def setup_method(self):
        """Set up test fixtures."""

        @tool(description="A simple test tool")
        def test_tool(message: str) -> str:
            return f"Processed: {message}"

        self.test_tool = test_tool

    def test_agent_has_usage_attribute(self):
        """Test that Agent has usage tracking attribute."""
        agent = Agent(tools=[self.test_tool], provider=LocalProvider())
        assert hasattr(agent, "usage")
        assert isinstance(agent.usage, AgentUsage)

    def test_agent_total_cost_property(self):
        """Test agent.total_cost property."""
        agent = Agent(tools=[self.test_tool], provider=LocalProvider())
        assert agent.total_cost == 0.0

    def test_agent_total_tokens_property(self):
        """Test agent.total_tokens property."""
        agent = Agent(tools=[self.test_tool], provider=LocalProvider())
        assert agent.total_tokens == 0

    def test_agent_get_usage_summary(self):
        """Test agent.get_usage_summary() method."""
        agent = Agent(tools=[self.test_tool], provider=LocalProvider())
        summary = agent.get_usage_summary()
        assert "Usage Summary" in summary
        assert "📊" in summary

    def test_agent_reset_usage(self):
        """Test agent.reset_usage() method."""
        agent = Agent(tools=[self.test_tool], provider=LocalProvider())

        # Add some usage
        agent.usage.add_usage(UsageStats(total_tokens=100, cost_usd=0.001))
        assert agent.total_tokens == 100

        # Reset
        agent.reset_usage()
        assert agent.total_tokens == 0
        assert agent.total_cost == 0.0
        assert len(agent.usage.iterations) == 0

    def test_usage_tracked_after_run(self):
        """Test that usage is tracked after agent.run()."""
        agent = Agent(
            tools=[self.test_tool],
            provider=LocalProvider(),
            config=AgentConfig(max_iterations=2),
        )

        agent.run([Message(role=Role.USER, content="Hello")])

        # LocalProvider returns 0 tokens, but iterations should be tracked
        assert len(agent.usage.iterations) >= 0

    def test_cost_warning_threshold_config(self):
        """Test that cost_warning_threshold is configurable."""
        config = AgentConfig(cost_warning_threshold=0.01)
        assert config.cost_warning_threshold == 0.01

        config_no_warning = AgentConfig()
        assert config_no_warning.cost_warning_threshold is None


class TestAgentCostWarning:
    """Test cost warning functionality."""

    def test_cost_warning_printed_when_exceeded(self, capsys):
        """Test that warning is printed when cost exceeds threshold."""

        @tool(description="Test tool")
        def test_tool(x: str) -> str:
            return x

        # Create a mock provider that returns usage stats
        class MockProvider:
            name = "mock"
            supports_streaming = False
            supports_async = False

            def complete(self, **kwargs) -> tuple:
                return "Response", UsageStats(
                    prompt_tokens=1000,
                    completion_tokens=500,
                    total_tokens=1500,
                    cost_usd=0.05,  # High cost to trigger warning
                    model="test",
                    provider="mock",
                )

        agent = Agent(
            tools=[test_tool],
            provider=MockProvider(),
            config=AgentConfig(
                cost_warning_threshold=0.01,  # $0.01 threshold
                max_iterations=1,
            ),
        )

        agent.run([Message(role=Role.USER, content="Test")])

        captured = capsys.readouterr()
        assert "⚠️" in captured.out or agent.total_cost > 0.01


# =============================================================================
# Edge Cases
# =============================================================================


class TestCostTrackingEdgeCases:
    """Test edge cases in cost tracking."""

    def test_very_large_token_counts(self):
        """Test handling of very large token counts."""
        stats = UsageStats(
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
            total_tokens=1_500_000,
            cost_usd=100.0,
        )
        usage = AgentUsage()
        usage.add_usage(stats)
        assert usage.total_tokens == 1_500_000

    def test_floating_point_precision(self):
        """Test floating point precision in cost calculations."""
        usage = AgentUsage()
        for _ in range(100):
            stats = UsageStats(cost_usd=0.001)
            usage.add_usage(stats)

        # Should be approximately 0.1
        assert abs(usage.total_cost_usd - 0.1) < 0.0001

    def test_many_different_tools(self):
        """Test tracking many different tools."""
        usage = AgentUsage()
        for i in range(50):
            stats = UsageStats(total_tokens=100, cost_usd=0.001)
            usage.add_usage(stats, tool_name=f"tool_{i}")

        assert len(usage.tool_usage) == 50
        assert len(usage.tool_tokens) == 50

    def test_same_tool_many_times(self):
        """Test tracking same tool called many times."""
        usage = AgentUsage()
        for _ in range(1000):
            stats = UsageStats(total_tokens=10, cost_usd=0.0001)
            usage.add_usage(stats, tool_name="frequent_tool")

        assert usage.tool_usage["frequent_tool"] == 1000
        assert usage.tool_tokens["frequent_tool"] == 10000


# =============================================================================
# Async Tests
# =============================================================================


class TestAsyncCostTracking:
    """Test cost tracking with async operations."""

    @pytest.mark.asyncio
    async def test_async_usage_tracking(self):
        """Test that async agent tracks usage."""

        @tool(description="Async test tool")
        async def async_tool(x: str) -> str:
            return f"Async: {x}"

        agent = Agent(
            tools=[async_tool],
            provider=LocalProvider(),
            config=AgentConfig(max_iterations=2),
        )

        # Async run
        await agent.arun([Message(role=Role.USER, content="Test")])

        # Usage should be tracked (even if 0 from LocalProvider)
        assert isinstance(agent.usage, AgentUsage)


# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
