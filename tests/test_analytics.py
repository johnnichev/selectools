"""
Tests for the analytics module and agent integration.
"""

import json

# Import FakeProvider from test_framework
import sys
import tempfile
from pathlib import Path

import pytest

from selectools import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from selectools.analytics import AgentAnalytics, ToolMetrics

tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))
from test_framework import FakeProvider  # noqa: E402


@pytest.fixture
def simple_tool():
    """Simple tool for testing analytics."""

    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    return Tool(
        name="greet",
        description="Greet someone by name",
        parameters=[ToolParameter(name="name", param_type=str, description="Person's name")],
        function=greet,
    )


@pytest.fixture
def calculator_tool():
    """Calculator tool for testing analytics."""

    def calculate(expression: str) -> str:
        """Calculate a math expression."""
        result = eval(expression)  # noqa: S307
        return str(result)

    return Tool(
        name="calculate",
        description="Calculate a mathematical expression",
        parameters=[
            ToolParameter(name="expression", param_type=str, description="Math expression")
        ],
        function=calculate,
    )


@pytest.fixture
def failing_tool():
    """Tool that always fails for testing error tracking."""

    def fail(message: str) -> str:
        """Always raises an error."""
        raise ValueError(f"Intentional failure: {message}")

    return Tool(
        name="fail",
        description="A tool that always fails",
        parameters=[ToolParameter(name="message", param_type=str, description="Error message")],
        function=fail,
    )


class TestToolMetrics:
    """Tests for ToolMetrics class."""

    def test_initial_metrics(self):
        """Test initial metrics values."""
        metrics = ToolMetrics(name="test_tool")

        assert metrics.name == "test_tool"
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.total_duration == 0.0
        assert metrics.total_cost == 0.0
        assert metrics.parameter_usage == {}

    def test_success_rate_zero_calls(self):
        """Test success rate with zero calls."""
        metrics = ToolMetrics(name="test_tool")
        assert metrics.success_rate == 0.0

    def test_success_rate_all_successful(self):
        """Test success rate with all successful calls."""
        metrics = ToolMetrics(name="test_tool", total_calls=10, successful_calls=10, failed_calls=0)
        assert metrics.success_rate == 100.0

    def test_success_rate_mixed(self):
        """Test success rate with mixed results."""
        metrics = ToolMetrics(name="test_tool", total_calls=10, successful_calls=7, failed_calls=3)
        assert metrics.success_rate == 70.0

    def test_failure_rate(self):
        """Test failure rate calculation."""
        metrics = ToolMetrics(name="test_tool", total_calls=10, successful_calls=7, failed_calls=3)
        assert metrics.failure_rate == 30.0

    def test_avg_duration_zero_calls(self):
        """Test average duration with zero calls."""
        metrics = ToolMetrics(name="test_tool")
        assert metrics.avg_duration == 0.0

    def test_avg_duration(self):
        """Test average duration calculation."""
        metrics = ToolMetrics(name="test_tool", total_calls=5, total_duration=10.0)
        assert metrics.avg_duration == 2.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ToolMetrics(
            name="test_tool",
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
            total_duration=15.5,
            total_cost=0.05,
            parameter_usage={"param1": {"value1": 5, "value2": 5}},
        )

        result = metrics.to_dict()

        assert result["name"] == "test_tool"
        assert result["total_calls"] == 10
        assert result["successful_calls"] == 8
        assert result["failed_calls"] == 2
        assert result["success_rate"] == 80.0
        assert result["failure_rate"] == 20.0
        assert result["total_duration"] == 15.5
        assert result["avg_duration"] == 1.55
        assert result["total_cost"] == 0.05
        assert result["parameter_usage"] == {"param1": {"value1": 5, "value2": 5}}


class TestAgentAnalytics:
    """Tests for AgentAnalytics class."""

    def test_initial_state(self):
        """Test initial analytics state."""
        analytics = AgentAnalytics()
        assert analytics.get_all_metrics() == {}
        assert analytics.summary() == "No tool usage data collected."

    def test_record_successful_call(self):
        """Test recording a successful tool call."""
        analytics = AgentAnalytics()
        analytics.record_tool_call(
            tool_name="test_tool",
            success=True,
            duration=1.5,
            params={"arg": "value"},
            cost=0.01,
        )

        metrics = analytics.get_metrics("test_tool")
        assert metrics is not None
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0
        assert metrics.total_duration == 1.5
        assert metrics.total_cost == 0.01

    def test_record_failed_call(self):
        """Test recording a failed tool call."""
        analytics = AgentAnalytics()
        analytics.record_tool_call(
            tool_name="test_tool",
            success=False,
            duration=0.5,
            params={"arg": "value"},
            cost=0.0,
        )

        metrics = analytics.get_metrics("test_tool")
        assert metrics is not None
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 1

    def test_record_multiple_calls(self):
        """Test recording multiple calls to same tool."""
        analytics = AgentAnalytics()

        analytics.record_tool_call("tool1", True, 1.0, {"x": 1}, 0.01)
        analytics.record_tool_call("tool1", True, 2.0, {"x": 2}, 0.02)
        analytics.record_tool_call("tool1", False, 0.5, {"x": 3}, 0.0)

        metrics = analytics.get_metrics("tool1")
        assert metrics.total_calls == 3
        assert metrics.successful_calls == 2
        assert metrics.failed_calls == 1
        assert metrics.total_duration == 3.5
        assert metrics.total_cost == 0.03

    def test_parameter_usage_tracking(self):
        """Test parameter usage tracking."""
        analytics = AgentAnalytics()

        analytics.record_tool_call("tool1", True, 1.0, {"mode": "fast", "limit": 10})
        analytics.record_tool_call("tool1", True, 1.0, {"mode": "fast", "limit": 20})
        analytics.record_tool_call("tool1", True, 1.0, {"mode": "slow", "limit": 10})

        metrics = analytics.get_metrics("tool1")
        assert "mode" in metrics.parameter_usage
        assert "limit" in metrics.parameter_usage
        assert metrics.parameter_usage["mode"]["fast"] == 2
        assert metrics.parameter_usage["mode"]["slow"] == 1
        assert metrics.parameter_usage["limit"]["10"] == 2
        assert metrics.parameter_usage["limit"]["20"] == 1

    def test_parameter_value_truncation(self):
        """Test that long parameter values are truncated."""
        analytics = AgentAnalytics()

        long_value = "x" * 150
        analytics.record_tool_call("tool1", True, 1.0, {"data": long_value})

        metrics = analytics.get_metrics("tool1")
        # Should be truncated to 100 chars + "..."
        tracked_values = list(metrics.parameter_usage["data"].keys())
        assert len(tracked_values) == 1
        assert len(tracked_values[0]) == 103  # 100 + "..."
        assert tracked_values[0].endswith("...")

    def test_get_metrics_nonexistent(self):
        """Test getting metrics for nonexistent tool."""
        analytics = AgentAnalytics()
        assert analytics.get_metrics("nonexistent") is None

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        analytics = AgentAnalytics()

        analytics.record_tool_call("tool1", True, 1.0, {})
        analytics.record_tool_call("tool2", True, 1.0, {})

        all_metrics = analytics.get_all_metrics()
        assert len(all_metrics) == 2
        assert "tool1" in all_metrics
        assert "tool2" in all_metrics

    def test_summary(self):
        """Test summary generation."""
        analytics = AgentAnalytics()

        analytics.record_tool_call("tool1", True, 1.0, {"x": 1}, 0.01)
        analytics.record_tool_call("tool1", True, 2.0, {"x": 1}, 0.02)
        analytics.record_tool_call("tool2", False, 0.5, {}, 0.0)

        summary = analytics.summary()

        assert "Tool Usage Analytics" in summary
        assert "tool1" in summary
        assert "tool2" in summary
        assert "Calls: 2" in summary
        assert "Calls: 1" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        analytics = AgentAnalytics()

        analytics.record_tool_call("tool1", True, 1.0, {}, 0.01)
        analytics.record_tool_call("tool2", True, 1.0, {}, 0.02)

        result = analytics.to_dict()

        assert "tools" in result
        assert "summary" in result
        assert "tool1" in result["tools"]
        assert "tool2" in result["tools"]
        assert result["summary"]["total_tools"] == 2
        assert result["summary"]["total_calls"] == 2
        assert result["summary"]["total_cost"] == 0.03

    def test_to_json(self):
        """Test JSON export."""
        analytics = AgentAnalytics()
        analytics.record_tool_call("tool1", True, 1.0, {"x": 1}, 0.01)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = Path(f.name)

        try:
            analytics.to_json(temp_path)

            with open(temp_path) as f:
                data = json.load(f)

            assert "tools" in data
            assert "summary" in data
            assert "tool1" in data["tools"]
        finally:
            temp_path.unlink()

    def test_to_csv(self):
        """Test CSV export."""
        analytics = AgentAnalytics()
        analytics.record_tool_call("tool1", True, 1.5, {}, 0.01)
        analytics.record_tool_call("tool2", False, 0.5, {}, 0.0)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = Path(f.name)

        try:
            analytics.to_csv(temp_path)

            with open(temp_path) as f:
                lines = f.readlines()

            # Check header
            assert "tool_name" in lines[0]
            assert "total_calls" in lines[0]
            assert "success_rate" in lines[0]

            # Check data rows
            assert len(lines) == 3  # Header + 2 tools
            assert "tool1" in lines[1]
            assert "tool2" in lines[2]
        finally:
            temp_path.unlink()

    def test_to_csv_empty(self):
        """Test CSV export with no data."""
        analytics = AgentAnalytics()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = Path(f.name)

        try:
            analytics.to_csv(temp_path)

            with open(temp_path) as f:
                lines = f.readlines()

            # Should have header only
            assert len(lines) == 1
            assert "tool_name" in lines[0]
        finally:
            temp_path.unlink()

    def test_reset(self):
        """Test resetting analytics."""
        analytics = AgentAnalytics()
        analytics.record_tool_call("tool1", True, 1.0, {})

        assert len(analytics.get_all_metrics()) == 1

        analytics.reset()

        assert len(analytics.get_all_metrics()) == 0
        assert analytics.summary() == "No tool usage data collected."


class TestAgentIntegration:
    """Tests for analytics integration with Agent."""

    def test_analytics_disabled_by_default(self, simple_tool):
        """Test that analytics is disabled by default."""
        provider = FakeProvider(responses=["Done!"])
        agent = Agent(tools=[simple_tool], provider=provider)

        assert agent.analytics is None
        assert agent.get_analytics() is None

    def test_analytics_enabled(self, simple_tool):
        """Test enabling analytics in agent config."""
        provider = FakeProvider(responses=["Done!"])
        config = AgentConfig(enable_analytics=True)
        agent = Agent(tools=[simple_tool], provider=provider, config=config)

        assert agent.analytics is not None
        assert isinstance(agent.get_analytics(), AgentAnalytics)

    def test_analytics_tracks_successful_call(self, simple_tool):
        """Test that successful tool calls are tracked."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Alice"}}',
                "Done!",  # Final response after tool execution
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=3)
        agent = Agent(tools=[simple_tool], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Greet Alice")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("greet")

        assert metrics is not None
        assert metrics.total_calls >= 1
        assert metrics.successful_calls >= 1
        assert metrics.failed_calls == 0

    def test_analytics_tracks_failed_call(self, failing_tool):
        """Test that failed tool calls are tracked."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "fail", "parameters": {"message": "test error"}}',
                "Task failed.",  # Final response after failed tool execution
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=3)
        agent = Agent(tools=[failing_tool], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Test failure")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("fail")

        assert metrics is not None
        assert metrics.total_calls >= 1
        assert metrics.failed_calls >= 1
        assert metrics.successful_calls == 0

    def test_analytics_tracks_duration(self, simple_tool):
        """Test that call duration is tracked."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Bob"}}',
                "Done!",
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=3)
        agent = Agent(tools=[simple_tool], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Greet Bob")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("greet")

        assert metrics is not None
        assert metrics.total_duration > 0.0
        assert metrics.avg_duration > 0.0

    def test_analytics_tracks_parameters(self, simple_tool):
        """Test that parameters are tracked."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Charlie"}}',
                "Done!",
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=3)
        agent = Agent(tools=[simple_tool], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Greet Charlie")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("greet")

        assert metrics is not None
        assert "name" in metrics.parameter_usage
        assert "Charlie" in metrics.parameter_usage["name"]

    def test_analytics_multiple_tools(self, simple_tool, calculator_tool):
        """Test analytics with multiple tools."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Alice"}}',
                'TOOL_CALL: {"tool_name": "calculate", "parameters": {"expression": "5+3"}}',
                "Done with both tasks!",  # Final response
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=5)
        agent = Agent(tools=[simple_tool, calculator_tool], provider=provider, config=config)

        agent.run([Message(role=Role.USER, content="Greet Alice and calculate 5+3")])

        analytics = agent.get_analytics()

        greet_metrics = analytics.get_metrics("greet")
        calc_metrics = analytics.get_metrics("calculate")

        assert greet_metrics is not None
        assert calc_metrics is not None
        assert greet_metrics.total_calls >= 1
        assert calc_metrics.total_calls >= 1

    @pytest.mark.asyncio
    async def test_analytics_async_agent(self, simple_tool):
        """Test analytics with async agent execution."""
        provider = FakeProvider(
            responses=[
                'TOOL_CALL: {"tool_name": "greet", "parameters": {"name": "Dave"}}',
                "Done!",
            ]
        )
        config = AgentConfig(enable_analytics=True, max_iterations=3)
        agent = Agent(tools=[simple_tool], provider=provider, config=config)

        await agent.arun([Message(role=Role.USER, content="Greet Dave")])

        analytics = agent.get_analytics()
        metrics = analytics.get_metrics("greet")

        assert metrics is not None
        assert metrics.total_calls >= 1
        assert metrics.successful_calls >= 1
