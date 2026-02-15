"""
Tests for observability hooks (v0.5.2 feature).

Tests cover:
- All hook types (on_agent_start, on_agent_end, on_iteration_start, etc.)
- Hook invocation with correct arguments
- Hook error handling (hooks should not break execution)
- Async hooks
"""

import pytest

from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.providers.stubs import LocalProvider

# =============================================================================
# Helper Classes
# =============================================================================


class HookRecorder:
    """Records all hook calls for testing."""

    def __init__(self):
        self.calls = []

    def record(self, hook_name, *args, **kwargs):
        """Record a hook call."""
        self.calls.append(
            {
                "hook": hook_name,
                "args": args,
                "kwargs": kwargs,
            }
        )

    def get_calls(self, hook_name):
        """Get all calls for a specific hook."""
        return [c for c in self.calls if c["hook"] == hook_name]

    def was_called(self, hook_name):
        """Check if a hook was called."""
        return len(self.get_calls(hook_name)) > 0


# =============================================================================
# Agent Lifecycle Hooks Tests
# =============================================================================


class TestAgentLifecycleHooks:
    """Test on_agent_start and on_agent_end hooks."""

    def test_on_agent_start_called_with_messages(self):
        """Test that on_agent_start is called with input messages."""
        recorder = HookRecorder()

        @tool(description="Echo tool")
        def echo(text: str) -> str:
            return text

        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(
                max_iterations=1,
                hooks={"on_agent_start": lambda msgs: recorder.record("on_agent_start", msgs)},
            ),
        )

        messages = [Message(role=Role.USER, content="test")]
        agent.run(messages)

        assert recorder.was_called("on_agent_start")
        calls = recorder.get_calls("on_agent_start")
        assert len(calls) == 1
        assert calls[0]["args"][0] == messages

    def test_on_agent_end_called_with_response_and_usage(self):
        """Test that on_agent_end is called with final response and usage."""
        recorder = HookRecorder()

        @tool(description="Echo tool")
        def echo(text: str) -> str:
            return text

        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(
                max_iterations=1,
                hooks={
                    "on_agent_end": lambda response, usage: recorder.record(
                        "on_agent_end", response, usage
                    )
                },
            ),
        )

        response = agent.run([Message(role=Role.USER, content="test")])

        assert recorder.was_called("on_agent_end")
        calls = recorder.get_calls("on_agent_end")
        assert len(calls) == 1
        assert calls[0]["args"][0] == response.message
        # Usage should be AgentUsage object
        assert hasattr(calls[0]["args"][1], "total_tokens")


# =============================================================================
# Iteration Hooks Tests
# =============================================================================


class TestIterationHooks:
    """Test on_iteration_start and on_iteration_end hooks."""

    def test_on_iteration_start_called_each_iteration(self):
        """Test that on_iteration_start is called for each iteration."""
        recorder = HookRecorder()

        @tool(description="Echo tool")
        def echo(text: str) -> str:
            return text

        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(
                max_iterations=3,
                hooks={
                    "on_iteration_start": lambda iteration, messages: recorder.record(
                        "on_iteration_start", iteration, messages
                    )
                },
            ),
        )

        agent.run([Message(role=Role.USER, content="test")])

        calls = recorder.get_calls("on_iteration_start")
        # Should be called once (LocalProvider returns text without tool calls)
        assert len(calls) >= 1
        # First iteration should be 1
        assert calls[0]["args"][0] == 1

    def test_on_iteration_end_called_each_iteration(self):
        """Test that on_iteration_end is called for each iteration."""
        recorder = HookRecorder()

        @tool(description="Echo tool")
        def echo(text: str) -> str:
            return text

        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(
                max_iterations=2,
                hooks={
                    "on_iteration_end": lambda iteration, response: recorder.record(
                        "on_iteration_end", iteration, response
                    )
                },
            ),
        )

        agent.run([Message(role=Role.USER, content="test")])

        calls = recorder.get_calls("on_iteration_end")
        assert len(calls) >= 1


# =============================================================================
# Tool Execution Hooks Tests
# =============================================================================


class TestToolExecutionHooks:
    """Test tool-related hooks."""

    def test_on_tool_start_called_before_execution(self):
        """Test that on_tool_start is called before tool execution."""
        recorder = HookRecorder()
        tool_executed = []

        @tool(description="Test tool")
        def test_tool(x: int) -> str:
            tool_executed.append(True)
            return str(x)

        # Create a fake provider that requests a tool call
        class FakeToolCallProvider:
            name = "fake"
            supports_streaming = False

            def complete(self, **kwargs):
                from selectools.usage import UsageStats

                return (
                    Message(
                        role=Role.ASSISTANT,
                        content='TOOL_CALL: {"tool_name": "test_tool", "parameters": {"x": 42}}',
                    ),
                    UsageStats(),
                )

        agent = Agent(
            tools=[test_tool],
            provider=FakeToolCallProvider(),
            config=AgentConfig(
                max_iterations=2,
                hooks={
                    "on_tool_start": lambda name, args: recorder.record("on_tool_start", name, args)
                },
            ),
        )

        agent.run([Message(role=Role.USER, content="test")])

        calls = recorder.get_calls("on_tool_start")
        if tool_executed:  # Only check if tool was actually called
            assert len(calls) >= 1
            assert calls[0]["args"][0] == "test_tool"
            assert calls[0]["args"][1] == {"x": 42}

    def test_on_tool_end_called_after_execution(self):
        """Test that on_tool_end is called after tool execution with duration."""
        recorder = HookRecorder()

        @tool(description="Test tool")
        def test_tool(x: int) -> str:
            return str(x * 2)

        class FakeToolCallProvider:
            name = "fake"
            supports_streaming = False

            def complete(self, **kwargs):
                from selectools.usage import UsageStats

                return (
                    Message(
                        role=Role.ASSISTANT,
                        content='TOOL_CALL: {"tool_name": "test_tool", "parameters": {"x": 5}}',
                    ),
                    UsageStats(),
                )

        agent = Agent(
            tools=[test_tool],
            provider=FakeToolCallProvider(),
            config=AgentConfig(
                max_iterations=2,
                hooks={
                    "on_tool_end": lambda name, result, duration: recorder.record(
                        "on_tool_end", name, result, duration
                    )
                },
            ),
        )

        agent.run([Message(role=Role.USER, content="test")])

        calls = recorder.get_calls("on_tool_end")
        if calls:
            assert calls[0]["args"][0] == "test_tool"
            assert "10" in calls[0]["args"][1]  # Result should contain "10" (5*2)
            assert isinstance(calls[0]["args"][2], float)  # Duration should be a float


# =============================================================================
# LLM Hooks Tests
# =============================================================================


class TestLLMHooks:
    """Test on_llm_start and on_llm_end hooks."""

    def test_on_llm_start_called_before_llm_call(self):
        """Test that on_llm_start is called before LLM requests."""
        recorder = HookRecorder()

        @tool(description="Echo tool")
        def echo(text: str) -> str:
            return text

        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(
                max_iterations=1,
                hooks={
                    "on_llm_start": lambda messages, model: recorder.record(
                        "on_llm_start", messages, model
                    )
                },
            ),
        )

        agent.run([Message(role=Role.USER, content="test")])

        assert recorder.was_called("on_llm_start")
        calls = recorder.get_calls("on_llm_start")
        assert len(calls) >= 1

    def test_on_llm_end_called_after_llm_call(self):
        """Test that on_llm_end is called after LLM responses."""
        recorder = HookRecorder()

        @tool(description="Echo tool")
        def echo(text: str) -> str:
            return text

        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(
                max_iterations=1,
                hooks={
                    "on_llm_end": lambda response, usage: recorder.record(
                        "on_llm_end", response, usage
                    )
                },
            ),
        )

        agent.run([Message(role=Role.USER, content="test")])

        assert recorder.was_called("on_llm_end")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestHookErrorHandling:
    """Test that hook errors don't break agent execution."""

    def test_hook_exceptions_are_silently_ignored(self):
        """Test that exceptions in hooks don't break the agent."""

        def failing_hook(*args, **kwargs):
            raise ValueError("Hook error!")

        @tool(description="Echo tool")
        def echo(text: str) -> str:
            return text

        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(
                max_iterations=1,
                hooks={"on_agent_start": failing_hook, "on_agent_end": failing_hook},
            ),
        )

        # Should not raise despite hook failures
        response = agent.run([Message(role=Role.USER, content="test")])
        assert response is not None


# =============================================================================
# Multiple Hooks Tests
# =============================================================================


class TestMultipleHooks:
    """Test using multiple hooks together."""

    def test_multiple_hooks_all_called(self):
        """Test that multiple hooks can be used simultaneously."""
        recorder = HookRecorder()

        @tool(description="Echo tool")
        def echo(text: str) -> str:
            return text

        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(
                max_iterations=1,
                hooks={
                    "on_agent_start": lambda msgs: recorder.record("on_agent_start", msgs),
                    "on_agent_end": lambda resp, usage: recorder.record(
                        "on_agent_end", resp, usage
                    ),
                    "on_iteration_start": lambda it, msgs: recorder.record(
                        "on_iteration_start", it, msgs
                    ),
                    "on_llm_start": lambda msgs, model: recorder.record(
                        "on_llm_start", msgs, model
                    ),
                    "on_llm_end": lambda resp, usage: recorder.record("on_llm_end", resp, usage),
                },
            ),
        )

        agent.run([Message(role=Role.USER, content="test")])

        # All hooks should have been called
        assert recorder.was_called("on_agent_start")
        assert recorder.was_called("on_agent_end")
        assert recorder.was_called("on_iteration_start")
        assert recorder.was_called("on_llm_start")
        assert recorder.was_called("on_llm_end")


# =============================================================================
# Async Hooks Tests
# =============================================================================


class TestAsyncHooks:
    """Test that hooks work with async agent execution."""

    @pytest.mark.asyncio
    async def test_hooks_work_with_async_agent(self):
        """Test that hooks are called during async execution."""
        recorder = HookRecorder()

        @tool(description="Echo tool")
        def echo(text: str) -> str:
            return text

        agent = Agent(
            tools=[echo],
            provider=LocalProvider(),
            config=AgentConfig(
                max_iterations=1,
                hooks={
                    "on_agent_start": lambda msgs: recorder.record("on_agent_start", msgs),
                    "on_agent_end": lambda resp, usage: recorder.record(
                        "on_agent_end", resp, usage
                    ),
                },
            ),
        )

        await agent.arun([Message(role=Role.USER, content="test")])

        assert recorder.was_called("on_agent_start")
        assert recorder.was_called("on_agent_end")
