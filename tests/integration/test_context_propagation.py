import asyncio
import contextvars
from typing import Any

import pytest

from selectools import Agent, AgentConfig, Tool
from selectools.providers.stubs import LocalProvider
from selectools.types import Message, Role, ToolCall
from selectools.usage import UsageStats

# Define a context variable to simulate a trace ID or auth token
request_id_var = contextvars.ContextVar("request_id", default="empty")


def get_context_sync() -> str:
    """Sync tool that reads the context variable."""
    return f"sync:{request_id_var.get()}"


async def get_context_async() -> str:
    """Async tool that reads the context variable."""
    # Ensure we yield to loop to test async propagation
    await asyncio.sleep(0.01)
    return f"async:{request_id_var.get()}"


@pytest.mark.asyncio
async def test_context_propagation_to_tools():
    """
    Verify that context variables set in the parent task are propagated
    to both synchronous (threaded) and asynchronous tools.
    """
    # 1. Set context in the parent task
    token = request_id_var.set("test-123")

    try:
        provider = LocalProvider()

        # Create tools manually to avoid decorator overhead/issues during test setup
        tools = [
            Tool(
                name="get_context_sync",
                description="Get context sync",
                function=get_context_sync,
                parameters={},
            ),
            Tool(
                name="get_context_async",
                description="Get context async",
                function=get_context_async,
                parameters={},
            ),
        ]

        agent = Agent(config=AgentConfig(), provider=provider, tools=tools)

        # 2. Run agent which calls the tools
        # We need to trick the LocalProvider or Agent to call our tools.
        # Since LocalProvider just echoes, we can manually inject a "ToolCall"
        # into the history processing, OR we can rely on the Agent's ability
        # to execute tools if the provider *tells* it to.
        #
        # A better way for this unit test might be to mock the provider response
        # to force a tool call.

        # Let's mock the provider response to return a tool call

        # Valid tool call messages
        sync_call_msg = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(tool_name="get_context_sync", parameters={}, id="call_1")],
            content="",
        )
        async_call_msg = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(tool_name="get_context_async", parameters={}, id="call_2")],
            content="",
        )

        # We also need a final response to stop the loop
        final_msg = Message(role=Role.ASSISTANT, content="Done")

        # Mock the provider's acomplete to return these in sequence
        # We can't easily mock the instance method of a fresh provider without a helper class
        # or `unittest.mock`. Let's use a simple mock class.

        class MockToolProvider(LocalProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0
                self.supports_async = True

            async def acomplete(self, **kwargs):
                self.call_count += 1
                dummy_usage = UsageStats(0, 0, 0, 0.0, "mock", "mock")
                if self.call_count == 1:
                    return sync_call_msg, dummy_usage
                elif self.call_count == 2:
                    return async_call_msg, dummy_usage
                else:
                    return final_msg, dummy_usage

        agent = Agent(
            config=AgentConfig(max_iterations=5), provider=MockToolProvider(), tools=tools
        )

        # 3. Execute agent
        result = await agent.arun([Message(role=Role.USER, content="start")])

        # 4. Verify results in history
        # Check agent history directly
        history = agent._history
        tool_msgs = [m for m in history if m.role == Role.TOOL]

        # Let's inspect the agent's history directly if possible, or reliance on `on_tool_end` hook?
        # Actually `Agent` exposes `memory` but we didn't pass one.
        # `agent._history` is available after run.

        history = agent._history
        tool_msgs = [m for m in history if m.role == Role.TOOL]

        assert len(tool_msgs) >= 2

        # Find results
        sync_result = next(m.content for m in tool_msgs if m.tool_name == "get_context_sync")
        async_result = next(m.content for m in tool_msgs if m.tool_name == "get_context_async")

        print(f"Sync Result: {sync_result}")
        print(f"Async Result: {async_result}")

        assert sync_result == "sync:test-123"
        assert async_result == "async:test-123"

    finally:
        request_id_var.reset(token)


if __name__ == "__main__":
    # Allow running directly
    asyncio.run(test_context_propagation_to_tools())
