from unittest.mock import Mock

import pytest

from selectools import Agent, AgentConfig, AgentResult, Message, Role, Tool, ToolCall
from selectools.providers.stubs import LocalProvider
from selectools.usage import UsageStats


def dummy_tool(arg: str) -> str:
    """A dummy tool that should not be called in routing mode."""
    return f"Executed: {arg}"


class RoutingMockProvider(LocalProvider):
    supports_async = True

    def __init__(self, tool_to_call: str, args: dict):
        super().__init__()
        self.tool_to_call = tool_to_call
        self.args = args

    def complete(self, **kwargs):
        msg = Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(tool_name=self.tool_to_call, parameters=self.args, id="call_1")],
            content="",
        )
        return msg, UsageStats(0, 0, 0, 0.0, "mock", "mock")

    async def acomplete(self, **kwargs):
        return self.complete(**kwargs)


def test_routing_mode_sync():
    """Verify sync run returns tool call without execution."""
    tool = Tool(
        name="test_tool", description="Test tool", function=Mock(wraps=dummy_tool), parameters={}
    )

    provider = RoutingMockProvider("test_tool", {"arg": "val"})
    config = AgentConfig(routing_only=True)
    agent = Agent(tools=[tool], provider=provider, config=config)

    result = agent.run([Message(role=Role.USER, content="hi")])

    # Assertions
    assert isinstance(result, AgentResult)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "test_tool"
    assert result.tool_calls[0].parameters == {"arg": "val"}

    # Verify tool was NOT executed
    tool.function.assert_not_called()

    # Verify history does NOT contain the tool result
    # The last message in history should be the prompt, NOT the assistant response
    # (based on the implementation, we don't append until execution loop continues)
    # Actually looking at the implementation:
    # `self._history.append(response_msg)` happens AFTER the routing check block.
    # So history should only contain the initial user message.
    assert len(agent._history) == 1
    assert agent._history[0].role == Role.USER

    # Verify memory (if we had one)


@pytest.mark.asyncio
async def test_routing_mode_async():
    """Verify async run returns tool call without execution."""
    tool = Tool(
        name="test_tool", description="Test tool", function=Mock(wraps=dummy_tool), parameters={}
    )

    provider = RoutingMockProvider("test_tool", {"arg": "val"})
    config = AgentConfig(routing_only=True)
    agent = Agent(tools=[tool], provider=provider, config=config)

    result = await agent.arun([Message(role=Role.USER, content="hi")])

    # Assertions
    assert isinstance(result, AgentResult)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "test_tool"

    # Verify tool was NOT executed
    tool.function.assert_not_called()

    # Verify history
    assert len(agent._history) == 1


class MockProviderMultiple(LocalProvider):
    supports_async = True

    def __init__(self, tool_calls: list):
        super().__init__()
        self.tool_calls = tool_calls

    def complete(self, **kwargs):
        msg = Message(
            role=Role.ASSISTANT,
            tool_calls=self.tool_calls,
            content="",
        )
        return msg, UsageStats(0, 0, 0, 0.0, "mock", "mock")

    async def acomplete(self, **kwargs):
        return self.complete(**kwargs)


class MockProviderText(LocalProvider):
    supports_async = True

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def complete(self, **kwargs):
        msg = Message(role=Role.ASSISTANT, content=self.text)
        return msg, UsageStats(0, 0, 0, 0.0, "mock", "mock")

    async def acomplete(self, **kwargs):
        return self.complete(**kwargs)


def test_routing_mode_text_only():
    """Verify routing mode behaves normally for text-only responses."""
    tool = Tool(
        name="test_tool", description="Test tool", function=Mock(wraps=dummy_tool), parameters={}
    )

    # Provider returns text, no tool calls
    provider = MockProviderText("Just chatting")
    config = AgentConfig(routing_only=True)
    agent = Agent(tools=[tool], provider=provider, config=config)

    result = agent.run([Message(role=Role.USER, content="hi")])

    # Assertions
    assert isinstance(result, AgentResult)
    assert result.content == "Just chatting"
    assert not result.tool_calls

    # History SHOULD be updated normally for text responses... wait, actually Agent does NOT append final response to _history if no memory!
    # So length should be 1 (just the user message)
    # This behavior is consistent with standard agent config (no memory = transient history of inputs)
    assert len(agent._history) == 1
    # Check return object for the response
    assert result.message.role == Role.ASSISTANT
    assert result.message.content == "Just chatting"


def test_routing_mode_multiple_tools():
    """Verify routing mode returns all selected tools."""
    tool1 = Tool(name="tool1", description="T1", function=Mock(), parameters={})
    tool2 = Tool(name="tool2", description="T2", function=Mock(), parameters={})

    calls = [
        ToolCall(tool_name="tool1", parameters={"a": 1}, id="1"),
        ToolCall(tool_name="tool2", parameters={"b": 2}, id="2"),
    ]
    provider = MockProviderMultiple(calls)
    config = AgentConfig(routing_only=True)
    agent = Agent(tools=[tool1, tool2], provider=provider, config=config)

    result = agent.run([Message(role=Role.USER, content="do both")])

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "tool1"
    assert result.tool_calls[1].tool_name == "tool2"

    # Verify no execution
    tool1.function.assert_not_called()
    tool2.function.assert_not_called()
