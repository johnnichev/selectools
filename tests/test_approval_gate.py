"""Tests for per-tool approval gate (R9)."""

from selectools.agent.config import AgentConfig
from selectools.agent.core import Agent
from selectools.policy import ToolPolicy
from selectools.tools.base import Tool
from selectools.tools.decorators import tool
from selectools.types import Message, Role, ToolCall


def _make_tool_with_approval(name="danger_tool", requires_approval=True):
    """Create a tool with requires_approval flag."""
    return Tool(
        name=name,
        description=f"A tool named {name}",
        parameters=[],
        function=lambda: "executed",
        requires_approval=requires_approval,
    )


def _tool_call_msg(tool_name):
    """Create a Message with a tool call."""
    return Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(tool_name=tool_name, parameters={})],
    )


class TestApprovalGate:
    """R9: Tools with requires_approval=True trigger REVIEW."""

    def test_decorator_requires_approval(self):
        """@tool(requires_approval=True) sets the flag."""

        @tool(requires_approval=True, description="Send email")
        def send_email() -> str:
            return "sent"

        assert send_email.requires_approval is True

    def test_decorator_default_no_approval(self):
        """@tool() defaults to requires_approval=False."""

        @tool(description="Read data")
        def read_data() -> str:
            return "data"

        assert read_data.requires_approval is False

    def test_tool_class_requires_approval(self):
        """Tool(requires_approval=True) sets the attribute."""
        t = _make_tool_with_approval()
        assert t.requires_approval is True

    def test_approval_triggers_review_without_policy(self, fake_provider):
        """Tool with requires_approval=True is denied when no confirm_action."""
        danger = _make_tool_with_approval()
        provider = fake_provider(responses=[_tool_call_msg("danger_tool"), "Done"])
        agent = Agent(
            tools=[danger],
            provider=provider,
            config=AgentConfig(max_iterations=3),
        )
        result = agent.run("do something dangerous")
        # Without confirm_action, the tool should be denied
        assert result.iterations <= 3

    def test_approval_with_confirm_action_approved(self, fake_provider):
        """Tool is executed when confirm_action returns True."""
        danger = _make_tool_with_approval()
        provider = fake_provider(responses=[_tool_call_msg("danger_tool"), "Done after approval"])
        agent = Agent(
            tools=[danger],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                confirm_action=lambda name, args, reason: True,
            ),
        )
        result = agent.run("do something")
        assert result.iterations >= 1

    def test_approval_with_confirm_action_denied(self, fake_provider):
        """Tool is not executed when confirm_action returns False."""
        danger = _make_tool_with_approval()
        provider = fake_provider(responses=[_tool_call_msg("danger_tool"), "Done"])
        agent = Agent(
            tools=[danger],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                confirm_action=lambda name, args, reason: False,
            ),
        )
        result = agent.run("do something")
        assert result.iterations <= 3

    def test_no_approval_flag_passes_through(self, fake_provider):
        """Tool without requires_approval passes through normally."""
        safe = Tool(
            name="safe_tool",
            description="Safe operation",
            parameters=[],
            function=lambda: "safe result",
        )
        provider = fake_provider(responses=[_tool_call_msg("safe_tool"), "All good"])
        agent = Agent(
            tools=[safe],
            provider=provider,
            config=AgentConfig(max_iterations=3),
        )
        result = agent.run("do something safe")
        assert result.iterations >= 1

    def test_policy_deny_overrides_requires_approval(self, fake_provider):
        """ToolPolicy deny overrides requires_approval (deny always wins)."""
        danger = _make_tool_with_approval()
        provider = fake_provider(responses=[_tool_call_msg("danger_tool"), "Done"])
        agent = Agent(
            tools=[danger],
            provider=provider,
            config=AgentConfig(
                max_iterations=3,
                tool_policy=ToolPolicy(deny=["danger_*"]),
                confirm_action=lambda name, args, reason: True,
            ),
        )
        result = agent.run("do something")
        # Tool should be denied by policy regardless of approval
        assert result.iterations <= 3
