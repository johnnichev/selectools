"""Tests for agent-level human-in-the-loop (ROADMAP P2: ToolConfig.require_approval)."""

import asyncio
import time
from typing import Any, List, Optional

import pytest

from selectools.agent.config import AgentConfig
from selectools.agent.config_groups import ToolConfig
from selectools.agent.core import Agent
from selectools.policy import ApprovalRequest, ToolPolicy
from selectools.tools.base import Tool
from selectools.types import Message, Role, ToolCall


def _make_tool(name: str = "send_email", result: str = "email sent", **kwargs: Any) -> Tool:
    return Tool(
        name=name,
        description=f"A tool named {name}",
        parameters=[],
        function=lambda: result,
        **kwargs,
    )


def _tool_call_msg(*tool_names: str) -> Message:
    return Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[
            ToolCall(tool_name=name, parameters={}, id=f"tc-{i}")
            for i, name in enumerate(tool_names)
        ],
    )


def _tool_messages(agent: Agent) -> List[Message]:
    return [m for m in agent._history if m.role == Role.TOOL]


class RecordingHandler:
    """Sync handler recording every ApprovalRequest it receives."""

    def __init__(self, decision: Any = True, raise_exc: Optional[Exception] = None) -> None:
        self.decision = decision
        self.raise_exc = raise_exc
        self.requests: List[ApprovalRequest] = []

    def __call__(self, request: ApprovalRequest) -> Any:
        self.requests.append(request)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.decision


def _agent(fake_provider, tools, tool_cfg, **config_kwargs) -> Agent:
    names = [t.name for t in tools]
    provider = fake_provider(responses=[_tool_call_msg(*names), "Done"])
    return Agent(
        tools=tools,
        provider=provider,
        config=AgentConfig(max_iterations=3, tool=tool_cfg, **config_kwargs),
    )


class TestConstructionValidation:
    def test_require_approval_without_handler_raises(self):
        with pytest.raises(ValueError, match="approval_handler"):
            ToolConfig(require_approval=["send_email"])

    def test_require_approval_star_without_handler_raises(self):
        with pytest.raises(ValueError, match="approval_handler"):
            ToolConfig(require_approval="*")

    def test_require_approval_with_handler_ok(self):
        cfg = ToolConfig(require_approval=["send_email"], approval_handler=lambda req: True)
        assert cfg.require_approval == ["send_email"]

    def test_require_approval_with_confirm_action_only_ok(self):
        """confirm_action is an accepted fallback approver."""
        cfg = ToolConfig(
            require_approval=["send_email"],
            confirm_action=lambda name, args, reason: True,
        )
        assert cfg.approval_handler is None

    def test_raises_via_agent_config_dict(self):
        with pytest.raises(ValueError, match="approval_handler"):
            AgentConfig(tool={"require_approval": ["send_email"]})

    def test_empty_list_is_not_gated(self):
        cfg = ToolConfig(require_approval=[])
        assert cfg.approval_handler is None


class TestGating:
    def test_gated_by_name_approved_executes(self, fake_provider):
        handler = RecordingHandler(decision=True)
        tool = _make_tool()
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(require_approval=["send_email"], approval_handler=handler),
        )
        result = agent.run("send the email")
        assert len(handler.requests) == 1
        req = handler.requests[0]
        assert req.tool_name == "send_email"
        assert req.tool_args == {}
        assert req.preview == "send_email()"
        assert "requires approval" in req.reason
        assert _tool_messages(agent)[0].content == "email sent"
        assert result.content == "Done"

    def test_ungated_tool_skips_handler(self, fake_provider):
        handler = RecordingHandler()
        tool = _make_tool(name="read_data", result="data")
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(require_approval=["send_email"], approval_handler=handler),
        )
        agent.run("read")
        assert handler.requests == []
        assert _tool_messages(agent)[0].content == "data"

    def test_star_gates_every_tool(self, fake_provider):
        handler = RecordingHandler(decision=True)
        a = _make_tool(name="tool_a", result="a")
        b = _make_tool(name="tool_b", result="b")
        agent = _agent(
            fake_provider,
            [a, b],
            ToolConfig(require_approval="*", approval_handler=handler),
            parallel_tool_execution=False,
        )
        agent.run("go")
        assert sorted(r.tool_name for r in handler.requests) == ["tool_a", "tool_b"]

    def test_deny_path_model_sees_denial(self, fake_provider):
        """Denied call returns a standardized result; the loop continues."""
        handler = RecordingHandler(decision=False)
        tool = _make_tool()
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(require_approval=["send_email"], approval_handler=handler),
        )
        result = agent.run("send the email")
        denial = _tool_messages(agent)[0].content
        assert denial == (
            "Tool 'send_email' denied by approval handler: Tool 'send_email' requires approval"
        )
        # The loop did not crash — the model saw the denial and answered.
        assert result.content == "Done"
        assert result.iterations == 2

    def test_handler_exception_denies_and_continues(self, fake_provider):
        handler = RecordingHandler(raise_exc=RuntimeError("pager down"))
        tool = _make_tool()
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(require_approval=["send_email"], approval_handler=handler),
        )
        result = agent.run("send")
        assert "approval failed: pager down" in _tool_messages(agent)[0].content
        assert result.content == "Done"

    def test_handler_timeout_denies(self, fake_provider):
        def slow_handler(request: ApprovalRequest) -> bool:
            time.sleep(0.5)
            return True

        tool = _make_tool()
        cfg = ToolConfig(
            require_approval=["send_email"],
            approval_handler=slow_handler,
            approval_timeout=0.05,
        )
        provider = fake_provider(responses=[_tool_call_msg("send_email"), "Done"])
        agent = Agent(
            tools=[tool],
            provider=provider,
            config=AgentConfig(max_iterations=3, tool=cfg),
        )
        result = agent.run("send")
        assert "approval timed out" in _tool_messages(agent)[0].content
        assert result.content == "Done"

    def test_preview_includes_args(self, fake_provider):
        handler = RecordingHandler(decision=True)
        tool = Tool(
            name="send_email",
            description="send",
            parameters=[],
            function=lambda **kw: "sent",
        )
        cfg = ToolConfig(require_approval=["send_email"], approval_handler=handler)
        provider = fake_provider(
            responses=[
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_name="send_email",
                            parameters={"to": "a@b.com", "subject": "hi"},
                            id="tc-0",
                        )
                    ],
                ),
                "Done",
            ]
        )
        agent = Agent(
            tools=[tool], provider=provider, config=AgentConfig(max_iterations=3, tool=cfg)
        )
        agent.run("send")
        assert handler.requests[0].preview == "send_email(to='a@b.com', subject='hi')"
        assert handler.requests[0].tool_args == {"to": "a@b.com", "subject": "hi"}


class TestAsyncBridging:
    def test_async_handler_from_sync_run(self, fake_provider):
        seen: List[ApprovalRequest] = []

        async def handler(request: ApprovalRequest) -> bool:
            await asyncio.sleep(0)
            seen.append(request)
            return request.tool_name == "send_email"

        tool = _make_tool()
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(require_approval=["send_email"], approval_handler=handler),
        )
        result = agent.run("send")
        assert len(seen) == 1
        assert _tool_messages(agent)[0].content == "email sent"
        assert result.content == "Done"

    def test_async_handler_from_arun(self, fake_provider):
        async def handler(request: ApprovalRequest) -> bool:
            await asyncio.sleep(0)
            return False

        tool = _make_tool()
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(require_approval=["send_email"], approval_handler=handler),
        )
        result = asyncio.run(agent.arun("send"))
        assert "denied by approval handler" in _tool_messages(agent)[0].content
        assert result.content == "Done"

    def test_sync_handler_from_arun(self, fake_provider):
        handler = RecordingHandler(decision=True)
        tool = _make_tool()
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(require_approval=["send_email"], approval_handler=handler),
        )
        result = asyncio.run(agent.arun("send"))
        assert len(handler.requests) == 1
        assert _tool_messages(agent)[0].content == "email sent"
        assert result.content == "Done"


class TestInteractionWithExistingMachinery:
    def test_per_tool_requires_approval_routes_to_handler(self, fake_provider):
        """Config OR tool gates → gated. Tool-level flag uses the handler too."""
        handler = RecordingHandler(decision=True)
        tool = _make_tool(requires_approval=True)
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(approval_handler=handler),
        )
        agent.run("send")
        assert len(handler.requests) == 1
        assert _tool_messages(agent)[0].content == "email sent"

    def test_handler_takes_precedence_over_confirm_action(self, fake_provider):
        confirm_calls: List[str] = []

        def confirm(name: str, args: dict, reason: str) -> bool:
            confirm_calls.append(name)
            return False

        handler = RecordingHandler(decision=True)
        tool = _make_tool()
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(
                require_approval=["send_email"],
                approval_handler=handler,
                confirm_action=confirm,
            ),
        )
        agent.run("send")
        assert len(handler.requests) == 1
        assert confirm_calls == []
        assert _tool_messages(agent)[0].content == "email sent"

    def test_falls_back_to_confirm_action_without_handler(self, fake_provider):
        confirm_calls: List[str] = []

        def confirm(name: str, args: dict, reason: str) -> bool:
            confirm_calls.append(name)
            return True

        tool = _make_tool()
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(require_approval=["send_email"], confirm_action=confirm),
        )
        agent.run("send")
        assert confirm_calls == ["send_email"]
        assert _tool_messages(agent)[0].content == "email sent"

    def test_policy_review_uses_handler(self, fake_provider):
        """A ToolPolicy review decision routes through approval_handler."""
        handler = RecordingHandler(decision=True)
        tool = _make_tool()
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(policy=ToolPolicy(review=["send_*"]), approval_handler=handler),
        )
        agent.run("send")
        assert len(handler.requests) == 1
        assert "review pattern" in handler.requests[0].reason
        assert _tool_messages(agent)[0].content == "email sent"

    def test_policy_deny_wins_over_handler(self, fake_provider):
        """deny is absolute — the handler is never consulted."""
        handler = RecordingHandler(decision=True)
        tool = _make_tool(name="delete_db", result="deleted")
        agent = _agent(
            fake_provider,
            [tool],
            ToolConfig(
                policy=ToolPolicy(deny=["delete_*"]),
                require_approval=["delete_db"],
                approval_handler=handler,
            ),
        )
        agent.run("delete")
        assert handler.requests == []
        assert "denied by policy" in _tool_messages(agent)[0].content

    def test_parallel_execution_gates_each_call(self, fake_provider):
        handler = RecordingHandler(decision=True)
        a = _make_tool(name="tool_a", result="a")
        b = _make_tool(name="tool_b", result="b")
        agent = _agent(
            fake_provider,
            [a, b],
            ToolConfig(require_approval="*", approval_handler=handler),
            parallel_tool_execution=True,
        )
        result = agent.run("go")
        assert sorted(r.tool_name for r in handler.requests) == ["tool_a", "tool_b"]
        assert result.content == "Done"

    def test_parallel_async_gates_each_call(self, fake_provider):
        async def handler(request: ApprovalRequest) -> bool:
            return request.tool_name != "tool_b"

        a = _make_tool(name="tool_a", result="a")
        b = _make_tool(name="tool_b", result="b")
        agent = _agent(
            fake_provider,
            [a, b],
            ToolConfig(require_approval="*", approval_handler=handler),
            parallel_tool_execution=True,
        )
        result = asyncio.run(agent.arun("go"))
        contents = {m.tool_name: m.content for m in _tool_messages(agent)}
        assert contents["tool_a"] == "a"
        assert "denied by approval handler" in contents["tool_b"]
        assert result.content == "Done"
