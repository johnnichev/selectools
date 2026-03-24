"""Regression tests for bug hunt batch 1 — tools, RAG, evals fixes."""

import asyncio
import json
import re

import pytest

from selectools.evals.evaluators import JsonValidityEvaluator, OutputEvaluator
from selectools.evals.types import CaseResult, TestCase
from selectools.tools.base import Tool, ToolParameter


class TestAsyncToolSync:
    """Bug #6: execute() on async tools must await, not stringify."""

    def test_async_tool_sync_execute(self):
        async def async_func() -> str:
            return "async result"

        tool = Tool(name="async_tool", description="async", parameters=[], function=async_func)
        result = tool.execute({})
        assert result == "async result"
        assert "coroutine" not in result


class TestExecutorPerCall:
    """Bug #5: aexecute() should use shared executor."""

    @pytest.mark.asyncio
    async def test_aexecute_works(self):
        def sync_func() -> str:
            return "sync result"

        tool = Tool(name="sync_tool", description="sync", parameters=[], function=sync_func)
        result = await tool.aexecute({})
        assert result == "sync result"


class TestOutputEvaluatorRegex:
    """Bug #11: Invalid regex should not crash."""

    def test_invalid_regex_returns_failure(self):
        evaluator = OutputEvaluator()
        case = TestCase(input="test", expect_output_regex="[unclosed")
        from selectools.types import AgentResult, Message, Role

        agent_result = AgentResult(
            message=Message(role=Role.ASSISTANT, content="some output"),
            iterations=1,
            tool_calls=[],
        )
        case_result = CaseResult(
            case=case, agent_result=agent_result, verdict="pass", latency_ms=100, failures=[]
        )
        failures = evaluator.check(case, case_result)
        assert len(failures) == 1
        assert "Invalid regex" in failures[0].message


class TestJsonValidityFalse:
    """Bug #12: expect_json=False should skip validation."""

    def test_expect_json_false_skips(self):
        evaluator = JsonValidityEvaluator()
        case = TestCase(input="test", expect_json=False)
        from selectools.types import AgentResult, Message, Role

        agent_result = AgentResult(
            message=Message(role=Role.ASSISTANT, content="not json"),
            iterations=1,
            tool_calls=[],
        )
        case_result = CaseResult(
            case=case, agent_result=agent_result, verdict="pass", latency_ms=100, failures=[]
        )
        failures = evaluator.check(case, case_result)
        assert len(failures) == 0  # Should not fail
