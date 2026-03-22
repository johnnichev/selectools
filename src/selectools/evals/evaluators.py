"""Built-in evaluators for the eval framework."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .types import CaseResult, EvalFailure, TestCase


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for eval evaluators.

    Evaluators inspect a TestCase and its CaseResult and return
    a list of failures. An empty list means the check passed.
    """

    name: str

    def check(self, case: TestCase, case_result: CaseResult) -> List[EvalFailure]: ...


class ToolUseEvaluator:
    """Checks expect_tool, expect_tools, and expect_tool_args."""

    name: str = "tool_use"

    def check(self, case: TestCase, case_result: CaseResult) -> List[EvalFailure]:
        failures: List[EvalFailure] = []
        called = case_result.tool_calls

        if case.expect_tool is not None:
            if case.expect_tool not in called:
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected=case.expect_tool,
                        actual=called,
                        message=f"Expected tool '{case.expect_tool}' to be called, "
                        f"but got: {called}",
                    )
                )

        if case.expect_tools is not None:
            missing = [t for t in case.expect_tools if t not in called]
            if missing:
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected=case.expect_tools,
                        actual=called,
                        message=f"Expected tools {missing} to be called, " f"but got: {called}",
                    )
                )

        if case.expect_tool_args is not None and case_result.agent_result is not None:
            result_tool_calls = case_result.agent_result.tool_calls
            for tool_name, expected_args in case.expect_tool_args.items():
                matching = [tc for tc in result_tool_calls if tc.tool_name == tool_name]
                if not matching:
                    failures.append(
                        EvalFailure(
                            evaluator_name=self.name,
                            expected={tool_name: expected_args},
                            actual=None,
                            message=f"Tool '{tool_name}' was not called",
                        )
                    )
                    continue
                last_call = matching[-1]
                for arg_name, expected_val in expected_args.items():
                    actual_val = last_call.parameters.get(arg_name)
                    if actual_val != expected_val:
                        failures.append(
                            EvalFailure(
                                evaluator_name=self.name,
                                expected={arg_name: expected_val},
                                actual={arg_name: actual_val},
                                message=f"Tool '{tool_name}' arg '{arg_name}': "
                                f"expected {expected_val!r}, got {actual_val!r}",
                            )
                        )

        return failures


class ContainsEvaluator:
    """Checks expect_contains and expect_not_contains."""

    name: str = "contains"

    def check(self, case: TestCase, case_result: CaseResult) -> List[EvalFailure]:
        failures: List[EvalFailure] = []
        if case_result.agent_result is None:
            return failures

        content = case_result.agent_result.content or ""
        content_lower = content.lower()

        if case.expect_contains is not None:
            if case.expect_contains.lower() not in content_lower:
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected=case.expect_contains,
                        actual=content[:200],
                        message=f"Response does not contain '{case.expect_contains}'",
                    )
                )

        if case.expect_not_contains is not None:
            if case.expect_not_contains.lower() in content_lower:
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected=f"NOT '{case.expect_not_contains}'",
                        actual=content[:200],
                        message=f"Response contains '{case.expect_not_contains}' "
                        f"but should not",
                    )
                )

        return failures


class OutputEvaluator:
    """Checks expect_output (exact match) and expect_output_regex."""

    name: str = "output"

    def check(self, case: TestCase, case_result: CaseResult) -> List[EvalFailure]:
        failures: List[EvalFailure] = []
        if case_result.agent_result is None:
            return failures

        content = case_result.agent_result.content or ""

        if case.expect_output is not None:
            if content.strip() != case.expect_output.strip():
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected=case.expect_output[:200],
                        actual=content[:200],
                        message="Response does not match expected output",
                    )
                )

        if case.expect_output_regex is not None:
            if not re.search(case.expect_output_regex, content):
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected=case.expect_output_regex,
                        actual=content[:200],
                        message=f"Response does not match regex " f"'{case.expect_output_regex}'",
                    )
                )

        return failures


class StructuredOutputEvaluator:
    """Checks expect_parsed against result.parsed (deep subset match)."""

    name: str = "structured_output"

    def check(self, case: TestCase, case_result: CaseResult) -> List[EvalFailure]:
        failures: List[EvalFailure] = []
        if case.expect_parsed is None or case_result.agent_result is None:
            return failures

        parsed = case_result.agent_result.parsed
        if parsed is None:
            failures.append(
                EvalFailure(
                    evaluator_name=self.name,
                    expected=case.expect_parsed,
                    actual=None,
                    message="No parsed output — result.parsed is None",
                )
            )
            return failures

        parsed_dict = parsed if isinstance(parsed, dict) else vars(parsed)
        for key, expected_val in case.expect_parsed.items():
            actual_val = parsed_dict.get(key)
            if actual_val != expected_val:
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected={key: expected_val},
                        actual={key: actual_val},
                        message=f"Parsed field '{key}': expected {expected_val!r}, "
                        f"got {actual_val!r}",
                    )
                )

        return failures


class PerformanceEvaluator:
    """Checks iteration count, latency, and cost thresholds."""

    name: str = "performance"

    def check(self, case: TestCase, case_result: CaseResult) -> List[EvalFailure]:
        failures: List[EvalFailure] = []

        if case.expect_iterations_lte is not None and case_result.agent_result is not None:
            actual = case_result.agent_result.iterations
            if actual > case.expect_iterations_lte:
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected=f"<= {case.expect_iterations_lte} iterations",
                        actual=actual,
                        message=f"Agent used {actual} iterations "
                        f"(limit: {case.expect_iterations_lte})",
                    )
                )

        if case.expect_latency_ms_lte is not None:
            if case_result.latency_ms > case.expect_latency_ms_lte:
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected=f"<= {case.expect_latency_ms_lte}ms",
                        actual=f"{case_result.latency_ms:.1f}ms",
                        message=f"Latency {case_result.latency_ms:.1f}ms exceeds "
                        f"limit {case.expect_latency_ms_lte}ms",
                    )
                )

        if case.expect_cost_usd_lte is not None:
            if case_result.cost_usd > case.expect_cost_usd_lte:
                failures.append(
                    EvalFailure(
                        evaluator_name=self.name,
                        expected=f"<= ${case.expect_cost_usd_lte}",
                        actual=f"${case_result.cost_usd:.6f}",
                        message=f"Cost ${case_result.cost_usd:.6f} exceeds "
                        f"limit ${case.expect_cost_usd_lte}",
                    )
                )

        return failures


class CustomEvaluator:
    """Runs the case's custom_evaluator callable."""

    name: str = "custom"

    def check(self, case: TestCase, case_result: CaseResult) -> List[EvalFailure]:
        failures: List[EvalFailure] = []
        if case.custom_evaluator is None or case_result.agent_result is None:
            return failures

        eval_name = case.custom_evaluator_name or "custom_evaluator"
        try:
            passed = case.custom_evaluator(case_result.agent_result)
        except Exception as exc:
            failures.append(
                EvalFailure(
                    evaluator_name=eval_name,
                    expected="custom evaluator to pass",
                    actual=f"{type(exc).__name__}: {exc}",
                    message=f"Custom evaluator '{eval_name}' raised: {exc}",
                )
            )
            return failures

        if not passed:
            failures.append(
                EvalFailure(
                    evaluator_name=eval_name,
                    expected="True",
                    actual="False",
                    message=f"Custom evaluator '{eval_name}' returned False",
                )
            )

        return failures


DEFAULT_EVALUATORS: List[Any] = [
    ToolUseEvaluator(),
    ContainsEvaluator(),
    OutputEvaluator(),
    StructuredOutputEvaluator(),
    PerformanceEvaluator(),
    CustomEvaluator(),
]
