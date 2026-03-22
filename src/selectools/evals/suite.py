"""EvalSuite — orchestrates running test cases against an agent."""

from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from .. import __version__
from ..agent import Agent
from .evaluators import DEFAULT_EVALUATORS
from .report import EvalReport
from .types import CaseResult, CaseVerdict, EvalFailure, EvalMetadata, TestCase


class EvalSuite:
    """Evaluate an agent against a list of test cases.

    Args:
        agent: The Agent instance to evaluate.
        cases: List of TestCase objects.
        name: Human-readable name for this suite.
        evaluators: Custom evaluator chain. Default: all built-in evaluators.
        max_concurrency: Max parallel cases. Default: 1 (sequential).
        on_progress: Optional (completed, total) callback.
        tags: Arbitrary metadata tags for this suite run.
    """

    def __init__(
        self,
        agent: Agent,
        cases: List[TestCase],
        *,
        name: str = "eval",
        evaluators: Optional[List[Any]] = None,
        max_concurrency: int = 1,
        on_progress: Optional[Callable[[int, int], None]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.agent = agent
        self.cases = cases
        self.name = name
        self.evaluators = evaluators if evaluators is not None else list(DEFAULT_EVALUATORS)
        self.max_concurrency = max_concurrency
        self.on_progress = on_progress
        self.tags = tags or {}

    def run(self) -> EvalReport:
        """Run all cases synchronously and return an EvalReport."""
        start = time.perf_counter()
        run_id = uuid.uuid4().hex[:12]

        if self.max_concurrency <= 1:
            results = []
            for i, case in enumerate(self.cases):
                results.append(self._run_case(case))
                if self.on_progress:
                    self.on_progress(i + 1, len(self.cases))
        else:
            completed = 0

            def _run_and_report(case: TestCase) -> CaseResult:
                nonlocal completed
                result = self._run_case(case)
                completed += 1
                if self.on_progress:
                    self.on_progress(completed, len(self.cases))
                return result

            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                results = list(pool.map(_run_and_report, self.cases))

        duration_ms = (time.perf_counter() - start) * 1000
        return self._build_report(results, run_id, duration_ms)

    async def arun(self) -> EvalReport:
        """Async version of run()."""
        start = time.perf_counter()
        run_id = uuid.uuid4().hex[:12]
        sem = asyncio.Semaphore(max(self.max_concurrency, 1))
        completed = 0

        async def _run_with_sem(case: TestCase) -> CaseResult:
            nonlocal completed
            async with sem:
                result = await self._arun_case(case)
                completed += 1
                if self.on_progress:
                    self.on_progress(completed, len(self.cases))
                return result

        tasks = [_run_with_sem(case) for case in self.cases]
        results = await asyncio.gather(*tasks)

        duration_ms = (time.perf_counter() - start) * 1000
        return self._build_report(list(results), run_id, duration_ms)

    def _run_case(self, case: TestCase) -> CaseResult:
        """Execute a single test case synchronously and evaluate it."""
        clone = self.agent._clone_for_isolation()
        start = time.perf_counter()

        try:
            result = clone.run(case.input, response_format=case.response_format)
        except Exception as exc:
            return CaseResult(
                case=case,
                verdict=CaseVerdict.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        return self._evaluate(case, result, start)

    async def _arun_case(self, case: TestCase) -> CaseResult:
        """Execute a single test case asynchronously and evaluate it."""
        clone = self.agent._clone_for_isolation()
        start = time.perf_counter()

        try:
            result = await clone.arun(case.input, response_format=case.response_format)
        except Exception as exc:
            return CaseResult(
                case=case,
                verdict=CaseVerdict.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        return self._evaluate(case, result, start)

    def _evaluate(self, case: TestCase, result: Any, start: float) -> CaseResult:
        """Run all evaluators against a case result."""
        latency_ms = (time.perf_counter() - start) * 1000
        cost_usd = 0.0
        tokens = 0
        if result.usage:
            cost_usd = result.usage.total_cost_usd
            tokens = result.usage.total_tokens

        tool_names = [tc.tool_name for tc in result.tool_calls]

        case_result = CaseResult(
            case=case,
            verdict=CaseVerdict.PASS,
            agent_result=result,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            tokens=tokens,
            tool_calls=tool_names,
        )

        all_failures: List[EvalFailure] = []
        for evaluator in self.evaluators:
            failures = evaluator.check(case, case_result)
            all_failures.extend(failures)

        case_result.failures = all_failures
        if all_failures:
            case_result.verdict = CaseVerdict.FAIL

        return case_result

    def _build_report(
        self,
        case_results: List[CaseResult],
        run_id: str,
        duration_ms: float,
    ) -> EvalReport:
        """Aggregate case results into an EvalReport."""
        model = getattr(self.agent, "_model", "") or ""
        provider = ""
        if hasattr(self.agent, "provider") and self.agent.provider:
            provider = type(self.agent.provider).__name__

        metadata = EvalMetadata(
            suite_name=self.name,
            model=model,
            provider=provider,
            timestamp=time.time(),
            run_id=run_id,
            total_cases=len(case_results),
            duration_ms=duration_ms,
            selectools_version=__version__,
            tags=self.tags,
        )

        return EvalReport(metadata=metadata, case_results=case_results)
