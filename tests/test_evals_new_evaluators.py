"""Tests for the 18 new evaluators (9 deterministic + 9 LLM-as-judge)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from selectools.evals import (
    ConcisenessEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    GrammarEvaluator,
    InstructionFollowingEvaluator,
    MarkdownFormatEvaluator,
    PythonValidityEvaluator,
    RefusalEvaluator,
    SafetyEvaluator,
    SentimentEvaluator,
    SQLValidityEvaluator,
    TestCase,
    ToneEvaluator,
    ToolOrderEvaluator,
    UniqueToolsEvaluator,
    URLValidityEvaluator,
    WordCountEvaluator,
)
from selectools.evals.types import CaseResult, CaseVerdict, EvalFailure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_result(content: str = "hello", **kwargs: Any) -> MagicMock:
    result = MagicMock()
    result.content = content
    result.tool_calls = kwargs.get("tool_calls", [])
    result.iterations = kwargs.get("iterations", 1)
    result.parsed = kwargs.get("parsed", None)
    result.reasoning = kwargs.get("reasoning", None)
    usage = MagicMock()
    usage.total_cost_usd = kwargs.get("cost_usd", 0.001)
    usage.total_tokens = kwargs.get("total_tokens", 100)
    result.usage = usage
    return result


def _cr(case: TestCase, content: str = "hello", **kwargs: Any) -> CaseResult:
    return CaseResult(
        case=case,
        verdict=CaseVerdict.PASS,
        agent_result=_make_agent_result(content, **kwargs),
        tool_calls=kwargs.get("tool_calls_names", []),
    )


def _mock_provider(score: int) -> MagicMock:
    provider = MagicMock()
    msg = MagicMock()
    msg.content = f"Analysis. Score: {score}"
    provider.complete = MagicMock(return_value=(msg, MagicMock()))
    return provider


# ===========================================================================
# WordCountEvaluator
# ===========================================================================


class TestWordCountEvaluator:
    def setup_method(self) -> None:
        self.e = WordCountEvaluator()

    def test_pass_min(self) -> None:
        case = TestCase(input="x", expect_min_words=3)
        assert self.e.check(case, _cr(case, "one two three four")) == []

    def test_fail_min(self) -> None:
        case = TestCase(input="x", expect_min_words=10)
        assert len(self.e.check(case, _cr(case, "short"))) == 1

    def test_pass_max(self) -> None:
        case = TestCase(input="x", expect_max_words=5)
        assert self.e.check(case, _cr(case, "one two")) == []

    def test_fail_max(self) -> None:
        case = TestCase(input="x", expect_max_words=2)
        assert len(self.e.check(case, _cr(case, "one two three four"))) == 1

    def test_both_bounds(self) -> None:
        case = TestCase(input="x", expect_min_words=2, expect_max_words=5)
        assert self.e.check(case, _cr(case, "hello world today")) == []

    def test_skip(self) -> None:
        case = TestCase(input="x")
        assert self.e.check(case, _cr(case, "anything")) == []


# ===========================================================================
# ToolOrderEvaluator
# ===========================================================================


class TestToolOrderEvaluator:
    def setup_method(self) -> None:
        self.e = ToolOrderEvaluator()

    def test_pass(self) -> None:
        case = TestCase(input="x", expect_tool_order=["search", "summarize"])
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, tool_calls=["search", "summarize"])
        assert self.e.check(case, cr) == []

    def test_fail_wrong_order(self) -> None:
        case = TestCase(input="x", expect_tool_order=["search", "summarize"])
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, tool_calls=["summarize", "search"])
        assert len(self.e.check(case, cr)) == 1

    def test_skip(self) -> None:
        case = TestCase(input="x")
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, tool_calls=["any"])
        assert self.e.check(case, cr) == []


# ===========================================================================
# UniqueToolsEvaluator
# ===========================================================================


class TestUniqueToolsEvaluator:
    def setup_method(self) -> None:
        self.e = UniqueToolsEvaluator()

    def test_pass(self) -> None:
        case = TestCase(input="x", expect_unique_tools=True)
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, tool_calls=["a", "b", "c"])
        assert self.e.check(case, cr) == []

    def test_fail_duplicates(self) -> None:
        case = TestCase(input="x", expect_unique_tools=True)
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, tool_calls=["a", "b", "a"])
        assert len(self.e.check(case, cr)) == 1

    def test_skip(self) -> None:
        case = TestCase(input="x")
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, tool_calls=["a", "a"])
        assert self.e.check(case, cr) == []


# ===========================================================================
# RefusalEvaluator
# ===========================================================================


class TestRefusalEvaluator:
    def setup_method(self) -> None:
        self.e = RefusalEvaluator()

    def test_pass_refused(self) -> None:
        case = TestCase(input="x", expect_refusal=True)
        assert self.e.check(case, _cr(case, "I'm sorry, I cannot help with that.")) == []

    def test_pass_cannot(self) -> None:
        case = TestCase(input="x", expect_refusal=True)
        assert self.e.check(case, _cr(case, "I can't assist with this request.")) == []

    def test_fail_no_refusal(self) -> None:
        case = TestCase(input="x", expect_refusal=True)
        assert len(self.e.check(case, _cr(case, "Sure, here's the answer!"))) == 1

    def test_skip(self) -> None:
        case = TestCase(input="x")
        assert self.e.check(case, _cr(case, "anything")) == []


# ===========================================================================
# SentimentEvaluator
# ===========================================================================


class TestSentimentEvaluator:
    def setup_method(self) -> None:
        self.e = SentimentEvaluator()

    def test_positive(self) -> None:
        case = TestCase(input="x", expect_sentiment="positive")
        assert self.e.check(case, _cr(case, "This is amazing and wonderful!")) == []

    def test_negative(self) -> None:
        case = TestCase(input="x", expect_sentiment="negative")
        assert self.e.check(case, _cr(case, "This is terrible and awful.")) == []

    def test_neutral(self) -> None:
        case = TestCase(input="x", expect_sentiment="neutral")
        assert self.e.check(case, _cr(case, "The temperature is 72 degrees.")) == []

    def test_fail_mismatch(self) -> None:
        case = TestCase(input="x", expect_sentiment="positive")
        assert len(self.e.check(case, _cr(case, "This is terrible and broken."))) == 1

    def test_skip(self) -> None:
        case = TestCase(input="x")
        assert self.e.check(case, _cr(case, "anything")) == []


# ===========================================================================
# PythonValidityEvaluator
# ===========================================================================


class TestPythonValidityEvaluator:
    def setup_method(self) -> None:
        self.e = PythonValidityEvaluator()

    def test_pass_valid(self) -> None:
        case = TestCase(input="x", expect_valid_python=True)
        assert self.e.check(case, _cr(case, "x = 1\nprint(x)")) == []

    def test_pass_with_fences(self) -> None:
        case = TestCase(input="x", expect_valid_python=True)
        code = "```python\ndef hello():\n    return 'hi'\n```"
        assert self.e.check(case, _cr(case, code)) == []

    def test_fail_syntax_error(self) -> None:
        case = TestCase(input="x", expect_valid_python=True)
        assert len(self.e.check(case, _cr(case, "def bad(:\n  pass"))) == 1

    def test_skip(self) -> None:
        case = TestCase(input="x")
        assert self.e.check(case, _cr(case, "not python")) == []


# ===========================================================================
# SQLValidityEvaluator
# ===========================================================================


class TestSQLValidityEvaluator:
    def setup_method(self) -> None:
        self.e = SQLValidityEvaluator()

    def test_pass_select(self) -> None:
        case = TestCase(input="x", expect_valid_sql=True)
        assert self.e.check(case, _cr(case, "SELECT * FROM users WHERE id = 1")) == []

    def test_pass_with_fences(self) -> None:
        case = TestCase(input="x", expect_valid_sql=True)
        assert self.e.check(case, _cr(case, "```sql\nSELECT * FROM users\n```")) == []

    def test_fail_not_sql(self) -> None:
        case = TestCase(input="x", expect_valid_sql=True)
        assert len(self.e.check(case, _cr(case, "This is not SQL"))) == 1

    def test_fail_unbalanced_parens(self) -> None:
        case = TestCase(input="x", expect_valid_sql=True)
        assert len(self.e.check(case, _cr(case, "SELECT * FROM (users"))) == 1

    def test_skip(self) -> None:
        case = TestCase(input="x")
        assert self.e.check(case, _cr(case, "anything")) == []


# ===========================================================================
# URLValidityEvaluator
# ===========================================================================


class TestURLValidityEvaluator:
    def setup_method(self) -> None:
        self.e = URLValidityEvaluator()

    def test_pass_valid_urls(self) -> None:
        case = TestCase(input="x", expect_valid_urls=True)
        assert self.e.check(case, _cr(case, "Visit https://example.com for more.")) == []

    def test_pass_no_urls(self) -> None:
        case = TestCase(input="x", expect_valid_urls=True)
        assert self.e.check(case, _cr(case, "No URLs here.")) == []

    def test_skip(self) -> None:
        case = TestCase(input="x")
        assert self.e.check(case, _cr(case, "https://bad")) == []


# ===========================================================================
# MarkdownFormatEvaluator
# ===========================================================================


class TestMarkdownFormatEvaluator:
    def setup_method(self) -> None:
        self.e = MarkdownFormatEvaluator()

    def test_pass_with_headers(self) -> None:
        case = TestCase(input="x", expect_markdown=True)
        assert self.e.check(case, _cr(case, "# Title\n\nSome text.")) == []

    def test_pass_with_list(self) -> None:
        case = TestCase(input="x", expect_markdown=True)
        assert self.e.check(case, _cr(case, "Items:\n- one\n- two")) == []

    def test_pass_with_bold(self) -> None:
        case = TestCase(input="x", expect_markdown=True)
        assert self.e.check(case, _cr(case, "This is **bold** text.")) == []

    def test_fail_no_markdown(self) -> None:
        case = TestCase(input="x", expect_markdown=True)
        assert len(self.e.check(case, _cr(case, "Plain text with no formatting."))) == 1

    def test_skip(self) -> None:
        case = TestCase(input="x")
        assert self.e.check(case, _cr(case, "plain")) == []


# ===========================================================================
# LLM-as-judge: new evaluators
# ===========================================================================


class TestConcisenessEvaluator:
    def test_pass(self) -> None:
        e = ConcisenessEvaluator(_mock_provider(9), "m")
        case = TestCase(input="x")
        assert e.check(case, _cr(case, "Short answer.")) == []

    def test_fail(self) -> None:
        e = ConcisenessEvaluator(_mock_provider(3), "m")
        case = TestCase(input="x")
        assert len(e.check(case, _cr(case, "verbose"))) == 1


class TestInstructionFollowingEvaluator:
    def test_pass(self) -> None:
        e = InstructionFollowingEvaluator(_mock_provider(9), "m")
        case = TestCase(input="x", rubric="Answer in bullet points")
        assert e.check(case, _cr(case, "- point 1\n- point 2")) == []

    def test_skip_no_rubric(self) -> None:
        e = InstructionFollowingEvaluator(_mock_provider(2), "m")
        case = TestCase(input="x")
        assert e.check(case, _cr(case, "anything")) == []

    def test_fail(self) -> None:
        e = InstructionFollowingEvaluator(_mock_provider(2), "m")
        case = TestCase(input="x", rubric="Use formal tone")
        assert len(e.check(case, _cr(case, "yo what up"))) == 1


class TestToneEvaluator:
    def test_pass(self) -> None:
        e = ToneEvaluator(_mock_provider(9), "m")
        case = TestCase(input="x", expected_tone="professional")
        assert e.check(case, _cr(case, "formal response")) == []

    def test_skip_no_tone(self) -> None:
        e = ToneEvaluator(_mock_provider(2), "m")
        case = TestCase(input="x")
        assert e.check(case, _cr(case, "anything")) == []

    def test_fail(self) -> None:
        e = ToneEvaluator(_mock_provider(3), "m")
        case = TestCase(input="x", expected_tone="casual")
        assert len(e.check(case, _cr(case, "formal"))) == 1


class TestContextRecallEvaluator:
    def test_pass(self) -> None:
        e = ContextRecallEvaluator(_mock_provider(9), "m")
        case = TestCase(input="x", context="The sky is blue.")
        assert e.check(case, _cr(case, "sky is blue")) == []

    def test_skip_no_context(self) -> None:
        e = ContextRecallEvaluator(_mock_provider(2), "m")
        case = TestCase(input="x")
        assert e.check(case, _cr(case, "anything")) == []

    def test_fail(self) -> None:
        e = ContextRecallEvaluator(_mock_provider(3), "m")
        case = TestCase(input="x", context="Important context")
        assert len(e.check(case, _cr(case, "missed it"))) == 1


class TestContextPrecisionEvaluator:
    def test_pass(self) -> None:
        e = ContextPrecisionEvaluator(_mock_provider(9), "m")
        case = TestCase(input="x", context="Relevant context")
        assert e.check(case, _cr(case, "good answer")) == []

    def test_skip_no_context(self) -> None:
        e = ContextPrecisionEvaluator(_mock_provider(2), "m")
        case = TestCase(input="x")
        assert e.check(case, _cr(case, "anything")) == []


class TestGrammarEvaluator:
    def test_pass(self) -> None:
        e = GrammarEvaluator(_mock_provider(9), "m")
        case = TestCase(input="x")
        assert e.check(case, _cr(case, "Well-written sentence.")) == []

    def test_fail(self) -> None:
        e = GrammarEvaluator(_mock_provider(3), "m")
        case = TestCase(input="x")
        assert len(e.check(case, _cr(case, "bad grammer"))) == 1


class TestSafetyEvaluator:
    def test_pass(self) -> None:
        e = SafetyEvaluator(_mock_provider(10), "m")
        case = TestCase(input="x")
        assert e.check(case, _cr(case, "Safe and helpful response.")) == []

    def test_fail(self) -> None:
        e = SafetyEvaluator(_mock_provider(3), "m")
        case = TestCase(input="x")
        assert len(e.check(case, _cr(case, "dangerous content"))) == 1
