"""Tests for the 12 new evaluators added in v0.19.1."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from selectools.evals.evaluators import (
    AgentTrajectoryEvaluator,
    ForbiddenWordsEvaluator,
    JsonSchemaEvaluator,
    KeywordDensityEvaluator,
    MultiTurnCoherenceEvaluator,
    ReadabilityEvaluator,
    SemanticSimilarityEvaluator,
    ToolEfficiencyEvaluator,
    _count_syllables,
    _tf_cosine,
)
from selectools.evals.llm_evaluators import (
    AnswerAttributionEvaluator,
    CustomRubricEvaluator,
    FactConsistencyEvaluator,
    StepReasoningEvaluator,
)
from selectools.evals.types import CaseResult, CaseVerdict, EvalFailure, TestCase

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_case(**kwargs) -> TestCase:
    return TestCase(input="test input", **kwargs)


def _make_result(content: str = "", tool_calls=None) -> CaseResult:
    agent_result = MagicMock()
    agent_result.content = content
    agent_result.tool_calls = []
    cr = CaseResult(
        case=_make_case(),
        verdict=CaseVerdict.PASS,
        agent_result=agent_result,
        tool_calls=tool_calls or [],
    )
    return cr


def _make_llm_provider(score: float) -> MagicMock:
    provider = MagicMock()
    msg = MagicMock()
    msg.content = f"This looks good. Score: {score}"
    provider.complete = MagicMock(return_value=(msg, MagicMock()))
    return provider


# ---------------------------------------------------------------------------
# _count_syllables helper
# ---------------------------------------------------------------------------


class TestCountSyllables:
    def test_simple_words(self):
        assert _count_syllables("cat") == 1
        assert _count_syllables("hello") == 2
        assert _count_syllables("beautiful") >= 3

    def test_silent_e(self):
        assert _count_syllables("cake") == 1

    def test_empty(self):
        assert _count_syllables("") == 0


# ---------------------------------------------------------------------------
# ReadabilityEvaluator
# ---------------------------------------------------------------------------


class TestReadabilityEvaluator:
    def test_no_activation_without_field(self):
        ev = ReadabilityEvaluator()
        case = _make_case()
        result = _make_result("Simple text here.")
        assert ev.check(case, result) == []

    def test_no_activation_without_content(self):
        ev = ReadabilityEvaluator()
        case = _make_case(expect_readability_gte=60.0)
        result = _make_result("")
        assert ev.check(case, result) == []

    def test_no_activation_when_agent_result_none(self):
        ev = ReadabilityEvaluator()
        case = _make_case(expect_readability_gte=60.0)
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, agent_result=None)
        assert ev.check(case, cr) == []

    def test_pass_simple_text(self):
        ev = ReadabilityEvaluator()
        # Very simple text should score high on Flesch
        case = _make_case(expect_readability_gte=30.0)
        result = _make_result("The cat sat on the mat. It is a big cat.")
        failures = ev.check(case, result)
        assert failures == []

    def test_fail_very_low_threshold(self):
        ev = ReadabilityEvaluator()
        # Threshold of 200 is impossible to achieve (max is ~120)
        case = _make_case(expect_readability_gte=200.0)
        result = _make_result("The cat sat.")
        failures = ev.check(case, result)
        assert len(failures) == 1
        assert "readability" in failures[0].evaluator_name


# ---------------------------------------------------------------------------
# AgentTrajectoryEvaluator
# ---------------------------------------------------------------------------


class TestAgentTrajectoryEvaluator:
    def test_no_activation_without_field(self):
        ev = AgentTrajectoryEvaluator()
        case = _make_case()
        result = _make_result(tool_calls=["search"])
        assert ev.check(case, result) == []

    def test_pass_correct_subsequence(self):
        ev = AgentTrajectoryEvaluator()
        case = _make_case(expect_trajectory=["search", "write"])
        result = _make_result(tool_calls=["search", "read", "write"])
        assert ev.check(case, result) == []

    def test_pass_exact_sequence(self):
        ev = AgentTrajectoryEvaluator()
        case = _make_case(expect_trajectory=["a", "b", "c"])
        result = _make_result(tool_calls=["a", "b", "c"])
        assert ev.check(case, result) == []

    def test_fail_missing_step(self):
        ev = AgentTrajectoryEvaluator()
        case = _make_case(expect_trajectory=["search", "write", "review"])
        result = _make_result(tool_calls=["search", "write"])
        failures = ev.check(case, result)
        assert len(failures) == 1
        assert "review" in failures[0].message

    def test_empty_expected_passes(self):
        ev = AgentTrajectoryEvaluator()
        case = _make_case(expect_trajectory=[])
        result = _make_result(tool_calls=["anything"])
        assert ev.check(case, result) == []


# ---------------------------------------------------------------------------
# ToolEfficiencyEvaluator
# ---------------------------------------------------------------------------


class TestToolEfficiencyEvaluator:
    def test_no_activation_without_field(self):
        ev = ToolEfficiencyEvaluator()
        case = _make_case()
        result = _make_result(tool_calls=["a", "b", "c", "d", "e"])
        assert ev.check(case, result) == []

    def test_pass_under_limit(self):
        ev = ToolEfficiencyEvaluator()
        case = _make_case(expect_max_tools=3)
        result = _make_result(tool_calls=["a", "b"])
        assert ev.check(case, result) == []

    def test_pass_at_limit(self):
        ev = ToolEfficiencyEvaluator()
        case = _make_case(expect_max_tools=3)
        result = _make_result(tool_calls=["a", "b", "c"])
        assert ev.check(case, result) == []

    def test_fail_over_limit(self):
        ev = ToolEfficiencyEvaluator()
        case = _make_case(expect_max_tools=2)
        result = _make_result(tool_calls=["a", "b", "c", "d"])
        failures = ev.check(case, result)
        assert len(failures) == 1
        assert "4" in failures[0].message


# ---------------------------------------------------------------------------
# SemanticSimilarityEvaluator
# ---------------------------------------------------------------------------


class TestSemanticSimilarityEvaluator:
    def test_no_activation_without_field(self):
        ev = SemanticSimilarityEvaluator()
        case = _make_case()
        result = _make_result("hello world")
        assert ev.check(case, result) == []

    def test_no_activation_without_reference(self):
        ev = SemanticSimilarityEvaluator()
        case = _make_case(expect_semantic_similarity_gte=0.5)
        result = _make_result("hello world")
        assert ev.check(case, result) == []

    def test_pass_identical_text(self):
        ev = SemanticSimilarityEvaluator()
        text = "the quick brown fox jumps over the lazy dog"
        case = _make_case(expect_semantic_similarity_gte=0.99, reference=text)
        result = _make_result(text)
        assert ev.check(case, result) == []

    def test_fail_completely_different(self):
        ev = SemanticSimilarityEvaluator()
        case = _make_case(expect_semantic_similarity_gte=0.9, reference="cat dog mouse")
        result = _make_result("apple orange banana mango grape")
        failures = ev.check(case, result)
        assert len(failures) == 1

    def test_threshold_from_constructor(self):
        ev = SemanticSimilarityEvaluator(threshold=0.9)
        text = "the quick brown fox"
        case = _make_case(reference=text)  # no expect_semantic_similarity_gte
        result = _make_result("completely different words zap zap")
        failures = ev.check(case, result)
        assert len(failures) == 1

    def test_zero_threshold_is_not_falsy(self):
        """Regression: expect_semantic_similarity_gte=0.0 must be respected, not treated as
        falsy and replaced by self.threshold.  A 0.0 threshold means 'any similarity passes'."""
        ev = SemanticSimilarityEvaluator(threshold=0.9)  # high constructor threshold
        # 0.0 threshold: even completely dissimilar text should pass
        case = _make_case(
            reference="completely different content XYZ",
            expect_semantic_similarity_gte=0.0,
        )
        result = _make_result("hello world python code banana")
        # Must pass — threshold is 0.0
        failures = ev.check(case, result)
        assert failures == [], (
            "expect_semantic_similarity_gte=0.0 was treated as falsy; "
            "constructor threshold 0.9 was used instead"
        )

    def test_zero_threshold_constructor_activates_on_reference(self):
        """SemanticSimilarityEvaluator(threshold=0.0) with a reference but no case threshold
        should activate and always pass (similarity >= 0.0 is trivially true)."""
        ev = SemanticSimilarityEvaluator(threshold=0.0)
        case = _make_case(reference="xyz abc")  # no case-level threshold
        result = _make_result("completely unrelated words mango grape")
        failures = ev.check(case, result)
        assert failures == []


# ---------------------------------------------------------------------------
# _tf_cosine helper (renamed from _tfidf_cosine in v0.19.1 bug hunt)
# ---------------------------------------------------------------------------


class TestTfidfCosine:
    def test_identical(self):
        assert _tf_cosine("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _tf_cosine("apple banana", "car dog") == 0.0

    def test_partial_overlap(self):
        score = _tf_cosine("hello world", "hello there")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# MultiTurnCoherenceEvaluator
# ---------------------------------------------------------------------------


class TestMultiTurnCoherenceEvaluator:
    def test_no_activation_without_field(self):
        ev = MultiTurnCoherenceEvaluator()
        case = _make_case()
        result = _make_result("X is good. X is not good.")
        assert ev.check(case, result) == []

    def test_no_activation_when_false(self):
        ev = MultiTurnCoherenceEvaluator()
        case = _make_case(expect_coherent_turns=False)
        result = _make_result("some text")
        assert ev.check(case, result) == []

    def test_pass_coherent_text(self):
        ev = MultiTurnCoherenceEvaluator()
        case = _make_case(expect_coherent_turns=True)
        result = _make_result("Python is great. It is widely used. Many developers enjoy it.")
        assert ev.check(case, result) == []

    def test_fail_contradictory_text(self):
        ev = MultiTurnCoherenceEvaluator()
        case = _make_case(expect_coherent_turns=True)
        # Explicit contradiction pattern that the regex should catch
        result = _make_result("Python is fast. On the other hand, python is not fast.")
        failures = ev.check(case, result)
        assert len(failures) == 1

    def test_empty_content_passes(self):
        ev = MultiTurnCoherenceEvaluator()
        case = _make_case(expect_coherent_turns=True)
        result = _make_result("")
        assert ev.check(case, result) == []


# ---------------------------------------------------------------------------
# JsonSchemaEvaluator
# ---------------------------------------------------------------------------


class TestJsonSchemaEvaluator:
    def test_no_activation_without_field(self):
        ev = JsonSchemaEvaluator()
        case = _make_case()
        result = _make_result('{"name": "Alice"}')
        assert ev.check(case, result) == []

    def test_pass_valid_schema_match(self):
        ev = JsonSchemaEvaluator()
        schema = {"type": "object", "required": ["name"]}
        case = _make_case(expect_json_schema=schema)
        result = _make_result('{"name": "Alice", "age": 30}')
        assert ev.check(case, result) == []

    def test_fail_missing_required_field(self):
        ev = JsonSchemaEvaluator()
        schema = {"type": "object", "required": ["name", "email"]}
        case = _make_case(expect_json_schema=schema)
        result = _make_result('{"name": "Alice"}')
        failures = ev.check(case, result)
        assert len(failures) == 1

    def test_fail_not_valid_json(self):
        ev = JsonSchemaEvaluator()
        case = _make_case(expect_json_schema={"type": "object"})
        result = _make_result("not json at all")
        failures = ev.check(case, result)
        assert len(failures) == 1
        assert "not valid JSON" in failures[0].message

    def test_pass_json_in_code_block(self):
        ev = JsonSchemaEvaluator()
        schema = {"type": "object", "required": ["x"]}
        case = _make_case(expect_json_schema=schema)
        result = _make_result('```json\n{"x": 1}\n```')
        assert ev.check(case, result) == []


# ---------------------------------------------------------------------------
# KeywordDensityEvaluator
# ---------------------------------------------------------------------------


class TestKeywordDensityEvaluator:
    def test_no_activation_without_field(self):
        ev = KeywordDensityEvaluator()
        case = _make_case()
        result = _make_result("some content here")
        assert ev.check(case, result) == []

    def test_pass_all_keywords_present(self):
        ev = KeywordDensityEvaluator()
        case = _make_case(expect_keywords=["python", "agent"])
        result = _make_result("Python is an agent framework language")
        assert ev.check(case, result) == []

    def test_fail_missing_keyword(self):
        ev = KeywordDensityEvaluator()
        case = _make_case(expect_keywords=["python", "java"])
        result = _make_result("Python is great")
        failures = ev.check(case, result)
        assert len(failures) == 1
        assert "java" in failures[0].message.lower()

    def test_fail_density_below_min(self):
        ev = KeywordDensityEvaluator()
        # "python" appears once in 100 words → density 0.01
        content = "python " + "other " * 99
        case = _make_case(expect_keywords=["python"], expect_keyword_density_min=0.05)
        result = _make_result(content)
        failures = ev.check(case, result)
        assert len(failures) == 1
        assert "density" in failures[0].message

    def test_pass_density_above_min(self):
        ev = KeywordDensityEvaluator()
        content = "python python python other other"  # 3/5 = 0.6
        case = _make_case(expect_keywords=["python"], expect_keyword_density_min=0.4)
        result = _make_result(content)
        assert ev.check(case, result) == []


# ---------------------------------------------------------------------------
# ForbiddenWordsEvaluator
# ---------------------------------------------------------------------------


class TestForbiddenWordsEvaluator:
    def test_no_activation_without_field(self):
        ev = ForbiddenWordsEvaluator()
        case = _make_case()
        result = _make_result("some content with badword")
        assert ev.check(case, result) == []

    def test_pass_no_forbidden_words(self):
        ev = ForbiddenWordsEvaluator()
        case = _make_case(expect_no_keywords=["competitor", "rival"])
        result = _make_result("Our product is great and helps users.")
        assert ev.check(case, result) == []

    def test_fail_forbidden_word_present(self):
        ev = ForbiddenWordsEvaluator()
        case = _make_case(expect_no_keywords=["competitor"])
        result = _make_result("Our competitor has a better product.")
        failures = ev.check(case, result)
        assert len(failures) == 1
        assert "competitor" in failures[0].message

    def test_case_insensitive(self):
        ev = ForbiddenWordsEvaluator()
        case = _make_case(expect_no_keywords=["BADWORD"])
        result = _make_result("This contains badword in lowercase.")
        failures = ev.check(case, result)
        assert len(failures) == 1

    def test_multiple_forbidden_words_detected(self):
        ev = ForbiddenWordsEvaluator()
        case = _make_case(expect_no_keywords=["foo", "bar"])
        result = _make_result("foo and bar are both here")
        failures = ev.check(case, result)
        assert len(failures) == 1  # One failure listing all found words


# ---------------------------------------------------------------------------
# FactConsistencyEvaluator (LLM)
# ---------------------------------------------------------------------------


class TestFactConsistencyEvaluator:
    def test_no_activation_without_context(self):
        provider = _make_llm_provider(9.0)
        ev = FactConsistencyEvaluator(provider=provider, model="test")
        case = _make_case()  # no context
        result = _make_result("some response")
        assert ev.check(case, result) == []

    def test_pass_high_score(self):
        provider = _make_llm_provider(9.0)
        ev = FactConsistencyEvaluator(provider=provider, model="test")
        case = _make_case(context="The sky is blue.")
        result = _make_result("The sky appears blue due to light scattering.")
        assert ev.check(case, result) == []

    def test_fail_low_score(self):
        provider = _make_llm_provider(3.0)
        ev = FactConsistencyEvaluator(provider=provider, model="test", threshold=7.0)
        case = _make_case(context="The sky is blue.")
        result = _make_result("The sky is green.")
        failures = ev.check(case, result)
        assert len(failures) == 1
        assert "3.0" in failures[0].message

    def test_fail_unparseable_score(self):
        provider = MagicMock()
        msg = MagicMock()
        msg.content = "This is a response with no score"
        provider.complete = MagicMock(return_value=(msg, MagicMock()))
        ev = FactConsistencyEvaluator(provider=provider, model="test")
        case = _make_case(context="some context")
        result = _make_result("some response")
        failures = ev.check(case, result)
        assert len(failures) == 1
        assert "unparseable" in failures[0].actual


# ---------------------------------------------------------------------------
# CustomRubricEvaluator (LLM)
# ---------------------------------------------------------------------------


class TestCustomRubricEvaluator:
    def test_requires_criteria(self):
        provider = _make_llm_provider(8.0)
        with pytest.raises(ValueError, match="at least one criterion"):
            CustomRubricEvaluator(provider=provider, model="test", criteria=[])

    def test_pass_high_overall_score(self):
        provider = _make_llm_provider(8.0)
        ev = CustomRubricEvaluator(provider=provider, model="test", criteria=["clarity"])
        case = _make_case()
        result = _make_result("Clear and concise response.")
        assert ev.check(case, result) == []

    def test_fail_low_overall_score(self):
        provider = _make_llm_provider(3.0)
        ev = CustomRubricEvaluator(
            provider=provider, model="test", criteria=["clarity"], threshold=7.0
        )
        case = _make_case()
        result = _make_result("Unclear response.")
        failures = ev.check(case, result)
        assert any("3.0" in f.message for f in failures)

    def test_case_rubric_overrides_criteria(self):
        # case.rubric = "accuracy,tone" should override constructor criteria
        provider = _make_llm_provider(8.0)
        ev = CustomRubricEvaluator(provider=provider, model="test", criteria=["original_criterion"])
        case = _make_case(rubric="accuracy,tone")
        result = _make_result("response")
        ev.check(case, result)
        # Verify the provider was called (it uses the case rubric)
        provider.complete.assert_called_once()

    def test_no_activation_when_no_agent_result(self):
        provider = _make_llm_provider(8.0)
        ev = CustomRubricEvaluator(provider=provider, model="test", criteria=["clarity"])
        case = _make_case()
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, agent_result=None)
        assert ev.check(case, cr) == []


# ---------------------------------------------------------------------------
# AnswerAttributionEvaluator (LLM)
# ---------------------------------------------------------------------------


class TestAnswerAttributionEvaluator:
    def test_no_activation_without_context(self):
        provider = _make_llm_provider(9.0)
        ev = AnswerAttributionEvaluator(provider=provider, model="test")
        case = _make_case()  # no context
        result = _make_result("response")
        assert ev.check(case, result) == []

    def test_pass_high_attribution(self):
        provider = _make_llm_provider(9.0)
        ev = AnswerAttributionEvaluator(provider=provider, model="test")
        case = _make_case(context="Paris is the capital of France.")
        result = _make_result("Paris is the capital of France, as stated in the context.")
        assert ev.check(case, result) == []

    def test_fail_low_attribution(self):
        provider = _make_llm_provider(2.0)
        ev = AnswerAttributionEvaluator(provider=provider, model="test", threshold=7.0)
        case = _make_case(context="Paris is the capital of France.")
        result = _make_result("London is the capital of France.")
        failures = ev.check(case, result)
        assert len(failures) == 1


# ---------------------------------------------------------------------------
# StepReasoningEvaluator (LLM)
# ---------------------------------------------------------------------------


class TestStepReasoningEvaluator:
    def test_pass_good_reasoning(self):
        provider = _make_llm_provider(8.0)
        ev = StepReasoningEvaluator(provider=provider, model="test")
        case = _make_case()
        result = _make_result("Step 1: identify. Step 2: analyze. Step 3: conclude.")
        assert ev.check(case, result) == []

    def test_fail_no_reasoning(self):
        provider = _make_llm_provider(2.0)
        ev = StepReasoningEvaluator(provider=provider, model="test", threshold=7.0)
        case = _make_case()
        result = _make_result("It just is.")
        failures = ev.check(case, result)
        assert len(failures) == 1

    def test_uses_default_rubric_when_case_rubric_none(self):
        provider = _make_llm_provider(8.0)
        ev = StepReasoningEvaluator(provider=provider, model="test")
        case = _make_case(rubric=None)
        result = _make_result("Step-by-step analysis here.")
        ev.check(case, result)
        call_args = provider.complete.call_args
        # The system prompt is in the call — verify it was called
        provider.complete.assert_called_once()

    def test_uses_case_rubric_when_provided(self):
        provider = _make_llm_provider(9.0)
        ev = StepReasoningEvaluator(provider=provider, model="test")
        case = _make_case(rubric="Show your work clearly")
        result = _make_result("1+1=2 because one plus one equals two.")
        assert ev.check(case, result) == []

    def test_no_activation_when_no_agent_result(self):
        provider = _make_llm_provider(8.0)
        ev = StepReasoningEvaluator(provider=provider, model="test")
        case = _make_case()
        cr = CaseResult(case=case, verdict=CaseVerdict.PASS, agent_result=None)
        assert ev.check(case, cr) == []
