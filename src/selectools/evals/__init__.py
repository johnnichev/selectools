"""Eval framework — evaluate agent accuracy, tool use, latency, and cost."""

from .dataset import DatasetLoader
from .evaluators import (
    ContainsEvaluator,
    CustomEvaluator,
    EndsWithEvaluator,
    Evaluator,
    InjectionResistanceEvaluator,
    JsonValidityEvaluator,
    LengthEvaluator,
    OutputEvaluator,
    PerformanceEvaluator,
    PIILeakEvaluator,
    StartsWithEvaluator,
    StructuredOutputEvaluator,
    ToolUseEvaluator,
)
from .llm_evaluators import (
    BiasEvaluator,
    CoherenceEvaluator,
    CompletenessEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
    LLMJudgeEvaluator,
    RelevanceEvaluator,
    SummaryEvaluator,
    ToxicityEvaluator,
)
from .regression import BaselineStore, RegressionResult
from .report import EvalReport
from .suite import EvalSuite
from .types import CaseResult, CaseVerdict, EvalFailure, EvalMetadata, TestCase

__all__ = [
    # Core
    "EvalSuite",
    "TestCase",
    "CaseResult",
    "CaseVerdict",
    "EvalFailure",
    "EvalMetadata",
    "EvalReport",
    "DatasetLoader",
    "BaselineStore",
    "RegressionResult",
    # Evaluator protocol
    "Evaluator",
    # Deterministic evaluators (12)
    "ToolUseEvaluator",
    "ContainsEvaluator",
    "OutputEvaluator",
    "StructuredOutputEvaluator",
    "PerformanceEvaluator",
    "JsonValidityEvaluator",
    "LengthEvaluator",
    "StartsWithEvaluator",
    "EndsWithEvaluator",
    "PIILeakEvaluator",
    "InjectionResistanceEvaluator",
    "CustomEvaluator",
    # LLM-as-judge evaluators (10)
    "LLMJudgeEvaluator",
    "CorrectnessEvaluator",
    "RelevanceEvaluator",
    "FaithfulnessEvaluator",
    "HallucinationEvaluator",
    "ToxicityEvaluator",
    "CoherenceEvaluator",
    "CompletenessEvaluator",
    "BiasEvaluator",
    "SummaryEvaluator",
]
