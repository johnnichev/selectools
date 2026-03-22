"""Eval framework — evaluate agent accuracy, tool use, latency, and cost."""

from .badge import generate_badge, generate_detailed_badge
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
from .generator import generate_cases
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
from .pairwise import PairwiseCaseResult, PairwiseEval, PairwiseReport
from .regression import BaselineStore, RegressionResult
from .report import EvalReport
from .serve import serve_eval
from .snapshot import SnapshotDiff, SnapshotResult, SnapshotStore
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
    # Pairwise A/B comparison
    "PairwiseEval",
    "PairwiseReport",
    "PairwiseCaseResult",
    # Synthetic test generation
    "generate_cases",
    # Badge generation
    "generate_badge",
    "generate_detailed_badge",
    # Snapshot testing
    "SnapshotStore",
    "SnapshotResult",
    "SnapshotDiff",
    # Live dashboard
    "serve_eval",
]
