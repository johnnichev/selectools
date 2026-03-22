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
    MarkdownFormatEvaluator,
    OutputEvaluator,
    PerformanceEvaluator,
    PIILeakEvaluator,
    PythonValidityEvaluator,
    RefusalEvaluator,
    SentimentEvaluator,
    SQLValidityEvaluator,
    StartsWithEvaluator,
    StructuredOutputEvaluator,
    ToolOrderEvaluator,
    ToolUseEvaluator,
    UniqueToolsEvaluator,
    URLValidityEvaluator,
    WordCountEvaluator,
)
from .generator import generate_cases
from .history import HistoryEntry, HistoryStore, HistoryTrend
from .llm_evaluators import (
    BiasEvaluator,
    CoherenceEvaluator,
    CompletenessEvaluator,
    ConcisenessEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    GrammarEvaluator,
    HallucinationEvaluator,
    InstructionFollowingEvaluator,
    LLMJudgeEvaluator,
    RelevanceEvaluator,
    SafetyEvaluator,
    SummaryEvaluator,
    ToneEvaluator,
    ToxicityEvaluator,
)
from .pairwise import PairwiseCaseResult, PairwiseEval, PairwiseReport
from .regression import BaselineStore, RegressionResult
from .report import EvalReport
from .serve import serve_eval
from .snapshot import SnapshotDiff, SnapshotResult, SnapshotStore
from .suite import EvalSuite
from .templates import code_quality_suite, customer_support_suite, rag_quality_suite, safety_suite
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
    # Deterministic evaluators (21)
    "ToolUseEvaluator",
    "ContainsEvaluator",
    "OutputEvaluator",
    "StructuredOutputEvaluator",
    "PerformanceEvaluator",
    "JsonValidityEvaluator",
    "LengthEvaluator",
    "StartsWithEvaluator",
    "EndsWithEvaluator",
    "WordCountEvaluator",
    "ToolOrderEvaluator",
    "UniqueToolsEvaluator",
    "PIILeakEvaluator",
    "InjectionResistanceEvaluator",
    "RefusalEvaluator",
    "SentimentEvaluator",
    "PythonValidityEvaluator",
    "SQLValidityEvaluator",
    "URLValidityEvaluator",
    "MarkdownFormatEvaluator",
    "CustomEvaluator",
    # LLM-as-judge evaluators (18)
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
    "ConcisenessEvaluator",
    "InstructionFollowingEvaluator",
    "ToneEvaluator",
    "ContextRecallEvaluator",
    "ContextPrecisionEvaluator",
    "GrammarEvaluator",
    "SafetyEvaluator",
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
    # History tracking
    "HistoryStore",
    "HistoryTrend",
    "HistoryEntry",
    # Pre-built templates
    "customer_support_suite",
    "rag_quality_suite",
    "safety_suite",
    "code_quality_suite",
]
