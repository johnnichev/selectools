"""Eval framework — evaluate agent accuracy, tool use, latency, and cost."""

from .badge import generate_badge, generate_detailed_badge
from .dataset import DatasetLoader
from .evaluators import (
    AgentTrajectoryEvaluator,
    ContainsEvaluator,
    CustomEvaluator,
    EndsWithEvaluator,
    Evaluator,
    ForbiddenWordsEvaluator,
    InjectionResistanceEvaluator,
    JsonSchemaEvaluator,
    JsonValidityEvaluator,
    KeywordDensityEvaluator,
    LengthEvaluator,
    MarkdownFormatEvaluator,
    MultiTurnCoherenceEvaluator,
    OutputEvaluator,
    PerformanceEvaluator,
    PIILeakEvaluator,
    PythonValidityEvaluator,
    ReadabilityEvaluator,
    RefusalEvaluator,
    SemanticSimilarityEvaluator,
    SentimentEvaluator,
    SQLValidityEvaluator,
    StartsWithEvaluator,
    StructuredOutputEvaluator,
    ToolEfficiencyEvaluator,
    ToolOrderEvaluator,
    ToolUseEvaluator,
    UniqueToolsEvaluator,
    URLValidityEvaluator,
    WordCountEvaluator,
)
from .generator import generate_cases
from .history import HistoryEntry, HistoryStore, HistoryTrend
from .llm_evaluators import (
    AnswerAttributionEvaluator,
    BiasEvaluator,
    CoherenceEvaluator,
    CompletenessEvaluator,
    ConcisenessEvaluator,
    ContextPrecisionEvaluator,
    ContextRecallEvaluator,
    CorrectnessEvaluator,
    CustomRubricEvaluator,
    FactConsistencyEvaluator,
    FaithfulnessEvaluator,
    GrammarEvaluator,
    HallucinationEvaluator,
    InstructionFollowingEvaluator,
    LLMJudgeEvaluator,
    RelevanceEvaluator,
    SafetyEvaluator,
    StepReasoningEvaluator,
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
    # Deterministic evaluators (29)
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
    "ReadabilityEvaluator",
    "AgentTrajectoryEvaluator",
    "ToolEfficiencyEvaluator",
    "SemanticSimilarityEvaluator",
    "MultiTurnCoherenceEvaluator",
    "JsonSchemaEvaluator",
    "KeywordDensityEvaluator",
    "ForbiddenWordsEvaluator",
    # LLM-as-judge evaluators (21)
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
    "FactConsistencyEvaluator",
    "CustomRubricEvaluator",
    "AnswerAttributionEvaluator",
    "StepReasoningEvaluator",
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
