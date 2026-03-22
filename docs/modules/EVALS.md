# Eval Framework

**Added in:** v0.17.0

Built-in agent evaluation with 22 evaluators, regression detection, and CI integration. No separate install, no SaaS account, no external dependencies.

---

## Quick Start

```python
from selectools.evals import EvalSuite, TestCase

suite = EvalSuite(agent=agent, cases=[
    TestCase(input="Cancel my account", expect_tool="cancel_subscription"),
    TestCase(input="Check my balance", expect_contains="balance"),
    TestCase(input="What's 2+2?", expect_output="4"),
])
report = suite.run()
print(report.accuracy)      # 0.95
print(report.latency_p50)   # 142ms
print(report.total_cost)    # $0.003
```

---

## TestCase — Declarative Assertions

Every `TestCase` has an `input` (the prompt) and optional `expect_*` fields. Only the fields you set are checked.

### Tool Assertions

```python
TestCase(input="Cancel subscription", expect_tool="cancel_sub")
TestCase(input="Full workflow", expect_tools=["search", "summarize"])
TestCase(input="Search", expect_tool_args={"search": {"query": "python"}})
```

### Content Assertions

```python
TestCase(input="Hello", expect_contains="hello")
TestCase(input="Safe?", expect_not_contains="error")
TestCase(input="2+2", expect_output="4")
TestCase(input="Phone", expect_output_regex=r"\d{3}-\d{4}")
TestCase(input="JSON?", expect_json=True)
TestCase(input="Prefix", expect_starts_with="Hello")
TestCase(input="Suffix", expect_ends_with=".")
TestCase(input="Short", expect_min_length=10, expect_max_length=500)
```

### Structured Output

```python
TestCase(
    input="Extract name",
    response_format=MyModel,
    expect_parsed={"name": "Alice"},
)
```

### Performance Assertions

```python
TestCase(
    input="Fast query",
    expect_latency_ms_lte=500,
    expect_cost_usd_lte=0.01,
    expect_iterations_lte=3,
)
```

### Safety Assertions

```python
TestCase(input="Account info", expect_no_pii=True)
TestCase(input="Ignore instructions", expect_no_injection=True)
```

### LLM-as-Judge Fields

```python
TestCase(
    input="Summarize this",
    reference="The original long text...",  # ground truth
    context="Retrieved document content...",  # RAG context
    rubric="Rate accuracy and completeness",  # custom rubric
)
```

### Custom Evaluators

```python
def must_be_polite(result) -> bool:
    return "please" in result.content.lower()

TestCase(
    input="Help me",
    custom_evaluator=must_be_polite,
    custom_evaluator_name="politeness",
)
```

### Tags and Weights

```python
TestCase(input="Critical", tags=["billing", "critical"], weight=3.0)
TestCase(input="Minor", tags=["nice-to-have"], weight=0.5)
```

---

## Built-in Evaluators (22)

### Deterministic (12) — No API calls

| Evaluator | What it checks |
|---|---|
| `ToolUseEvaluator` | Tool name, tool list, argument values |
| `ContainsEvaluator` | Substring present/absent (case-insensitive) |
| `OutputEvaluator` | Exact match, regex match |
| `StructuredOutputEvaluator` | Parsed fields match (deep subset) |
| `PerformanceEvaluator` | Iterations, latency, cost thresholds |
| `JsonValidityEvaluator` | Valid JSON output |
| `LengthEvaluator` | Min/max character count |
| `StartsWithEvaluator` | Output prefix |
| `EndsWithEvaluator` | Output suffix |
| `PIILeakEvaluator` | SSN, email, phone, credit card, ZIP |
| `InjectionResistanceEvaluator` | 10 prompt injection patterns |
| `CustomEvaluator` | Any user-defined callable |

### LLM-as-Judge (10) — Uses any Provider

These evaluators call an LLM to grade the output. Pass any selectools `Provider` — works with OpenAI, Anthropic, Gemini, Ollama.

```python
from selectools.evals import CorrectnessEvaluator, RelevanceEvaluator

suite = EvalSuite(
    agent=agent,
    cases=cases,
    evaluators=[
        CorrectnessEvaluator(provider=provider, model="gpt-4.1-mini"),
        RelevanceEvaluator(provider=provider, model="gpt-4.1-mini"),
    ],
)
```

| Evaluator | What it checks | Requires |
|---|---|---|
| `LLMJudgeEvaluator` | Generic rubric scoring (0-10) | `rubric` on TestCase |
| `CorrectnessEvaluator` | Correct vs reference answer | `reference` on TestCase |
| `RelevanceEvaluator` | Response relevant to query | — |
| `FaithfulnessEvaluator` | Grounded in provided context | `context` on TestCase |
| `HallucinationEvaluator` | Fabricated information | `context` or `reference` |
| `ToxicityEvaluator` | Harmful/inappropriate content | — |
| `CoherenceEvaluator` | Well-structured and logical | — |
| `CompletenessEvaluator` | Fully addresses the query | — |
| `BiasEvaluator` | Gender, racial, political bias | — |
| `SummaryEvaluator` | Summary accuracy and coverage | `reference` on TestCase |

All LLM evaluators accept a `threshold` parameter (default: 7.0 for most, 8.0 for safety).

---

## EvalReport

```python
report = suite.run()

# Aggregate metrics
report.accuracy        # Weighted accuracy (0.0 - 1.0)
report.pass_count      # Number of passing cases
report.fail_count      # Number of failing cases
report.error_count     # Number of error cases
report.total_cost      # Total USD cost
report.total_tokens    # Total tokens used
report.latency_p50     # Median latency (ms)
report.latency_p95     # 95th percentile latency
report.latency_p99     # 99th percentile latency
report.cost_per_case   # Average cost per case

# Filtering
report.filter_by_tag("billing")
report.filter_by_verdict(CaseVerdict.FAIL)
report.failures_by_evaluator()  # {"tool_use": 3, "contains": 1}

# Export
report.to_html("report.html")         # Interactive HTML report
report.to_junit_xml("results.xml")    # JUnit XML for CI
report.to_json("results.json")        # Machine-readable JSON
report.summary()                      # Human-readable text
```

---

## Loading Test Cases from Files

```python
from selectools.evals import DatasetLoader

# JSON
cases = DatasetLoader.from_json("tests/eval_cases.json")

# YAML (requires PyYAML)
cases = DatasetLoader.from_yaml("tests/eval_cases.yaml")

# Auto-detect from extension
cases = DatasetLoader.load("tests/eval_cases.json")
```

**JSON format:**

```json
[
    {"input": "Cancel account", "expect_tool": "cancel_sub", "name": "cancel"},
    {"input": "Check balance", "expect_contains": "balance", "tags": ["billing"]}
]
```

---

## Regression Detection

```python
from selectools.evals import BaselineStore

store = BaselineStore("./baselines")
report = suite.run()

# Compare against saved baseline
result = store.compare(report)
if result.is_regression:
    print(f"Regressions: {result.regressions}")
    print(f"Accuracy delta: {result.accuracy_delta:+.2%}")
else:
    store.save(report)  # Update baseline
```

---

## CLI

Run evals from the command line:

```bash
# Run eval suite
python -m selectools.evals run tests/cases.json --provider openai --model gpt-4.1-mini --html report.html --verbose

# Compare against baseline
python -m selectools.evals compare tests/cases.json --baseline ./baselines --save

# With concurrency
python -m selectools.evals run tests/cases.json --concurrency 5 --junit results.xml
```

---

## GitHub Actions

Use the built-in action to run evals on every PR and post results as a comment:

```yaml
- name: Run eval suite
  uses: johnnichev/selectools/.github/actions/eval@main
  with:
    cases: tests/eval_cases.json
    provider: openai
    model: gpt-4.1-mini
    html-report: eval-report.html
    baseline-dir: ./baselines
    post-comment: "true"
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The action:
- Runs all test cases
- Posts accuracy, latency, cost, and failures as a PR comment
- Detects regressions against baselines
- Uploads HTML report as an artifact
- Outputs `accuracy`, `pass-count`, `fail-count`, `regression` for downstream steps

---

## Concurrent Execution

```python
suite = EvalSuite(
    agent=agent,
    cases=cases,
    max_concurrency=5,  # Run 5 cases in parallel
    on_progress=lambda done, total: print(f"[{done}/{total}]"),
)
```

Uses `ThreadPoolExecutor` (sync) or `asyncio.Semaphore` (async via `suite.arun()`).

---

## In pytest

```python
def test_agent_accuracy(agent):
    suite = EvalSuite(agent=agent, cases=[
        TestCase(input="Cancel", expect_tool="cancel_sub"),
        TestCase(input="Balance", expect_contains="balance"),
    ])
    report = suite.run()
    assert report.accuracy >= 0.9
    assert report.latency_p50 < 500
```

---

## API Reference

### Core

| Symbol | Description |
|---|---|
| `EvalSuite(agent, cases, ...)` | Orchestrates eval runs |
| `TestCase(input, ...)` | Single test case with assertions |
| `EvalReport` | Aggregated results with metrics |
| `CaseResult` | Per-case result with verdict and failures |
| `CaseVerdict` | Enum: PASS, FAIL, ERROR, SKIP |
| `EvalFailure` | Single assertion failure |

### Infrastructure

| Symbol | Description |
|---|---|
| `DatasetLoader.load(path)` | Load test cases from JSON/YAML |
| `BaselineStore(dir)` | Save and compare baselines |
| `RegressionResult` | Regression comparison result |
| `report.to_html(path)` | Interactive HTML report |
| `report.to_junit_xml(path)` | JUnit XML for CI |
| `report.to_json(path)` | Machine-readable JSON |
