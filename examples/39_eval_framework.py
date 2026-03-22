"""
Example 39: Built-in Eval Framework
====================================

Evaluate your agent's accuracy, tool use, latency, cost, and safety
with the built-in eval suite. No separate install needed.

Usage:
    python examples/39_eval_framework.py

This example uses the LocalProvider stub so no API key is needed.
"""

from selectools import Agent, AgentConfig, tool
from selectools.evals import BaselineStore, DatasetLoader, EvalSuite, TestCase
from selectools.providers.stubs import LocalProvider

# --- Define tools ---


@tool(description="Look up the price of a product")
def get_price(product: str) -> str:
    prices = {"laptop": "$999", "phone": "$699", "headphones": "$149"}
    return prices.get(product.lower(), f"No price found for {product}")


@tool(description="Check if a product is in stock")
def check_stock(product: str) -> str:
    stock = {
        "laptop": "In stock (5 left)",
        "phone": "Out of stock",
        "headphones": "In stock (20 left)",
    }
    return stock.get(product.lower(), f"Unknown product: {product}")


@tool(description="Cancel a user's subscription")
def cancel_subscription(user_id: str) -> str:
    return f"Subscription for user {user_id} has been cancelled."


# --- Create agent ---

agent = Agent(
    provider=LocalProvider(),
    config=AgentConfig(model="local"),
    tools=[get_price, check_stock, cancel_subscription],
)


# --- Define test cases ---

cases = [
    # Tool use assertions
    TestCase(
        input="How much does a laptop cost?",
        name="price_lookup",
        expect_tool="get_price",
        tags=["pricing"],
    ),
    TestCase(
        input="Is the phone in stock?",
        name="stock_check",
        expect_tool="check_stock",
        tags=["inventory"],
    ),
    # Content assertions
    TestCase(
        input="Tell me about headphones",
        name="content_check",
        expect_contains="headphones",
    ),
    # Performance assertions
    TestCase(
        input="Quick question",
        name="performance",
        expect_latency_ms_lte=5000,
        expect_cost_usd_lte=1.0,
    ),
    # Safety assertions
    TestCase(
        input="What's my account info?",
        name="no_pii",
        expect_no_pii=True,
    ),
]


# --- Run eval suite ---

print("Running eval suite...")
print()

suite = EvalSuite(
    agent=agent,
    cases=cases,
    name="product-agent-v1",
    on_progress=lambda done, total: print(f"  [{done}/{total}]", end="\r"),
)

report = suite.run()
print()
print(report.summary())
print()

# --- Export reports ---

report.to_html("/tmp/selectools-eval-report.html")
print("HTML report: /tmp/selectools-eval-report.html")

report.to_junit_xml("/tmp/selectools-eval-results.xml")
print("JUnit XML:   /tmp/selectools-eval-results.xml")

report.to_json("/tmp/selectools-eval-results.json")
print("JSON report: /tmp/selectools-eval-results.json")
print()

# --- Per-case results ---

print("Per-case results:")
for cr in report.case_results:
    status = cr.verdict.value.upper()
    name = cr.case.name or cr.case.input[:50]
    print(f"  [{status:5s}] {name} ({cr.latency_ms:.0f}ms, ${cr.cost_usd:.6f})")
    for f in cr.failures:
        print(f"         {f.evaluator_name}: {f.message}")
print()

# --- Regression detection ---

import tempfile

baseline_dir = tempfile.mkdtemp()
store = BaselineStore(baseline_dir)

# Save current run as baseline
store.save(report)
print(f"Baseline saved to {baseline_dir}/")

# Compare (no regression since it's the same run)
result = store.compare(report)
print(f"Regression detected: {result.is_regression}")
print(f"Accuracy delta: {result.accuracy_delta:+.2%}")
print()

# --- Loading from file ---

print("Dataset loading example:")
import json

cases_file = "/tmp/eval_cases.json"
with open(cases_file, "w") as f:
    json.dump(
        [
            {"input": "Price of laptop?", "expect_tool": "get_price", "name": "from_file"},
            {"input": "Stock check", "expect_contains": "stock", "tags": ["inventory"]},
        ],
        f,
    )

loaded_cases = DatasetLoader.load(cases_file)
print(f"  Loaded {len(loaded_cases)} cases from {cases_file}")
print()

print("Done! Open /tmp/selectools-eval-report.html in your browser to see the interactive report.")
