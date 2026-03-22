"""
Example 40: Advanced Eval — A/B Testing, LLM Judges, Snapshots, Badges
======================================================================

Showcases advanced eval features:
- PairwiseEval: compare two agents head-to-head
- LLM-as-judge evaluators with a judge provider
- Snapshot testing: detect output changes across runs
- Badge generation for README
- Cost estimation before running
- History tracking across runs
- Pre-built eval templates

Usage:
    python examples/40_eval_advanced.py

Uses LocalProvider — no API key needed.
"""

import tempfile

from selectools import Agent, AgentConfig, tool
from selectools.evals import (
    BaselineStore,
    EvalSuite,
    PairwiseEval,
    SnapshotStore,
    TestCase,
    generate_badge,
    generate_detailed_badge,
)
from selectools.evals.history import HistoryStore
from selectools.evals.templates import code_quality_suite, customer_support_suite, safety_suite
from selectools.providers.stubs import LocalProvider

# --- Tools ---


@tool(description="Search the knowledge base")
def search(query: str) -> str:
    return f"Found results for: {query}"


@tool(description="Cancel a subscription")
def cancel_subscription(user_id: str) -> str:
    return f"Subscription {user_id} cancelled successfully"


@tool(description="Get account balance")
def get_balance(user_id: str) -> str:
    return f"Balance for {user_id}: $150.00"


# --- Create two agents for A/B comparison ---

agent_a = Agent(
    provider=LocalProvider(),
    config=AgentConfig(model="local"),
    tools=[search, cancel_subscription, get_balance],
)

agent_b = Agent(
    provider=LocalProvider(),
    config=AgentConfig(model="local"),
    tools=[search, cancel_subscription, get_balance],
)

# --- 1. Cost Estimation ---
print("=" * 60)
print("1. COST ESTIMATION")
print("=" * 60)

cases = [
    TestCase(input="Cancel my account", name="cancel", expect_tool="cancel_subscription"),
    TestCase(input="What's my balance?", name="balance", expect_tool="get_balance"),
    TestCase(input="Search for refund policy", name="search", expect_tool="search"),
]

suite = EvalSuite(agent=agent_a, cases=cases, name="demo")
estimate = suite.estimate_cost()
print(f"  Model: {estimate['model']}")
print(f"  Cases: {estimate['cases']}")
print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.6f}")
print(f"  Pricing available: {estimate['pricing_available']}")
print()

# --- 2. A/B Testing ---
print("=" * 60)
print("2. PAIRWISE A/B COMPARISON")
print("=" * 60)

comparison = PairwiseEval(
    agent_a=agent_a,
    agent_b=agent_b,
    cases=cases,
    agent_a_name="Agent-v1",
    agent_b_name="Agent-v2",
)
result = comparison.run()
print(result.summary())
print()

# --- 3. Snapshot Testing ---
print("=" * 60)
print("3. SNAPSHOT TESTING")
print("=" * 60)

tmpdir = tempfile.mkdtemp()
snap_store = SnapshotStore(f"{tmpdir}/snapshots")
report = suite.run()

# First run: all cases are new
snap_result = snap_store.compare(report, "demo")
print(f"  First run — new cases: {len(snap_result.new_cases)}")

# Save snapshot
snap_store.save(report, "demo")

# Second run: compare against snapshot
snap_result = snap_store.compare(report, "demo")
print(f"  Second run — unchanged: {len(snap_result.unchanged)}")
print(f"  Changes detected: {snap_result.has_changes}")
print()

# --- 4. Badge Generation ---
print("=" * 60)
print("4. BADGE GENERATION")
print("=" * 60)

badge_path = f"{tmpdir}/eval-badge.svg"
generate_badge(report, badge_path)
print(f"  Badge: {badge_path}")

detail_path = f"{tmpdir}/eval-badge-detail.svg"
generate_detailed_badge(report, detail_path)
print(f"  Detailed badge: {detail_path}")
print()

# --- 5. History Tracking ---
print("=" * 60)
print("5. HISTORY TRACKING")
print("=" * 60)

history = HistoryStore(f"{tmpdir}/history")
history.record(report)
history.record(report)  # Record twice to see trends
history.record(report)

trend = history.trend("demo")
print(trend.summary())
print(f"  Improving: {trend.is_improving}")
print()

# --- 6. Pre-built Templates ---
print("=" * 60)
print("6. PRE-BUILT EVAL TEMPLATES")
print("=" * 60)

# Customer support template
cs_suite = customer_support_suite(agent_a)
print(f"  Customer Support: {len(cs_suite.cases)} test cases")

# Safety template
safety = safety_suite(agent_a)
print(f"  Safety: {len(safety.cases)} test cases")

# Code quality template
code = code_quality_suite(agent_a)
print(f"  Code Quality: {len(code.cases)} test cases")
print()

# --- 7. Export Reports ---
print("=" * 60)
print("7. EXPORT")
print("=" * 60)

report.to_html(f"{tmpdir}/report.html")
print(f"  HTML: {tmpdir}/report.html")

report.to_json(f"{tmpdir}/report.json")
print(f"  JSON: {tmpdir}/report.json")

report.to_junit_xml(f"{tmpdir}/report.xml")
print(f"  JUnit XML: {tmpdir}/report.xml")
print()

print("Done! All advanced eval features demonstrated.")
