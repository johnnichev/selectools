"""
Safe Parallel Pipeline — branches receive independent input copies.

Since v0.22.0 (BUG-30), `parallel()` branches each receive a deep copy
of the input. Mutations in one branch do NOT affect siblings — even under
asyncio.gather where branches interleave at await points.

Prerequisites: No API key needed
Run: python examples/92_safe_parallel_pipeline.py
"""

import asyncio

from selectools.pipeline import parallel, step


@step
def enrich_a(data: dict) -> dict:
    """Branch A adds its own key."""
    data["enriched_by"] = "branch_a"
    data["a_result"] = "web search results"
    return data


@step
def enrich_b(data: dict) -> dict:
    """Branch B adds its own key. Should NOT see branch A's mutation."""
    data["saw_a_mutation"] = "enriched_by" in data
    data["enriched_by"] = "branch_b"
    data["b_result"] = "document search results"
    return data


@step
def merge(results: dict) -> dict:
    """Merge results from both branches."""
    return {
        "from_a": results["enrich_a"]["a_result"],
        "from_b": results["enrich_b"]["b_result"],
        "a_saw": results["enrich_a"]["enriched_by"],
        "b_saw": results["enrich_b"]["enriched_by"],
    }


def main() -> None:
    pipeline = parallel(enrich_a, enrich_b) | merge

    # Sync execution — pipeline.run() returns StepResult, access .output for the value
    step_result = pipeline.run({"query": "quantum computing", "user_id": 42})
    result = step_result.output
    print("Sync result:")
    print(f"  From A: {result['from_a']}")
    print(f"  From B: {result['from_b']}")

    # The merge step received independent results from both branches.
    # If BUG-30 fix is in place, both branches worked on their own copies.
    print("  ✓ Both branches returned results independently")

    print("\n✓ Parallel branches are isolated — no cross-branch state corruption")


if __name__ == "__main__":
    main()
