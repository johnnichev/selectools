#!/usr/bin/env python3
"""
Token Estimation — estimate costs before running an agent.

Demonstrates:
- estimate_tokens() for single strings
- estimate_run_tokens() for full agent run breakdown
- Pre-execution budget validation

Prerequisites:
    pip install selectools
"""

from selectools import Message, Role
from selectools.token_estimation import TokenEstimate, estimate_run_tokens, estimate_tokens
from selectools.tools import tool

# ---------------------------------------------------------------------------
# Tools (used for schema estimation, not executed)
# ---------------------------------------------------------------------------


@tool(description="Search the web for information")
def web_search(query: str, num_results: int = 5) -> str:
    return f"Results for: {query}"


@tool(description="Read the contents of a file")
def read_file(path: str) -> str:
    return f"Contents of {path}"


@tool(description="Write content to a file")
def write_file(path: str, content: str) -> str:
    return f"Wrote to {path}"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  Token Estimation Demo")
    print("=" * 70)

    # --- Demo 1: Estimate tokens for strings ---
    print("\n--- Demo 1: estimate_tokens() for strings ---\n")

    texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Write a Python function that implements binary search on a sorted list.",
        "A" * 4000,
    ]

    for text in texts:
        tokens = estimate_tokens(text, model="gpt-4o")
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"  {tokens:6d} tokens  |  {preview}")

    # --- Demo 2: Full run estimation ---
    print("\n--- Demo 2: estimate_run_tokens() for agent run ---\n")

    system_prompt = (
        "You are a helpful coding assistant. You can search the web, "
        "read files, and write files. Always explain your reasoning."
    )

    messages = [
        Message(role=Role.USER, content="Read main.py and add error handling to every function."),
        Message(
            role=Role.ASSISTANT,
            content="I'll read the file first to understand the current code.",
        ),
        Message(
            role=Role.USER,
            content="Also add type hints and docstrings to each function.",
        ),
    ]

    estimate: TokenEstimate = estimate_run_tokens(
        messages=messages,
        tools=[web_search, read_file, write_file],
        system_prompt=system_prompt,
        model="gpt-4o",
    )

    print(f"  Model:            {estimate.model}")
    print(f"  Method:           {estimate.method}")
    print(f"  System prompt:    {estimate.system_tokens:6d} tokens")
    print(f"  Messages:         {estimate.message_tokens:6d} tokens")
    print(f"  Tool schemas:     {estimate.tool_schema_tokens:6d} tokens")
    print(f"  Total (1st iter): {estimate.total_tokens:6d} tokens")
    if estimate.context_window:
        print(f"  Context window:   {estimate.context_window:6d} tokens")
        print(f"  Remaining:        {estimate.remaining_tokens:6d} tokens")
        pct = (estimate.total_tokens / estimate.context_window) * 100
        print(f"  Utilization:      {pct:.1f}%")

    # --- Demo 3: Budget validation ---
    print("\n--- Demo 3: Pre-execution budget check ---\n")

    budget_tokens = 5000
    if estimate.total_tokens > budget_tokens:
        print(f"  OVER BUDGET: {estimate.total_tokens} > {budget_tokens} tokens")
        print(f"  Reduce messages or tools before running.")
    else:
        headroom = budget_tokens - estimate.total_tokens
        print(f"  WITHIN BUDGET: {estimate.total_tokens} / {budget_tokens} tokens")
        print(f"  Headroom: {headroom} tokens for LLM response + iterations")

    # --- Demo 4: Compare across models ---
    print("\n--- Demo 4: Compare estimates across models ---\n")

    models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20240620", "gemini-2.0-flash"]

    for model in models:
        est = estimate_run_tokens(
            messages=messages,
            tools=[web_search, read_file, write_file],
            system_prompt=system_prompt,
            model=model,
        )
        ctx = f"ctx={est.context_window:>7d}" if est.context_window else "ctx=unknown"
        print(f"  {model:40s}  {est.total_tokens:5d} tokens  {ctx}  [{est.method}]")

    print("\n" + "=" * 70)
    print("  Token estimation runs locally — no API calls, no cost.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
