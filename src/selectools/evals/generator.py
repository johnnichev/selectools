"""Synthetic test case generator — auto-generate eval cases from agent tools."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..types import Message, Role
from .types import TestCase


def generate_cases(
    provider: Any,
    model: str,
    tools: List[Any],
    *,
    n: int = 10,
    categories: Optional[List[str]] = None,
    max_tokens: int = 2000,
) -> List[TestCase]:
    """Generate synthetic test cases from tool definitions.

    Uses an LLM to create diverse, realistic test inputs that exercise
    the provided tools, including edge cases and adversarial inputs.

    Args:
        provider: Any selectools Provider (OpenAI, Anthropic, Gemini, Ollama).
        model: Model name to use for generation.
        tools: List of Tool objects (from @tool decorator or Tool class).
        n: Number of test cases to generate.
        categories: Optional list of categories to focus on
            (e.g. ["happy_path", "edge_case", "adversarial"]).
        max_tokens: Max tokens for the generation call.

    Returns:
        List of TestCase objects ready to use in an EvalSuite.
    """
    if not categories:
        categories = ["happy_path", "edge_case", "error_handling", "adversarial"]

    tool_descriptions = []
    for t in tools:
        name = t.name if hasattr(t, "name") else str(t)
        desc = t.description if hasattr(t, "description") else ""
        params = ""
        if hasattr(t, "parameters"):
            params = str(t.parameters)
        elif hasattr(t, "func"):
            import inspect

            sig = inspect.signature(t.func)
            params = str(sig)
        tool_descriptions.append(f"- {name}{params}: {desc}")

    tools_text = "\n".join(tool_descriptions)
    categories_text = ", ".join(categories)

    prompt = f"""Generate exactly {n} test cases for an AI agent that has the following tools:

{tools_text}

Generate a diverse mix across these categories: {categories_text}

For each test case, output a JSON object with these fields:
- "input": the user's query (string)
- "name": short descriptive name (string)
- "expect_tool": which tool should be called (string or null)
- "expect_contains": substring that should appear in the response (string or null)
- "tags": list of tags like the category (list of strings)

Output a JSON array of {n} objects. Output ONLY the JSON array, no other text."""

    messages = [Message(role=Role.USER, content=prompt)]
    response, _ = provider.complete(
        model=model,
        system_prompt="You are a test case generator. Output only valid JSON.",
        messages=messages,
        tools=None,
        temperature=0.7,
        max_tokens=max_tokens,
    )

    return _parse_generated_cases(response.content or "")


def _parse_generated_cases(text: str) -> List[TestCase]:
    """Parse LLM output into TestCase objects."""
    # Try to extract JSON array from the response
    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    cases: List[TestCase] = []
    for item in data:
        if not isinstance(item, dict) or "input" not in item:
            continue
        kwargs: Dict[str, Any] = {"input": item["input"]}
        if item.get("name"):
            kwargs["name"] = item["name"]
        if item.get("expect_tool"):
            kwargs["expect_tool"] = item["expect_tool"]
        if item.get("expect_contains"):
            kwargs["expect_contains"] = item["expect_contains"]
        if item.get("expect_not_contains"):
            kwargs["expect_not_contains"] = item["expect_not_contains"]
        if item.get("tags"):
            kwargs["tags"] = item["tags"]
        cases.append(TestCase(**kwargs))

    return cases
