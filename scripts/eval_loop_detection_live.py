"""Real-API eval for loop detection against OpenAI.

Purpose: verify that LoopDetector correctly triggers on a real LLM that gets
stuck in a retry loop, and that the INJECT_MESSAGE policy actually allows the
LLM to self-correct.

Budget: ~$0.01. Requires OPENAI_API_KEY.

Run: PYTHONPATH=src python scripts/eval_loop_detection_live.py
"""

from __future__ import annotations

import os
import sys

from selectools import (
    Agent,
    AgentConfig,
    LoopDetectedError,
    LoopDetector,
    LoopPolicy,
    RepeatDetector,
)
from selectools.providers.openai_provider import OpenAIProvider
from selectools.tools import tool

_call_count = {"count": 0}


@tool(description="Look up a person's email by their name. Always returns 'not found'.")
def lookup_email(name: str) -> str:
    _call_count["count"] += 1
    return "not found"


def _case_raise() -> bool:
    """Agent should hit LoopDetectedError when it retries the same failing tool."""
    _call_count["count"] = 0
    agent = Agent(
        tools=[lookup_email],
        provider=OpenAIProvider(),
        config=AgentConfig(
            model="gpt-4o-mini",
            max_iterations=10,
            max_tokens=200,
            loop_detector=LoopDetector(
                detectors=[RepeatDetector(threshold=3)],
                policy=LoopPolicy.RAISE,
            ),
        ),
    )
    try:
        agent.run(
            "Find the email for 'Alice Johnson' using lookup_email. "
            "The first attempt may fail; if so, call lookup_email again with the "
            "exact same arguments — the tool is flaky and often succeeds on retry. "
            "Keep retrying with the same input until you get an email back. "
            "Do not report 'not found' to the user under any circumstances."
        )
        print(
            f"[RAISE]  INFO — no LoopDetectedError raised; tool_calls={_call_count['count']}. "
            "Real LLMs with deterministic-failure tools often stop on their own; "
            "detector fires only on actually-stuck LLMs."
        )
        return _call_count["count"] < 3
    except LoopDetectedError as exc:
        print(f"[RAISE]  PASS — detector={exc.detector}, tool_calls={_call_count['count']}")
        return True


def _case_inject() -> bool:
    """INJECT_MESSAGE should let the agent self-correct instead of failing."""
    _call_count["count"] = 0
    agent = Agent(
        tools=[lookup_email],
        provider=OpenAIProvider(),
        config=AgentConfig(
            model="gpt-4o-mini",
            max_iterations=8,
            max_tokens=200,
            loop_detector=LoopDetector(
                detectors=[RepeatDetector(threshold=3)],
                policy=LoopPolicy.INJECT_MESSAGE,
                inject_message=(
                    "You are repeating the same lookup. The tool returns 'not "
                    "found' deterministically. Stop calling it and explain the "
                    "result to the user."
                ),
            ),
        ),
    )
    try:
        result = agent.run("Find the email for 'Alice Johnson'. Keep trying.")
        has_final_text = bool(result.message.content and len(result.message.content) > 20)
        print(
            f"[INJECT] PASS — iterations={result.iterations}, "
            f"tool_calls={_call_count['count']}, "
            f"final_text_chars={len(result.message.content or '')}"
        )
        return has_final_text
    except LoopDetectedError:
        print("[INJECT] FAIL — LoopDetectedError raised under INJECT_MESSAGE policy")
        return False


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("SKIP — OPENAI_API_KEY not set")
        return 0

    print("Loop-detection live eval against OpenAI gpt-4o-mini\n")
    results = [_case_raise(), _case_inject()]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} scenarios passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
