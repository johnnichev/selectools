"""
Reasoning tools — make the agent's reasoning explicit, bounded, and inspectable.

Demonstrates:
  - make_reasoning_tools() vs holding a ReasoningTools instance
  - the shared think/analyze budget
  - max_steps enforcement and min_steps guidance
  - reading back the recorded chain

Run: python examples/114_reasoning_tools.py
"""

from selectools.toolbox.reasoning_tools import ReasoningTools, make_reasoning_tools


def main() -> None:
    # Convenience: two tools, default bounds (min 1, max 10).
    tools = make_reasoning_tools()
    print("Tools:", [t.name for t in tools])

    # Hold the instance to inspect the chain afterward.
    reasoning = ReasoningTools(min_steps=2, max_steps=3)
    think, analyze = reasoning.tools

    # Simulate what the agent would do across a few turns.
    print(think.function(thought="The task has three sub-steps; start with data load."))
    print(analyze.function(analysis="Data loaded; row count looks right."))
    print(think.function(thought="Now transform, then write."))
    # Fourth call exceeds max_steps=3 -> enforced stop message.
    print(think.function(thought="One more idea..."))

    print("\nRecorded chain:")
    for step in reasoning.steps:
        print(f"  {step.index}. [{step.kind}] {step.content}")
    print(f"\nTotal reasoning steps: {reasoning.count} (cap was {reasoning.max_steps})")


if __name__ == "__main__":
    main()
