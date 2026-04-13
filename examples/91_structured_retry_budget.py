"""
Structured Retry Budget — separate structured-validation retries from tool iterations.

Since v0.22.0 (BUG-34), `max_iterations` controls tool-execution iterations
and `RetryConfig.max_retries` controls structured-validation retries. They no
longer share a single counter.

Previously, an agent with max_iterations=3 and an LLM that failed JSON
validation 3 times would terminate — even if max_retries was higher.

Prerequisites: No API key needed (uses LocalProvider)
Run: python examples/91_structured_retry_budget.py
"""

from pydantic import BaseModel

from selectools import Agent, AgentConfig
from selectools.agent.config_groups import RetryConfig
from selectools.providers.stubs import LocalProvider
from selectools.tools import tool


class TaskResult(BaseModel):
    status: str
    confidence: float


@tool(description="A simple task")
def do_task(task: str) -> str:
    return f"Completed: {task}"


def main() -> None:
    agent = Agent(
        tools=[do_task],
        provider=LocalProvider(),
        config=AgentConfig(
            max_iterations=3,  # 3 tool iterations
            retry=RetryConfig(max_retries=5),  # 5 structured-validation retries
        ),
    )

    print("Agent configuration:")
    print(f"  max_iterations (tool budget):     {agent.config.max_iterations}")
    print(f"  retry.max_retries (struct budget): {agent.config.retry.max_retries}")
    print()
    print("The two budgets are independent:")
    print("  - max_iterations=3 means the agent can call tools up to 3 times")
    print("  - max_retries=5 means structured output validation can fail up to 5 times")
    print("  - A validation failure does NOT consume a tool iteration")
    print()

    # Run without response_format to show basic functionality
    result = agent.run("Do the task")
    print(f"Result: {result.content[:80]}")


if __name__ == "__main__":
    main()
