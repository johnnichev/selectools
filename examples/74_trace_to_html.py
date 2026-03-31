"""
Example 74: trace_to_html — HTML Trace Viewer

Renders an AgentTrace as a standalone HTML waterfall timeline.
No external dependencies — the output is a single self-contained HTML file.

Features demonstrated:
- Color-coded step types (LLM call, tool execution, cache hit, error, graph steps)
- Proportional duration bars showing relative timing
- Expandable detail rows (model, token count, cost, tool args/result)
- XSS-safe — user data is HTML-escaped

Run: python examples/74_trace_to_html.py
"""

from pathlib import Path

from selectools import Agent, AgentConfig, trace_to_html
from selectools.providers.stubs import LocalProvider
from selectools.tools import tool


@tool()
def get_weather(city: str) -> str:
    """Return a mock weather report for the given city."""
    return f"Sunny, 22°C in {city}"


@tool()
def get_population(city: str) -> int:
    """Return a mock population for the given city."""
    populations = {"Paris": 2_161_000, "London": 8_982_000, "Tokyo": 13_960_000}
    return populations.get(city, 1_000_000)


def main() -> None:
    provider = LocalProvider()
    agent = Agent(
        provider=provider,
        tools=[get_weather, get_population],
        config=AgentConfig(name="city-reporter", max_iterations=3),
    )

    result = agent.run("What is the weather and population of Paris?")

    # Render the trace as a self-contained HTML file
    html = trace_to_html(result.trace)
    out = Path("trace.html")
    out.write_text(html, encoding="utf-8")

    print(f"Trace written to {out.resolve()}")
    print(f"Steps recorded: {len(result.trace.steps)}")
    print()

    # Introspect the trace programmatically
    for step in result.trace.steps:
        label = step.tool_name or step.model or step.type.value
        print(f"  [{step.type.value}] {label} ({step.duration_ms:.0f}ms)")

    print()
    print("Open trace.html in a browser to view the waterfall timeline.")


if __name__ == "__main__":
    main()
