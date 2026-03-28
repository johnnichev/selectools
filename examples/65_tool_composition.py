"""
Example 65: Tool Composition with compose()

Demonstrates composing multiple tools into a single composite tool (v0.19.0):
- Define individual tools with @tool()
- Chain them with compose() into a single tool
- The composite tool exposes the first tool's parameters to the LLM
- Use the composite tool in an agent like any other tool

Uses LocalProvider so no API keys are needed.

Run:
    python examples/65_tool_composition.py
"""

from selectools import Agent, AgentConfig, tool
from selectools.compose import compose
from selectools.providers.stubs import LocalProvider

# --- Individual tools ---


@tool(description="Fetch raw text from a URL")
def fetch_url(url: str) -> str:
    """Fetch the contents of a web page (simulated)."""
    pages = {
        "https://example.com/article": (
            "<h1>AI Agents in 2026</h1>"
            "<p>AI agents are transforming how we build software. "
            "They can plan, reason, and use tools autonomously. "
            "Key trends include multi-agent orchestration, "
            "composable pipelines, and built-in eval frameworks.</p>"
        ),
        "https://example.com/pricing": (
            "<h1>Pricing</h1>" "<p>Starter: $9/mo. Pro: $29/mo. Enterprise: custom.</p>"
        ),
    }
    return pages.get(url, f"<p>Page not found: {url}</p>")


@tool(description="Strip HTML tags from text")
def strip_html(html: str) -> str:
    """Remove HTML tags, returning plain text."""
    import re

    clean = re.sub(r"<[^>]+>", " ", html)
    return " ".join(clean.split())


@tool(description="Summarize text into a one-line synopsis")
def one_line_summary(text: str) -> str:
    """Compress text into a single-line summary."""
    words = text.split()
    if len(words) <= 15:
        return text
    return " ".join(words[:12]) + "... (" + str(len(words)) + " words total)"


def main() -> None:
    print("=" * 60)
    print("Tool Composition Demo")
    print("=" * 60)

    # --- Step 1: Compose fetch + strip + summarize ---
    fetch_and_summarize = compose(
        fetch_url,
        strip_html,
        one_line_summary,
        name="fetch_and_summarize",
        description="Fetch a URL, strip HTML, and return a one-line summary.",
    )

    print(f"\n1. Composed tool: {fetch_and_summarize.name}")
    print(f"   Description: {fetch_and_summarize.description}")
    print(f"   Parameters: {[p.name for p in fetch_and_summarize.parameters]}")

    # --- Step 2: Call the composite tool directly ---
    result = fetch_and_summarize.function(url="https://example.com/article")
    print(f"\n2. Direct call result:")
    print(f"   {result}")

    # --- Step 3: Compose just two tools ---
    fetch_clean = compose(
        fetch_url,
        strip_html,
        name="fetch_clean",
        description="Fetch a URL and return clean plain text.",
    )
    print(f"\n3. Two-tool composition: {fetch_clean.name}")
    result2 = fetch_clean.function(url="https://example.com/pricing")
    print(f"   Result: {result2}")

    # --- Step 4: Use composed tool in an agent ---
    agent = Agent(
        tools=[fetch_and_summarize, fetch_clean],
        provider=LocalProvider(),
        config=AgentConfig(
            model="gpt-5-mini",
            max_iterations=3,
        ),
    )

    print(f"\n4. Agent created with composed tools:")
    for t in agent.tools:
        print(f"   - {t.name}: {t.description}")

    response = agent.run("Summarize the article at https://example.com/article")
    print(f"\n   Agent response: {response.content[:80]}...")

    # --- Step 5: Show that individual tools still work standalone ---
    print(f"\n5. Individual tools still work standalone:")
    html = fetch_url.function(url="https://example.com/pricing")
    print(f"   fetch_url -> {html[:50]}...")
    clean = strip_html.function(html=html)
    print(f"   strip_html -> {clean}")

    print("\nDone! compose() chains tools into a single LLM-callable tool.")


if __name__ == "__main__":
    main()
