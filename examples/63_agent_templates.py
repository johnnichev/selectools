"""
Example 63: Agent Templates

Demonstrates the template system (v0.19.0):
- List available built-in templates
- Load a template to create a pre-configured agent
- Customize a template with overrides
- Run the templated agent

Templates provide ready-made agent configurations for common patterns
like research, code review, data analysis, and customer support.

Uses LocalProvider so no API keys are needed.

Run:
    python examples/63_agent_templates.py
"""

from selectools import Agent, AgentConfig, tool
from selectools.providers.stubs import LocalProvider

# --- Built-in template definitions ---
# In v0.19.0 these ship with selectools; here we define them inline
# to make the example self-contained and runnable today.

TEMPLATES = {
    "researcher": {
        "name": "researcher",
        "system_prompt": (
            "You are a research assistant. Break complex questions into sub-queries, "
            "search for relevant information, and synthesize findings into clear summaries."
        ),
        "model": "gpt-5-mini",
        "temperature": 0.1,
        "max_iterations": 8,
        "reasoning_strategy": "react",
    },
    "code-reviewer": {
        "name": "code-reviewer",
        "system_prompt": (
            "You are a senior code reviewer. Analyze code for bugs, security issues, "
            "performance problems, and style violations. Be specific and actionable."
        ),
        "model": "gpt-5-mini",
        "temperature": 0.0,
        "max_iterations": 4,
        "reasoning_strategy": "cot",
    },
    "data-analyst": {
        "name": "data-analyst",
        "system_prompt": (
            "You are a data analyst. Use available tools to query, transform, "
            "and visualize data. Always explain your methodology."
        ),
        "model": "gpt-5-mini",
        "temperature": 0.0,
        "max_iterations": 10,
    },
    "customer-support": {
        "name": "customer-support",
        "system_prompt": (
            "You are a friendly customer support agent. Be empathetic, helpful, "
            "and concise. Escalate complex issues when appropriate."
        ),
        "model": "gpt-5-mini",
        "temperature": 0.3,
        "max_iterations": 5,
    },
}


def list_templates():
    """List all available templates."""
    return list(TEMPLATES.keys())


def load_template(template_name: str, **overrides) -> AgentConfig:
    """Load a template by name, optionally overriding fields."""
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name!r}. " f"Available: {list_templates()}")
    config_dict = {**TEMPLATES[template_name], **overrides}
    valid_fields = AgentConfig.__dataclass_fields__
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    return AgentConfig(**filtered)


# --- Tools for the demo ---


@tool(description="Search for articles on a topic")
def search_articles(query: str) -> str:
    """Search for research articles."""
    return f"Found 5 articles about '{query}': [1] Overview, [2] Deep Dive, [3] Tutorial, [4] Case Study, [5] Review"


@tool(description="Summarize a piece of text")
def summarize(text: str) -> str:
    """Produce a concise summary."""
    words = text.split()
    return f"Summary ({len(words)} words compressed): {' '.join(words[:10])}..."


def main() -> None:
    print("=" * 60)
    print("Agent Templates Demo")
    print("=" * 60)

    # --- Step 1: List available templates ---
    templates = list_templates()
    print(f"\n1. Available templates ({len(templates)}):")
    for name in templates:
        tmpl = TEMPLATES[name]
        print(f"   - {name}: {tmpl['system_prompt'][:60]}...")

    # --- Step 2: Load the 'researcher' template ---
    config = load_template("researcher")
    print(f"\n2. Loaded 'researcher' template:")
    print(f"   model={config.model}")
    print(f"   temperature={config.temperature}")
    print(f"   max_iterations={config.max_iterations}")
    print(f"   reasoning_strategy={config.reasoning_strategy}")

    agent = Agent(
        tools=[search_articles, summarize],
        provider=LocalProvider(),
        config=config,
    )
    result = agent.run("Research recent advances in quantum computing")
    print(f"   Response: {result.content[:80]}...")

    # --- Step 3: Load with overrides ---
    custom_config = load_template(
        "researcher",
        temperature=0.5,
        max_iterations=3,
    )
    print(f"\n3. Loaded 'researcher' with overrides:")
    print(f"   temperature={custom_config.temperature} (was 0.1)")
    print(f"   max_iterations={custom_config.max_iterations} (was 8)")

    # --- Step 4: Load a different template ---
    support_config = load_template("customer-support")
    print(f"\n4. Loaded 'customer-support' template:")
    print(f"   temperature={support_config.temperature}")
    print(f"   system_prompt={support_config.system_prompt[:50]}...")

    @tool(description="Escalate issue to a human agent")
    def escalate(issue: str) -> str:
        """Escalate a support issue."""
        return f"Escalated to human agent: {issue}"

    support_agent = Agent(
        tools=[escalate],
        provider=LocalProvider(),
        config=support_config,
    )
    result = support_agent.run("I can't log into my account")
    print(f"   Response: {result.content[:80]}...")

    print("\nDone! Templates make it easy to spin up pre-configured agents.")


if __name__ == "__main__":
    main()
