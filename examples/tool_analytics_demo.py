"""
Tool Usage Analytics Demo

This example demonstrates the tool analytics feature for tracking and analyzing
agent performance:
- Call frequency and success rates
- Execution times and performance metrics
- Parameter usage patterns
- Cost attribution per tool
- Export analytics to JSON/CSV

Run this demo:
    python examples/tool_analytics_demo.py
"""

import os
import tempfile
from pathlib import Path

from selectools import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from selectools.providers import OpenAIProvider

# Set up API key (use environment variable or .env file)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# ========================
# 1. Define Tools
# ========================


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for information.
    (Mock implementation for demo purposes)
    """
    results = [
        f"Result {i+1} for '{query}': Sample information..." for i in range(min(max_results, 3))
    ]
    return "\n".join(results)


def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)  # noqa: S307 (safe for demo)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def translate_text(text: str, target_language: str = "spanish") -> str:
    """
    Translate text to another language.
    (Mock implementation for demo purposes)
    """
    translations = {
        "spanish": {
            "hello": "hola",
            "world": "mundo",
            "thank you": "gracias",
        },
        "french": {
            "hello": "bonjour",
            "world": "monde",
            "thank you": "merci",
        },
    }

    text_lower = text.lower()
    lang_dict = translations.get(target_language.lower(), {})

    for english, translated in lang_dict.items():
        if english in text_lower:
            return f"Translation to {target_language}: {translated}"

    return f"Translation to {target_language}: [Mock translation of '{text}']"


def format_data(data: str, format_type: str = "json") -> str:
    """Format data in different formats."""
    if format_type == "json":
        return f'{{"data": "{data}"}}'
    elif format_type == "xml":
        return f"<data>{data}</data>"
    elif format_type == "csv":
        return f"data\n{data}"
    else:
        return data


# ========================
# 2. Create Tool Instances
# ========================

search_tool = Tool(
    name="search_web",
    description="Search the web for information",
    parameters=[
        ToolParameter(name="query", param_type=str, description="Search query"),
        ToolParameter(
            name="max_results",
            param_type=int,
            description="Maximum number of results",
            required=False,
        ),
    ],
    function=search_web,
)

calculator_tool = Tool(
    name="calculate",
    description="Calculate mathematical expressions",
    parameters=[ToolParameter(name="expression", param_type=str, description="Math expression")],
    function=calculate,
)

translator_tool = Tool(
    name="translate_text",
    description="Translate text to another language",
    parameters=[
        ToolParameter(name="text", param_type=str, description="Text to translate"),
        ToolParameter(
            name="target_language",
            param_type=str,
            description="Target language",
            required=False,
        ),
    ],
    function=translate_text,
)

formatter_tool = Tool(
    name="format_data",
    description="Format data in different formats",
    parameters=[
        ToolParameter(name="data", param_type=str, description="Data to format"),
        ToolParameter(
            name="format_type",
            param_type=str,
            description="Output format (json, xml, csv)",
            required=False,
        ),
    ],
    function=format_data,
)

# ========================
# 3. Create Agent with Analytics Enabled
# ========================

print("📊 Tool Usage Analytics Demo")
print("=" * 80)
print()

# IMPORTANT: Enable analytics in config
config = AgentConfig(
    model="gpt-4o-mini",
    max_iterations=8,
    enable_analytics=True,  # 🔑 Enable analytics tracking
    verbose=False,  # Keep output clean for this demo
)

provider = OpenAIProvider()
agent = Agent(
    tools=[search_tool, calculator_tool, translator_tool, formatter_tool],
    provider=provider,
    config=config,
)

print("✅ Agent created with analytics enabled")
print(f"   Tools: {len(agent.tools)}")
print(f"   Model: {config.model}")
print()

# ========================
# 4. Run Multiple Queries
# ========================

print("🤖 Running example queries to generate analytics data...")
print()

queries = [
    "Search for Python programming tutorials",
    "Calculate 25 * 18",
    "Translate 'hello world' to Spanish",
    "Search for machine learning with max 3 results",
    "Calculate (100 + 50) / 3",
    "Translate 'thank you' to French",
    "Format 'Hello Analytics' as JSON",
    "Search for AI trends and calculate 2^10",
]

for i, query in enumerate(queries, 1):
    print(f"[{i}/{len(queries)}] Query: {query[:60]}...")
    try:
        response = agent.run([Message(role=Role.USER, content=query)])
        print(f"         ✓ Complete")
    except Exception as e:
        print(f"         ✗ Error: {e}")
    print()

# ========================
# 5. Display Analytics Summary
# ========================

print("=" * 80)
print("📈 ANALYTICS SUMMARY")
print("=" * 80)

analytics = agent.get_analytics()

if analytics:
    # Print formatted summary
    print(analytics.summary())
else:
    print("⚠️  Analytics not available (enable_analytics=True required)")

# ========================
# 6. Detailed Metrics Per Tool
# ========================

print()
print("=" * 80)
print("🔍 DETAILED TOOL METRICS")
print("=" * 80)
print()

if analytics:
    all_metrics = analytics.get_all_metrics()

    for tool_name, metrics in sorted(
        all_metrics.items(), key=lambda x: x[1].total_calls, reverse=True
    ):
        print(f"Tool: {tool_name}")
        print(f"  Total calls: {metrics.total_calls}")
        print(f"  Success rate: {metrics.success_rate:.1f}%")
        print(f"  Failure rate: {metrics.failure_rate:.1f}%")
        print(f"  Avg duration: {metrics.avg_duration:.4f}s")
        print(f"  Total duration: {metrics.total_duration:.4f}s")
        print(f"  Total cost: ${metrics.total_cost:.6f}")

        if metrics.parameter_usage:
            print("  Parameter usage:")
            for param, values in metrics.parameter_usage.items():
                print(f"    {param}:")
                for value, count in sorted(values.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"      '{value}': {count}x")
        print()

# ========================
# 7. Export Analytics
# ========================

print("=" * 80)
print("💾 EXPORTING ANALYTICS")
print("=" * 80)
print()

if analytics:
    # Create temporary directory for exports
    temp_dir = Path(tempfile.mkdtemp())

    # Export to JSON
    json_path = temp_dir / "analytics.json"
    analytics.to_json(json_path)
    print(f"✅ Exported to JSON: {json_path}")

    # Export to CSV
    csv_path = temp_dir / "analytics.csv"
    analytics.to_csv(csv_path)
    print(f"✅ Exported to CSV: {csv_path}")

    print()
    print("JSON content preview:")
    with open(json_path) as f:
        content = f.read()
        print(content[:500] + "..." if len(content) > 500 else content)

    print()
    print("CSV content preview:")
    with open(csv_path) as f:
        lines = f.readlines()
        for line in lines[:5]:  # Show first 5 lines
            print(f"  {line.rstrip()}")
        if len(lines) > 5:
            print(f"  ... ({len(lines)-5} more lines)")

    print()
    print(f"📁 Files saved to: {temp_dir}")

# ========================
# 8. Usage Patterns Analysis
# ========================

print()
print("=" * 80)
print("🎯 USAGE INSIGHTS")
print("=" * 80)
print()

if analytics:
    all_metrics = analytics.get_all_metrics()

    # Most used tool
    if all_metrics:
        most_used = max(all_metrics.values(), key=lambda m: m.total_calls)
        print(f"🏆 Most used tool: {most_used.name} ({most_used.total_calls} calls)")

        # Fastest tool
        fastest = min(
            [m for m in all_metrics.values() if m.total_calls > 0],
            key=lambda m: m.avg_duration,
        )
        print(f"⚡ Fastest tool: {fastest.name} ({fastest.avg_duration:.4f}s avg)")

        # Most reliable tool
        most_reliable = max(all_metrics.values(), key=lambda m: m.success_rate)
        print(f"✅ Most reliable: {most_reliable.name} ({most_reliable.success_rate:.1f}% success)")

        print()
        print("💰 Cost Analysis:")
        print(f"   Total agent cost: ${agent.total_cost:.6f}")
        print(f"   Total tokens: {agent.total_tokens:,}")
        print(f"   Tools executed: {sum(m.total_calls for m in all_metrics.values())}")

# ========================
# 9. Tips and Best Practices
# ========================

print()
print("=" * 80)
print("💡 TIPS FOR USING ANALYTICS")
print("=" * 80)
print()
print("1. Enable analytics with: AgentConfig(enable_analytics=True)")
print("2. Use analytics.summary() for quick overview")
print("3. Export to JSON/CSV for detailed analysis")
print("4. Track parameter patterns to optimize tool design")
print("5. Monitor success rates to identify problematic tools")
print("6. Use duration metrics to optimize performance")
print("7. Combine with cost tracking for budget management")
print()

print("=" * 80)
print("✨ Demo complete!")
print("=" * 80)
