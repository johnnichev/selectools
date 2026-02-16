"""
Toolbox Demo - Showcase pre-built tools from the selectools toolbox.

This example demonstrates using the built-in tools for:
- File operations
- Data processing
- Text manipulation
- Date/time utilities
- Web requests (commented out to avoid external dependencies)

Run: python examples/toolbox_demo.py
"""

from selectools import Agent, AgentConfig, Message, Role
from selectools.providers.stubs import LocalProvider
from selectools.toolbox import get_all_tools, get_tools_by_category

# Try to use OpenAI if available, otherwise fall back to LocalProvider
try:
    from selectools.models import OpenAI
    from selectools.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)
    print(f"Using OpenAI provider ({OpenAI.GPT_4O_MINI.id})")
except Exception:
    provider = LocalProvider()
    print("Using LocalProvider (no API calls)")


def demo_file_operations() -> None:
    """Demo file operation tools."""
    print("\n" + "=" * 60)
    print("FILE OPERATIONS DEMO")
    print("=" * 60)

    tools = get_tools_by_category("file")
    agent = Agent(
        tools=tools, provider=provider, config=AgentConfig(max_iterations=5, temperature=0.3)
    )

    # Test writing and reading files
    response = agent.run(
        [
            Message(
                role=Role.USER,
                content="Write 'Hello from Selectools!' to a file called test_output.txt, then read it back to confirm",
            )
        ]
    )
    print(f"\nAgent: {response.content}")


def demo_data_processing() -> None:
    """Demo data processing tools."""
    print("\n" + "=" * 60)
    print("DATA PROCESSING DEMO")
    print("=" * 60)

    tools = get_tools_by_category("data")
    agent = Agent(
        tools=tools, provider=provider, config=AgentConfig(max_iterations=5, temperature=0.3)
    )

    # Test JSON parsing and CSV conversion
    sample_json = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
    response = agent.run(
        [
            Message(
                role=Role.USER,
                content=f"Parse this JSON and convert it to CSV format: {sample_json}",
            )
        ]
    )
    print(f"\nAgent: {response.content}")


def demo_text_processing() -> None:
    """Demo text processing tools."""
    print("\n" + "=" * 60)
    print("TEXT PROCESSING DEMO")
    print("=" * 60)

    tools = get_tools_by_category("text")
    agent = Agent(
        tools=tools, provider=provider, config=AgentConfig(max_iterations=5, temperature=0.3)
    )

    sample_text = """
    Contact us at: support@example.com or sales@example.com
    Visit our website: https://example.com
    Or call us at our office.
    """

    response = agent.run(
        [
            Message(
                role=Role.USER,
                content=f"Extract all email addresses and URLs from this text: {sample_text}",
            )
        ]
    )
    print(f"\nAgent: {response.content}")


def demo_datetime_utilities() -> None:
    """Demo datetime tools."""
    print("\n" + "=" * 60)
    print("DATETIME UTILITIES DEMO")
    print("=" * 60)

    tools = get_tools_by_category("datetime")
    agent = Agent(
        tools=tools, provider=provider, config=AgentConfig(max_iterations=5, temperature=0.3)
    )

    response = agent.run(
        [
            Message(
                role=Role.USER,
                content="What's the current time in UTC? Then calculate what date it will be 30 days from today.",
            )
        ]
    )
    print(f"\nAgent: {response.content}")


def demo_all_tools() -> None:
    """Demo using all tools together."""
    print("\n" + "=" * 60)
    print("ALL TOOLS DEMO - Multi-step Task")
    print("=" * 60)

    # Get all tools from toolbox
    all_tools = get_all_tools()
    print(f"Loaded {len(all_tools)} tools from toolbox")

    agent = Agent(
        tools=all_tools, provider=provider, config=AgentConfig(max_iterations=8, temperature=0.3)
    )

    # Complex multi-step task using multiple tool categories
    response = agent.run(
        [
            Message(
                role=Role.USER,
                content="""
                Perform these tasks:
                1. Get the current time in UTC
                2. Count the words in this sentence: "The quick brown fox jumps over the lazy dog"
                3. Create a JSON object with the time and word count, then format it as a table
                """,
            )
        ]
    )
    print(f"\nAgent: {response.content}")


def list_available_tools() -> None:
    """List all available tools in the toolbox."""
    print("\n" + "=" * 60)
    print("AVAILABLE TOOLS IN TOOLBOX")
    print("=" * 60)

    categories = ["file", "web", "data", "datetime", "text"]

    for category in categories:
        tools = get_tools_by_category(category)
        print(f"\n{category.upper()} TOOLS ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")


if __name__ == "__main__":
    print("Selectools Toolbox Demo")
    print("This demo showcases the pre-built tools available in selectools.toolbox")

    # List all available tools
    list_available_tools()

    # Run individual demos
    try:
        demo_file_operations()
    except Exception as e:
        print(f"File operations demo error: {e}")

    try:
        demo_data_processing()
    except Exception as e:
        print(f"Data processing demo error: {e}")

    try:
        demo_text_processing()
    except Exception as e:
        print(f"Text processing demo error: {e}")

    try:
        demo_datetime_utilities()
    except Exception as e:
        print(f"Datetime utilities demo error: {e}")

    try:
        demo_all_tools()
    except Exception as e:
        print(f"All tools demo error: {e}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
