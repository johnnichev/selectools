"""
Dynamic Tools — ToolLoader, plugin directories, hot-reload, runtime tool management.

Prerequisites: OPENAI_API_KEY (examples 01-05)
Run: python examples/13_dynamic_tools.py
"""

import os
import tempfile
from pathlib import Path

from selectools import Agent, AgentConfig, Message, OpenAIProvider, Role
from selectools.models import OpenAI
from selectools.providers.stubs import LocalProvider
from selectools.tools import ToolLoader, tool


# Base tool to include in the agent from the start
@tool(description="Get the current date in ISO format")
def get_current_date() -> str:
    """Return today's date as YYYY-MM-DD."""
    from datetime import date

    return str(date.today())


def main() -> None:
    """Run the dynamic tools demonstration."""

    print("=" * 80)
    print("🔧 Dynamic Tools Demo: ToolLoader & Agent Tool Management")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        plugins_dir = Path(tmp_dir) / "plugins"
        plugins_dir.mkdir()

        # -------------------------------------------------------------------------
        # Step 1: Create a temp plugin directory with tool files
        # -------------------------------------------------------------------------
        print("\n📁 Step 1: Creating temp plugin directory with tool files")
        print("-" * 80)

        (plugins_dir / "math_tools.py").write_text(
            '''
from selectools.tools import tool

@tool(name="add_numbers", description="Add two numbers together")
def add_numbers(a: float, b: float) -> str:
    """Add a and b."""
    return f"{a} + {b} = {a + b}"

@tool(name="multiply_numbers", description="Multiply two numbers")
def multiply_numbers(a: float, b: float) -> str:
    """Multiply a and b."""
    return f"{a} * {b} = {a * b}"
'''
        )

        (plugins_dir / "greeting_tools.py").write_text(
            '''
from selectools.tools import tool

@tool(name="greet_user", description="Greet a user by name")
def greet_user(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}! Nice to meet you."
'''
        )

        single_file = plugins_dir / "math_tools.py"
        print(f"   Created plugins/math_tools.py (add_numbers, multiply_numbers)")
        print(f"   Created plugins/greeting_tools.py (greet_user)")

        # -------------------------------------------------------------------------
        # Step 2: Load tools with ToolLoader.from_file()
        # -------------------------------------------------------------------------
        print("\n📄 Step 2: Load tools with ToolLoader.from_file()")
        print("-" * 80)
        file_tools = ToolLoader.from_file(str(single_file))
        print(f"   Loaded {len(file_tools)} tools from math_tools.py:")
        for t in file_tools:
            print(f"   - {t.name}: {t.description[:50]}...")
        add_tool = next(t for t in file_tools if t.name == "add_numbers")
        result = add_tool.execute({"a": 10.0, "b": 5.0})
        print(f"   Executed add_numbers(10, 5): {result}")

        # -------------------------------------------------------------------------
        # Step 3: Load tools with ToolLoader.from_directory()
        # -------------------------------------------------------------------------
        print("\n📂 Step 3: Load tools with ToolLoader.from_directory()")
        print("-" * 80)
        dir_tools = ToolLoader.from_directory(str(plugins_dir))
        print(f"   Loaded {len(dir_tools)} tools from plugins/:")
        for t in dir_tools:
            print(f"   - {t.name}")

        # -------------------------------------------------------------------------
        # Step 4: Create an agent with base tools
        # -------------------------------------------------------------------------
        print("\n🤖 Step 4: Create an agent with base tools")
        print("-" * 80)
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        provider = (
            OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id) if has_openai else LocalProvider()
        )
        agent = Agent(
            tools=[get_current_date],
            provider=provider,
            config=AgentConfig(max_iterations=5),
        )
        print("   Agent created with 1 base tool: get_current_date")
        if not has_openai:
            print("   (Using LocalProvider - set OPENAI_API_KEY for real LLM calls)")

        # -------------------------------------------------------------------------
        # Step 5: Dynamically add loaded tools to the agent
        # -------------------------------------------------------------------------
        print("\n➕ Step 5: Dynamically add loaded tools to the agent")
        print("-" * 80)
        agent.add_tools(dir_tools)
        print(f"   Added {len(dir_tools)} tools. Agent now has {len(agent.tools)} tools:")
        for t in agent.tools:
            print(f"   - {t.name}")

        # -------------------------------------------------------------------------
        # Step 6: Run the agent (show it can use new tools)
        # -------------------------------------------------------------------------
        print("\n▶️  Step 6: Run the agent with new tools")
        print("-" * 80)
        if has_openai:
            try:
                response = agent.run(
                    [
                        Message(
                            role=Role.USER,
                            content="Add 7 and 13, then greet the user named Alice.",
                        )
                    ]
                )
                print(f"   Response: {response.content[:200]}...")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            result = add_tool.execute({"a": 7.0, "b": 13.0})
            greet_tool = next(t for t in agent.tools if t.name == "greet_user")
            greet_result = greet_tool.execute({"name": "Alice"})
            print("   (Skipping agent.run - no API key. Demonstrating tools work directly:)")
            print(f"   add_numbers(7, 13): {result}")
            print(f"   greet_user(Alice): {greet_result}")

        # -------------------------------------------------------------------------
        # Step 7: Remove a tool and show agent adapts
        # -------------------------------------------------------------------------
        print("\n➖ Step 7: Remove a tool")
        print("-" * 80)
        removed = agent.remove_tool("multiply_numbers")
        print(f"   Removed tool: {removed.name}")
        print(f"   Agent now has {len(agent.tools)} tools: {[t.name for t in agent.tools]}")

        # -------------------------------------------------------------------------
        # Step 8: Replace a tool with updated version
        # -------------------------------------------------------------------------
        print("\n🔄 Step 8: Replace a tool with updated version")
        print("-" * 80)
        (plugins_dir / "greeting_tools.py").write_text(
            '''
from selectools.tools import tool

@tool(name="greet_user", description="Greet a user by name (enhanced version)")
def greet_user(name: str, formal: bool = False) -> str:
    """Return a friendly or formal greeting."""
    if formal:
        return f"Good day, {name}. It is a pleasure to make your acquaintance."
    return f"Hello, {name}! Nice to meet you."
'''
        )
        updated_tools = ToolLoader.from_file(str(plugins_dir / "greeting_tools.py"))
        updated_greet = next(t for t in updated_tools if t.name == "greet_user")
        old_tool = agent.replace_tool(updated_greet)
        print(f"   Replaced greet_user (old desc: {old_tool.description[:40]}...)")
        print(f"   New description: {updated_greet.description}")

        # -------------------------------------------------------------------------
        # Step 9: Hot-reload a modified plugin file
        # -------------------------------------------------------------------------
        print("\n🔥 Step 9: Hot-reload a modified plugin file")
        print("-" * 80)
        (plugins_dir / "math_tools.py").write_text(
            '''
from selectools.tools import tool

@tool(name="add_numbers", description="Add two numbers (v2: now with rounding)")
def add_numbers(a: float, b: float, round_result: bool = False) -> str:
    """Add a and b, optionally round to integer."""
    total = a + b
    if round_result:
        total = round(total)
    return f"{a} + {b} = {total}"
'''
        )
        reloaded = ToolLoader.reload_file(str(plugins_dir / "math_tools.py"))
        add_v2 = next(t for t in reloaded if t.name == "add_numbers")
        agent.replace_tool(add_v2)
        print("   Reloaded math_tools.py with updated add_numbers (now has round_result param)")
        exec_result = add_v2.execute({"a": 3.7, "b": 4.3, "round_result": True})
        print(f"   add_numbers(3.7, 4.3, round_result=True): {exec_result}")

    print("\n" + "=" * 80)
    print("✅ Dynamic Tools Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
