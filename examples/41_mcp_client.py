"""
Example 41: MCP Client — Connect to MCP Tool Servers
=====================================================

Use tools from any MCP-compatible server in your selectools agent.
Supports stdio (local subprocess) and Streamable HTTP (remote).

Requires: pip install selectools[mcp]

Usage:
    python examples/41_mcp_client.py

This example spawns a local MCP server as a subprocess.
No external API key needed for the MCP part.
"""

import asyncio
import os
import sys
import tempfile

# Write a simple MCP server for the demo
SERVER_SCRIPT = '''
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two numbers together."""
    return str(a + b)

@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Welcome to the MCP demo."

@mcp.tool()
def reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


async def main() -> None:
    try:
        from selectools.mcp import MCPClient, MCPServerConfig
    except ImportError:
        print("MCP support requires: pip install selectools[mcp]")
        sys.exit(1)

    # Write the server script to a temp file
    server_path = os.path.join(tempfile.mkdtemp(), "demo_server.py")
    with open(server_path, "w") as f:
        f.write(SERVER_SCRIPT)

    # --- Connect to MCP server ---
    print("Connecting to MCP server...")
    config = MCPServerConfig(
        command=sys.executable,  # Use the current Python
        args=[server_path],
        name="demo",
    )

    async with MCPClient(config) as client:
        # --- Discover tools ---
        tools = await client.list_tools()
        print(f"Discovered {len(tools)} tools:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")
            print(f"    params: {[p.name for p in t.parameters]}")
        print()

        # --- Call tools directly ---
        print("Direct tool calls:")
        result = await client._call_tool("add", {"a": 42, "b": 58})
        print(f"  add(42, 58) = {result}")

        result = await client._call_tool("greet", {"name": "Selectools"})
        print(f"  greet('Selectools') = {result}")

        result = await client._call_tool("reverse", {"text": "Hello MCP"})
        print(f"  reverse('Hello MCP') = {result}")
        print()

        # --- Use with selectools Agent ---
        print("Using MCP tools with selectools Agent:")
        try:
            from selectools import Agent, AgentConfig
            from selectools.providers.stubs import LocalProvider

            agent = Agent(
                provider=LocalProvider(),
                config=AgentConfig(model="local"),
                tools=tools,
            )
            print(f"  Agent created with {len(tools)} MCP tools")
            print(f"  Tool names: {[t.name for t in agent.tools]}")
        except Exception as e:
            print(f"  Agent creation: {e}")
        print()

        # --- Eval on MCP tools ---
        print("Evaluating MCP tools:")
        from selectools.evals import EvalSuite, TestCase

        suite = EvalSuite(
            agent=agent,
            cases=[
                TestCase(input="Add 10 and 20", name="add_test"),
                TestCase(input="Greet John", name="greet_test"),
            ],
            name="mcp-demo",
        )
        report = suite.run()
        print(f"  Accuracy: {report.accuracy:.0%}")
        print(f"  Cases: {report.metadata.total_cases}")

    os.unlink(server_path)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
