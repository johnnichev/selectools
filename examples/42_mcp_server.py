"""
Example 42: MCP Server — Expose Selectools Tools as MCP
========================================================

Turn any selectools @tool function into an MCP-compliant server.
Other MCP clients (Claude Desktop, Cursor, VS Code, other agents)
can discover and call your tools.

Requires: pip install selectools[mcp]

Usage:
    python examples/42_mcp_server.py

This starts an MCP server on stdio transport.
"""

import sys

try:
    from selectools import tool
    from selectools.mcp import MCPServer
except ImportError:
    print("MCP support requires: pip install selectools[mcp]")
    sys.exit(1)


# --- Define selectools tools ---


@tool(description="Get the current weather for a city")
def get_weather(city: str) -> str:
    """Simulated weather lookup."""
    weather_data = {
        "new york": "72°F, sunny",
        "london": "55°F, cloudy",
        "tokyo": "68°F, partly cloudy",
        "paris": "63°F, light rain",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool(description="Convert temperature between Fahrenheit and Celsius")
def convert_temp(value: float, from_unit: str) -> str:
    """Convert between F and C."""
    if from_unit.upper() == "F":
        celsius = (value - 32) * 5 / 9
        return f"{value}°F = {celsius:.1f}°C"
    elif from_unit.upper() == "C":
        fahrenheit = value * 9 / 5 + 32
        return f"{value}°C = {fahrenheit:.1f}°F"
    return f"Unknown unit: {from_unit}. Use 'F' or 'C'."


@tool(description="Search the knowledge base for information")
def search(query: str) -> str:
    """Simulated knowledge base search."""
    return f"Found 3 results for '{query}': [Result 1, Result 2, Result 3]"


# --- Create and run MCP server ---

if __name__ == "__main__":
    print("Starting selectools MCP server...", file=sys.stderr)
    print(f"Exposing {3} tools: get_weather, convert_temp, search", file=sys.stderr)
    print("Transport: stdio", file=sys.stderr)
    print("", file=sys.stderr)
    print("To connect from another selectools agent:", file=sys.stderr)
    print("  from selectools.mcp import mcp_tools, MCPServerConfig", file=sys.stderr)
    print(f"  config = MCPServerConfig(command='{sys.executable}',", file=sys.stderr)
    print(f"                           args=['{__file__}'])", file=sys.stderr)
    print("  with mcp_tools(config) as tools:", file=sys.stderr)
    print("      agent = Agent(provider=p, tools=tools, config=c)", file=sys.stderr)
    print("", file=sys.stderr)

    server = MCPServer(
        tools=[get_weather, convert_temp, search],
        name="selectools-demo",
    )
    server.serve(transport="stdio")
