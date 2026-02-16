"""
Hello World — Your first Selectools agent.

No API key needed. Runs entirely offline with the built-in LocalProvider.

Prerequisites: None
    pip install selectools

Run:
    python examples/01_hello_world.py
"""

from selectools import Agent, AgentConfig, tool
from selectools.providers.stubs import LocalProvider


@tool(description="Look up the price of a product")
def get_price(product: str) -> str:
    prices = {"laptop": "$999", "phone": "$699", "headphones": "$149"}
    return prices.get(product.lower(), f"No price found for {product}")


@tool(description="Check if a product is in stock")
def check_stock(product: str) -> str:
    stock = {"laptop": "5 left", "phone": "Out of stock", "headphones": "20 left"}
    return stock.get(product.lower(), f"Unknown product: {product}")


def main() -> None:
    agent = Agent(
        tools=[get_price, check_stock],
        provider=LocalProvider(),
        config=AgentConfig(max_iterations=3),
    )

    print("Agent created with 2 tools: get_price, check_stock\n")

    result = agent.ask("How much is a laptop?")
    print(f"Response: {result.content}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool calls: {len(result.tool_calls)}")


if __name__ == "__main__":
    main()
