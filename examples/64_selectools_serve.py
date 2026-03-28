"""
Example 64: Serving an Agent over HTTP

Demonstrates the `selectools serve` pattern (v0.19.0):
- Create an agent with tools
- Wrap it in an AgentServer
- Print the available endpoints
- Show how requests would be handled

This example sets up the server but does NOT start it
(no blocking call). It shows the programmatic API for
embedding agent serving in your own application.

Uses LocalProvider so no API keys are needed.

Prerequisites:
    pip install selectools

Run:
    python examples/64_selectools_serve.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from selectools import Agent, AgentConfig, tool
from selectools.providers.stubs import LocalProvider

# --- Tools for the served agent ---


@tool(description="Look up a customer by email")
def lookup_customer(email: str) -> str:
    """Find customer record by email address."""
    db = {
        "alice@example.com": "Alice Smith, Premium Plan, joined 2023",
        "bob@example.com": "Bob Jones, Free Plan, joined 2024",
    }
    return db.get(email, f"No customer found with email: {email}")


@tool(description="Check the status of an order")
def order_status(order_id: str) -> str:
    """Get current status for an order."""
    statuses = {
        "ORD-001": "Shipped, arriving March 28",
        "ORD-002": "Processing, expected ship date March 30",
    }
    return statuses.get(order_id, f"Order {order_id} not found")


# --- Minimal AgentServer skeleton ---
# In v0.19.0 this ships as selectools.serve.AgentServer; here we define
# a lightweight stand-in to demonstrate the API surface.


@dataclass
class Endpoint:
    """Describes one HTTP endpoint."""

    method: str
    path: str
    description: str


@dataclass
class AgentServer:
    """Wraps an Agent as an HTTP service.

    In production this is backed by FastAPI/Flask; here we show the
    configuration surface without starting a real server.
    """

    agent: Agent
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "Selectools Agent API"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key: Optional[str] = None
    _endpoints: List[Endpoint] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._register_endpoints()

    def _register_endpoints(self) -> None:
        self._endpoints = [
            Endpoint("POST", "/v1/chat", "Send a message and get a response"),
            Endpoint("POST", "/v1/chat/stream", "Stream a response (SSE)"),
            Endpoint("GET", "/v1/tools", "List available tools"),
            Endpoint("GET", "/v1/health", "Health check"),
            Endpoint("GET", "/v1/usage", "Current token/cost usage"),
            Endpoint("POST", "/v1/reset", "Reset conversation memory"),
        ]

    @property
    def endpoints(self) -> List[Endpoint]:
        return list(self._endpoints)

    def handle_chat(self, user_message: str) -> Dict[str, Any]:
        """Simulate handling a /v1/chat request."""
        result = self.agent.run(user_message)
        return {
            "content": result.content,
            "iterations": result.iterations,
            "tool_calls": len(result.tool_calls),
        }

    def handle_list_tools(self) -> List[Dict[str, str]]:
        """Simulate handling a /v1/tools request."""
        return [{"name": t.name, "description": t.description} for t in self.agent.tools]

    def handle_health(self) -> Dict[str, str]:
        """Simulate handling a /v1/health request."""
        return {"status": "healthy", "agent": self.agent.config.name}

    def serve(self) -> None:
        """Start the HTTP server (not called in this example)."""
        print(f"Starting server on {self.host}:{self.port}...")
        # In production: uvicorn.run(app, host=self.host, port=self.port)
        raise NotImplementedError("Full server requires selectools[serve]")


def main() -> None:
    print("=" * 60)
    print("Agent Server Demo")
    print("=" * 60)

    # --- Step 1: Create the agent ---
    agent = Agent(
        tools=[lookup_customer, order_status],
        provider=LocalProvider(),
        config=AgentConfig(
            name="support-bot",
            model="gpt-5-mini",
            max_iterations=5,
            system_prompt="You are a customer support agent. Use tools to help customers.",
        ),
    )
    print(f"\n1. Created agent: {agent.config.name}")
    print(f"   Tools: {[t.name for t in agent.tools]}")

    # --- Step 2: Create the server ---
    server = AgentServer(
        agent=agent,
        host="0.0.0.0",
        port=8080,
        title="Support Bot API",
        api_key="sk-demo-key-12345",
    )
    print(f"\n2. Created AgentServer on {server.host}:{server.port}")

    # --- Step 3: Print endpoints ---
    print(f"\n3. Available endpoints:")
    for ep in server.endpoints:
        print(f"   {ep.method:6s} {ep.path:25s}  {ep.description}")

    # --- Step 4: Simulate requests ---
    print(f"\n4. Simulating requests:")

    health = server.handle_health()
    print(f"\n   GET /v1/health")
    print(f"   -> {health}")

    tools_list = server.handle_list_tools()
    print(f"\n   GET /v1/tools")
    for t in tools_list:
        print(f"   -> {t['name']}: {t['description']}")

    chat_response = server.handle_chat("What's the status of ORD-001?")
    print(f"\n   POST /v1/chat")
    print(f"   -> content: {chat_response['content'][:80]}...")
    print(f"   -> iterations: {chat_response['iterations']}")

    print(f"\n5. To start the real server:")
    print(f"   server.serve()  # requires selectools[serve] extras")
    print(f"   # Or from CLI: selectools serve --config agent.yaml --port 8080")

    print("\nDone! Server configured with 6 endpoints.")


if __name__ == "__main__":
    main()
