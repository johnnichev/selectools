"""
Customer Support Bot Example

This example demonstrates how to build a production-ready customer support bot using Selectools.

Features demonstrated:
- Multiple tools for common support tasks (order lookup, refunds, knowledge base)
- Conversation memory for maintaining context across multiple turns
- Async support for handling concurrent requests
- Error handling and graceful degradation
- Realistic mock data for testing

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="sk-..."

    # Run the example
    python examples/customer_support_bot.py

Requirements:
    pip install selectools
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from selectools import (
    Agent,
    AgentConfig,
    ConversationMemory,
    Message,
    OpenAIProvider,
    Role,
    Tool,
    ToolParameter,
)

# ============================================================================
# Mock Database (In production, replace with real database queries)
# ============================================================================

ORDERS_DB: Dict[str, Dict[str, Any]] = {
    "ORD-12345": {
        "customer_email": "john@example.com",
        "status": "shipped",
        "items": ["Laptop Pro 15", "USB-C Cable"],
        "total": 1299.99,
        "order_date": "2024-12-01",
        "tracking_number": "1Z999AA10123456784",
        "estimated_delivery": "2024-12-10",
    },
    "ORD-67890": {
        "customer_email": "jane@example.com",
        "status": "processing",
        "items": ["Wireless Mouse", "Keyboard"],
        "total": 89.99,
        "order_date": "2024-12-05",
        "tracking_number": None,
        "estimated_delivery": "2024-12-12",
    },
    "ORD-11111": {
        "customer_email": "john@example.com",
        "status": "delivered",
        "items": ["Phone Case"],
        "total": 24.99,
        "order_date": "2024-11-20",
        "tracking_number": "1Z999AA10123456785",
        "estimated_delivery": "2024-11-25",
    },
}

KNOWLEDGE_BASE: Dict[str, List[str]] = {
    "shipping": [
        "Standard shipping takes 5-7 business days",
        "Express shipping takes 2-3 business days",
        "Free shipping on orders over $50",
        "International shipping available to 150+ countries",
    ],
    "returns": [
        "30-day money-back guarantee on all products",
        "Items must be in original condition with tags attached",
        "Refunds processed within 5-7 business days",
        "Return shipping is free for defective items",
    ],
    "warranty": [
        "1-year manufacturer warranty on all electronics",
        "Extended warranty available for purchase",
        "Warranty covers manufacturing defects only",
        "Register your product within 30 days for warranty",
    ],
    "payment": [
        "We accept Visa, Mastercard, Amex, and PayPal",
        "Payment is processed securely via Stripe",
        "Monthly payment plans available on orders over $500",
        "Gift cards and promo codes can be applied at checkout",
    ],
}

REFUND_REQUESTS: List[Dict[str, Any]] = []  # Track refund requests


# ============================================================================
# Support Tools
# ============================================================================


def check_order_status(order_id: str) -> str:
    """
    Check the status of a customer order.

    Args:
        order_id: The order ID (e.g., ORD-12345)

    Returns:
        Order status information including tracking, items, and delivery date
    """
    order_id = order_id.upper().strip()

    if order_id not in ORDERS_DB:
        return f"Order {order_id} not found. Please verify the order ID and try again."

    order = ORDERS_DB[order_id]

    result = f"Order {order_id}:\n"
    result += f"Status: {str(order['status']).title()}\n"
    result += f"Order Date: {order['order_date']}\n"
    result += f"Items: {', '.join(str(item) for item in order['items'])}\n"
    result += f"Total: ${float(order['total']):.2f}\n"

    if order["tracking_number"]:
        result += f"Tracking: {order['tracking_number']}\n"

    if order["status"] != "delivered":
        result += f"Estimated Delivery: {order['estimated_delivery']}"

    return result


def process_refund(order_id: str, reason: str) -> str:
    """
    Process a refund request for an order.

    Args:
        order_id: The order ID to refund
        reason: The reason for the refund

    Returns:
        Confirmation message with refund ticket number
    """
    order_id = order_id.upper().strip()

    if order_id not in ORDERS_DB:
        return f"Order {order_id} not found. Cannot process refund."

    order = ORDERS_DB[order_id]

    # Check if order is eligible for refund
    order_date = datetime.strptime(str(order["order_date"]), "%Y-%m-%d")
    days_since_order = (datetime.now() - order_date).days

    if days_since_order > 30:
        return f"Order {order_id} is outside the 30-day return window. Please contact support for assistance."

    if order["status"] == "processing":
        return f"Order {order_id} is still processing. You can cancel it instead for an immediate refund."

    # Create refund request
    ticket_number = f"REF-{len(REFUND_REQUESTS) + 1000:05d}"
    REFUND_REQUESTS.append(
        {
            "ticket": ticket_number,
            "order_id": order_id,
            "reason": reason,
            "amount": order["total"],
            "status": "pending",
        }
    )

    result = f"Refund request created successfully!\n\n"
    result += f"Ticket Number: {ticket_number}\n"
    result += f"Order ID: {order_id}\n"
    result += f"Refund Amount: ${order['total']:.2f}\n"
    result += f"Status: Pending review\n\n"
    result += "Your refund will be processed within 5-7 business days. "
    result += f"You'll receive a confirmation email at {order['customer_email']}."

    return result


def update_shipping_address(order_id: str, new_address: str) -> str:
    """
    Update the shipping address for an order.

    Args:
        order_id: The order ID to update
        new_address: The new shipping address

    Returns:
        Confirmation message
    """
    order_id = order_id.upper().strip()

    if order_id not in ORDERS_DB:
        return f"Order {order_id} not found. Cannot update address."

    order = ORDERS_DB[order_id]

    if order["status"] == "delivered":
        return f"Order {order_id} has already been delivered. Cannot update address."

    if order["status"] == "shipped":
        return (
            f"Order {order_id} has already shipped. "
            "Please contact the carrier directly to request an address change: "
            f"Tracking: {order['tracking_number']}"
        )

    # Update address (in real app, this would update the database)
    result = f"Shipping address updated successfully for order {order_id}!\n\n"
    result += f"New Address: {new_address}\n"
    result += f"Order Status: {str(order['status']).title()}\n"
    result += f"Estimated Delivery: {order['estimated_delivery']}"

    return result


def search_knowledge_base(topic: str) -> str:
    """
    Search the knowledge base for information on a specific topic.

    Args:
        topic: The topic to search for (shipping, returns, warranty, payment)

    Returns:
        Relevant information from the knowledge base
    """
    topic = topic.lower().strip()

    # Fuzzy matching for common variations
    topic_map = {
        "ship": "shipping",
        "delivery": "shipping",
        "return": "returns",
        "refund": "returns",
        "warranties": "warranty",
        "pay": "payment",
        "payments": "payment",
        "billing": "payment",
    }

    topic = topic_map.get(topic, topic)

    if topic not in KNOWLEDGE_BASE:
        return (
            f"No information found for '{topic}'. "
            f"Available topics: {', '.join(KNOWLEDGE_BASE.keys())}"
        )

    result = f"Information about {topic.title()}:\n\n"
    for idx, info in enumerate(KNOWLEDGE_BASE[topic], 1):
        result += f"{idx}. {info}\n"

    return result


def escalate_to_human(issue_summary: str, priority: str = "normal") -> str:
    """
    Escalate a complex issue to a human support agent.

    Args:
        issue_summary: Brief summary of the issue
        priority: Priority level (low, normal, high, urgent)

    Returns:
        Ticket information for the escalated issue
    """
    priority = priority.lower()
    if priority not in ["low", "normal", "high", "urgent"]:
        priority = "normal"

    ticket_number = f"SUP-{len(REFUND_REQUESTS) + 2000:05d}"

    result = f"Issue escalated to human support agent.\n\n"
    result += f"Support Ticket: {ticket_number}\n"
    result += f"Priority: {priority.title()}\n"
    result += f"Issue: {issue_summary}\n\n"

    if priority in ["high", "urgent"]:
        result += "A senior support agent will contact you within 1 hour.\n"
    else:
        result += "A support agent will contact you within 24 hours.\n"

    result += "You can check your ticket status at support.example.com"

    return result


# ============================================================================
# Create Support Agent
# ============================================================================


def create_support_agent() -> Agent:
    """Create a customer support agent with all necessary tools."""

    # Define tools
    tools = [
        Tool(
            name="check_order_status",
            description="Check the status, tracking, and delivery information for a customer order. Use this when customers ask about their order.",
            parameters=[
                ToolParameter(
                    name="order_id",
                    type="string",
                    description="The order ID (format: ORD-XXXXX)",
                    required=True,
                )
            ],
            function=check_order_status,
        ),
        Tool(
            name="process_refund",
            description="Process a refund request for an order. Use this when customers want to return items or request a refund.",
            parameters=[
                ToolParameter(
                    name="order_id",
                    type="string",
                    description="The order ID to refund",
                    required=True,
                ),
                ToolParameter(
                    name="reason",
                    type="string",
                    description="The customer's reason for requesting a refund",
                    required=True,
                ),
            ],
            function=process_refund,
        ),
        Tool(
            name="update_shipping_address",
            description="Update the shipping address for an order that hasn't shipped yet. Use this when customers need to change their delivery address.",
            parameters=[
                ToolParameter(
                    name="order_id",
                    type="string",
                    description="The order ID to update",
                    required=True,
                ),
                ToolParameter(
                    name="new_address",
                    type="string",
                    description="The new shipping address",
                    required=True,
                ),
            ],
            function=update_shipping_address,
        ),
        Tool(
            name="search_knowledge_base",
            description="Search the knowledge base for information about policies and procedures. Topics: shipping, returns, warranty, payment.",
            parameters=[
                ToolParameter(
                    name="topic",
                    type="string",
                    description="The topic to search for (shipping, returns, warranty, payment)",
                    required=True,
                )
            ],
            function=search_knowledge_base,
        ),
        Tool(
            name="escalate_to_human",
            description="Escalate complex issues that require human judgment or are outside your capabilities. Use sparingly - try to resolve issues yourself first.",
            parameters=[
                ToolParameter(
                    name="issue_summary",
                    type="string",
                    description="Brief summary of the issue that needs human attention",
                    required=True,
                ),
                ToolParameter(
                    name="priority",
                    type="string",
                    description="Priority level: low, normal, high, or urgent",
                    required=False,
                ),
            ],
            function=escalate_to_human,
        ),
    ]

    # Create provider
    provider = OpenAIProvider(default_model="gpt-4o-mini")

    # Create conversation memory (maintain context across turns)
    memory = ConversationMemory(max_messages=20)

    # Configure agent
    config = AgentConfig(
        max_iterations=5,
        verbose=True,
        temperature=0.7,  # Slightly warm for friendly responses
        max_tokens=500,
    )

    # System prompt for support agent personality
    system_prompt = """You are a helpful and empathetic customer support agent for an e-commerce company.

Your responsibilities:
- Help customers with order tracking, refunds, and general inquiries
- Be friendly, professional, and understanding
- Use the available tools to look up information and take actions
- Always confirm details before processing refunds or updates
- Escalate to human agents only when absolutely necessary

Important guidelines:
- Always ask for the order ID if the customer doesn't provide it
- Verify customer information before processing sensitive requests
- Be apologetic when issues occur and proactive about solutions
- Keep responses concise but warm
- Never make promises outside of company policies

Remember: You're representing the company, so maintain professionalism while being helpful and human."""

    return Agent(
        tools=tools,
        provider=provider,
        memory=memory,
        config=config,
    )


# ============================================================================
# Example Conversations
# ============================================================================


def example_order_tracking():
    """Example: Customer checks order status."""
    print("=" * 70)
    print("EXAMPLE 1: Order Tracking")
    print("=" * 70)

    agent = create_support_agent()

    # Initial question
    response = agent.run(
        [Message(role=Role.USER, content="Hi! I'd like to check the status of my order ORD-12345")]
    )

    print(f"\n🤖 Agent: {response.content}\n")

    # Follow-up question (using memory)
    response = agent.run([Message(role=Role.USER, content="When will it arrive?")])

    print(f"🤖 Agent: {response.content}\n")


def example_refund_request():
    """Example: Customer requests a refund."""
    print("=" * 70)
    print("EXAMPLE 2: Refund Request")
    print("=" * 70)

    agent = create_support_agent()

    response = agent.run(
        [
            Message(
                role=Role.USER,
                content="I need to return my order ORD-11111. The phone case doesn't fit my phone.",
            )
        ]
    )

    print(f"\n🤖 Agent: {response.content}\n")


def example_knowledge_base():
    """Example: Customer asks about shipping policy."""
    print("=" * 70)
    print("EXAMPLE 3: Policy Question")
    print("=" * 70)

    agent = create_support_agent()

    response = agent.run(
        [
            Message(
                role=Role.USER, content="What's your shipping policy? How long does delivery take?"
            )
        ]
    )

    print(f"\n🤖 Agent: {response.content}\n")


async def example_concurrent_requests():
    """Example: Handle multiple customer requests concurrently."""
    print("=" * 70)
    print("EXAMPLE 4: Concurrent Requests (Async)")
    print("=" * 70)

    # Create multiple agents for concurrent users
    agents = [create_support_agent() for _ in range(3)]

    questions = [
        "Check status of order ORD-12345",
        "I want to return order ORD-67890, it's not what I expected",
        "What's your warranty policy?",
    ]

    async def handle_request(agent: Agent, question: str) -> str:
        """Handle a single customer request asynchronously."""
        response = await agent.arun([Message(role=Role.USER, content=question)])
        return str(response.content)

    # Process all requests concurrently
    print("\n📨 Processing 3 customer requests concurrently...\n")

    tasks = [handle_request(agent, question) for agent, question in zip(agents, questions)]

    responses = await asyncio.gather(*tasks)

    for idx, (question, response) in enumerate(zip(questions, responses), 1):
        print(f"Customer {idx}: {question}")
        print(f"🤖 Agent: {response}\n")


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all customer support examples."""
    print("\n🎯 Customer Support Bot Examples\n")

    # Synchronous examples
    example_order_tracking()
    print("\n")

    example_refund_request()
    print("\n")

    example_knowledge_base()
    print("\n")

    # Async example
    asyncio.run(example_concurrent_requests())

    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
