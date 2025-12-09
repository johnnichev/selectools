"""
Ollama Local Model Demo

This example demonstrates using selectools with local Ollama models for:
- Privacy-preserving agent execution (no data sent to cloud)
- Zero-cost inference (no API fees)
- Offline development and testing

Prerequisites:
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Start Ollama server: `ollama serve`

Run this demo:
    python examples/ollama_demo.py
"""

from selectools import Agent, AgentConfig, Message, Role, Tool, ToolParameter
from selectools.providers import OllamaProvider

# ========================
# 1. Define Tools
# ========================


def search_web(query: str) -> str:
    """
    Search the web for information.
    (Mock implementation for demo purposes)
    """
    # In a real application, you would call a search API here
    mock_results = {
        "python": "Python is a high-level programming language known for its simplicity...",
        "ollama": "Ollama is a tool for running large language models locally...",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn...",
    }

    for keyword in mock_results:
        if keyword.lower() in query.lower():
            return f"Search results for '{query}':\n{mock_results[keyword]}"

    return f"Search results for '{query}':\nNo specific results found."


def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """
    try:
        result = eval(expression)  # noqa: S307 (safe for demo)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {e}"


def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    (Mock implementation for demo purposes)
    """
    # In a real application, you would call a weather API here
    mock_weather = {
        "san francisco": "Partly cloudy, 18°C (64°F), Light breeze",
        "new york": "Sunny, 22°C (72°F), Moderate wind",
        "london": "Rainy, 12°C (54°F), Strong wind",
        "tokyo": "Clear, 25°C (77°F), Calm",
    }

    city_lower = city.lower()
    for location in mock_weather:
        if location in city_lower:
            return f"Weather in {city.title()}: {mock_weather[location]}"

    return f"Weather in {city.title()}: Data not available (mock API)"


# ========================
# 2. Create Tool Instances
# ========================

search_tool = Tool(
    name="search_web",
    description="Search the web for information about any topic",
    parameters=[
        ToolParameter(
            name="query",
            param_type=str,
            description="Search query (e.g., 'Python programming', 'climate change')",
        )
    ],
    function=search_web,
)

calculator_tool = Tool(
    name="calculate",
    description="Calculate mathematical expressions",
    parameters=[
        ToolParameter(
            name="expression",
            param_type=str,
            description="Math expression to evaluate (e.g., '2+2', '(10*5)/2')",
        )
    ],
    function=calculate,
)

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a specific city",
    parameters=[
        ToolParameter(name="city", param_type=str, description="City name (e.g., 'San Francisco')")
    ],
    function=get_weather,
)

# ========================
# 3. Configure Ollama Provider
# ========================

print("🦙 Ollama Local Model Demo")
print("=" * 60)
print()

try:
    # Initialize Ollama provider
    # You can use different models: llama3.2, llama3.1, mistral, codellama, etc.
    provider = OllamaProvider(
        model="llama3.2",  # Change to your preferred model
        base_url="http://localhost:11434",  # Default Ollama URL
        temperature=0.7,
    )

    print(f"✅ Connected to Ollama")
    print(f"   Model: llama3.2")
    print(f"   URL: http://localhost:11434")
    print()

except Exception as e:
    print(f"❌ Failed to connect to Ollama: {e}")
    print()
    print("Make sure Ollama is running:")
    print("  1. Install: https://ollama.ai")
    print("  2. Pull model: ollama pull llama3.2")
    print("  3. Start server: ollama serve")
    exit(1)

# ========================
# 4. Create Agent
# ========================

config = AgentConfig(
    model="llama3.2",
    max_iterations=6,
    verbose=True,  # Show execution details
    temperature=0.7,
)

agent = Agent(
    tools=[search_tool, calculator_tool, weather_tool],
    provider=provider,
    config=config,
)

# ========================
# 5. Run Example Queries
# ========================

print("🤖 Running example queries...")
print()

# Example 1: Search and calculate
print("━" * 60)
print("📝 Query 1: Search + Calculate")
print("━" * 60)

response = agent.run(
    [
        Message(
            role=Role.USER,
            content="Search for information about Python and calculate 15 * 23",
        )
    ]
)

print(f"\n💬 Response: {response.content}")
print()

# Example 2: Weather query
print("━" * 60)
print("📝 Query 2: Weather")
print("━" * 60)

response = agent.run([Message(role=Role.USER, content="What's the weather like in Tokyo?")])

print(f"\n💬 Response: {response.content}")
print()

# Example 3: Complex multi-step query
print("━" * 60)
print("📝 Query 3: Multi-step")
print("━" * 60)

response = agent.run(
    [
        Message(
            role=Role.USER,
            content="Search for Ollama, get the weather in London, and calculate 100 / 4",
        )
    ]
)

print(f"\n💬 Response: {response.content}")
print()

# ========================
# 6. Show Cost Savings
# ========================

print("━" * 60)
print("💰 Cost Analysis")
print("━" * 60)

print(f"Total API Cost: ${agent.total_cost:.6f} (FREE!)")
print(f"Total Tokens: {agent.total_tokens:,}")
print()
print("Benefits of using Ollama:")
print("  ✅ Zero API costs")
print("  ✅ Complete privacy (no data sent to cloud)")
print("  ✅ Works offline")
print("  ✅ Full control over model and hardware")
print("  ✅ Great for development and testing")
print()

# ========================
# 7. Model Comparison
# ========================

print("━" * 60)
print("🔄 Available Ollama Models")
print("━" * 60)

models = [
    ("llama3.2", "Small, fast, good for general tasks"),
    ("llama3.1", "Larger, more capable, slightly slower"),
    ("mistral", "Strong performance, efficient"),
    ("codellama", "Specialized for code generation"),
    ("phi", "Tiny but capable, very fast"),
    ("qwen", "Good multilingual support"),
]

print("\nTo use a different model:")
print("  1. Pull it: ollama pull <model_name>")
print("  2. Change provider: OllamaProvider(model='<model_name>')")
print()

for model, description in models:
    print(f"  • {model:15} - {description}")

print()
print("=" * 60)
print("✨ Demo complete!")
