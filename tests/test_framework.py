"""
Test Suite for AI Tool Calling Framework

This script tests the core functionality of the framework without requiring API calls.
It validates the implementation of classes, validation logic, and error handling.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Role, Message, ToolParameter, Tool, Agent
import json


def test_role_enum():
    """Test Role enum values."""
    print("\n" + "="*60)
    print("TEST: Role Enum")
    print("="*60)
    
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.SYSTEM.value == "system"
    assert Role.TOOL.value == "tool"
    
    print("✓ All role values correct")


def test_message_creation():
    """Test Message class creation and formatting."""
    print("\n" + "="*60)
    print("TEST: Message Creation")
    print("="*60)
    
    msg = Message(role=Role.USER, content="Hello")
    assert msg.role == Role.USER
    assert msg.content == "Hello"
    assert msg.image_path is None
    
    formatted = msg.to_openai_format()
    assert formatted["role"] == "user"
    assert formatted["content"] == "Hello"
    
    print("✓ Message creation and formatting works")


def test_tool_parameter_schema():
    """Test ToolParameter schema generation."""
    print("\n" + "="*60)
    print("TEST: ToolParameter Schema")
    print("="*60)
    
    param = ToolParameter(
        name="query",
        param_type=str,
        description="Search query",
        required=True
    )
    
    schema = param.to_schema()
    assert schema["type"] == "string"
    assert schema["description"] == "Search query"
    assert schema["required"] is True
    
    int_param = ToolParameter(name="count", param_type=int, description="Count", required=False)
    int_schema = int_param.to_schema()
    assert int_schema["type"] == "integer"
    assert int_schema["required"] is False
    
    print("✓ Parameter schema generation works")


def test_tool_validation():
    """Test Tool parameter validation."""
    print("\n" + "="*60)
    print("TEST: Tool Parameter Validation")
    print("="*60)
    
    def dummy_func(query: str, count: int) -> str:
        return f"{query}: {count}"
    
    tool = Tool(
        name="test_tool",
        description="Test tool",
        parameters=[
            ToolParameter(name="query", param_type=str, description="Query", required=True),
            ToolParameter(name="count", param_type=int, description="Count", required=True)
        ],
        function=dummy_func
    )
    
    valid, error = tool.validate_parameters({"query": "test", "count": 5})
    assert valid is True
    assert error is None
    print("✓ Valid parameters accepted")
    
    valid, error = tool.validate_parameters({"query": "test"})
    assert valid is False
    assert "Missing required parameter: count" in error
    print("✓ Missing required parameter detected")
    
    valid, error = tool.validate_parameters({"query": "test", "count": "not_an_int"})
    assert valid is False
    assert "must be of type int" in error
    print("✓ Type mismatch detected")


def test_tool_execution():
    """Test Tool execution."""
    print("\n" + "="*60)
    print("TEST: Tool Execution")
    print("="*60)
    
    def add_numbers(num1: float, num2: float) -> str:
        result = num1 + num2
        return json.dumps({"result": result})
    
    tool = Tool(
        name="add",
        description="Add two numbers",
        parameters=[
            ToolParameter(name="num1", param_type=float, description="First number", required=True),
            ToolParameter(name="num2", param_type=float, description="Second number", required=True)
        ],
        function=add_numbers
    )
    
    result = tool.execute({"num1": 5.0, "num2": 3.0})
    result_data = json.loads(result)
    assert result_data["result"] == 8.0
    print("✓ Tool execution works correctly")
    
    try:
        tool.execute({"num1": 5.0})
        assert False, "Should have raised ValueError"
    except ValueError as error:
        assert "Missing required parameter" in str(error)
        print("✓ Execution with invalid parameters raises error")


def test_tool_schema_generation():
    """Test Tool schema generation for LLM."""
    print("\n" + "="*60)
    print("TEST: Tool Schema Generation")
    print("="*60)
    
    def dummy_func(query: str) -> str:
        return query
    
    tool = Tool(
        name="search",
        description="Search the web",
        parameters=[
            ToolParameter(name="query", param_type=str, description="Search query", required=True)
        ],
        function=dummy_func
    )
    
    schema = tool.to_schema()
    assert schema["name"] == "search"
    assert schema["description"] == "Search the web"
    assert "query" in schema["parameters"]
    assert schema["parameters"]["query"]["type"] == "string"
    
    print("✓ Tool schema generation works")


def test_agent_tool_call_parsing():
    """Test Agent's tool call parsing logic."""
    print("\n" + "="*60)
    print("TEST: Tool Call Parsing")
    print("="*60)
    
    def dummy_func(query: str) -> str:
        return query
    
    tool = Tool(
        name="search",
        description="Search",
        parameters=[ToolParameter(name="query", param_type=str, description="Query", required=True)],
        function=dummy_func
    )
    
    agent = Agent(tools=[tool], max_iterations=1, api_key="test_key")
    
    response1 = 'I will search for that. TOOL_CALL: {"tool_name": "search", "parameters": {"query": "test"}}'
    parsed = agent._parse_tool_call(response1)
    assert parsed is not None
    tool_name, params = parsed
    assert tool_name == "search"
    assert params["query"] == "test"
    print("✓ Tool call parsing works")
    
    response2 = "This is just a regular response without a tool call."
    parsed = agent._parse_tool_call(response2)
    assert parsed is None
    print("✓ Non-tool responses correctly identified")
    
    response3 = 'TOOL_CALL: {"tool_name": "search", "parameters": {"query": "multi line\nquery"}}'
    parsed = agent._parse_tool_call(response3)
    assert parsed is not None
    print("✓ Multi-line parameters handled")


def test_agent_system_prompt():
    """Test Agent system prompt generation."""
    print("\n" + "="*60)
    print("TEST: System Prompt Generation")
    print("="*60)
    
    def dummy_func(query: str) -> str:
        return query
    
    tool = Tool(
        name="search",
        description="Search the web",
        parameters=[ToolParameter(name="query", param_type=str, description="Query", required=True)],
        function=dummy_func
    )
    
    agent = Agent(tools=[tool], api_key="test_key")
    system_prompt = agent._build_system_prompt()
    
    assert "TOOL_CALL" in system_prompt
    assert "search" in system_prompt
    assert "Search the web" in system_prompt
    assert "query" in system_prompt
    
    print("✓ System prompt includes tool definitions")


def test_message_with_image():
    """Test Message with image encoding."""
    print("\n" + "="*60)
    print("TEST: Message with Image")
    print("="*60)
    
    try:
        image_path = Path(__file__).resolve().parents[1] / "assets" / "dog.png"
        msg = Message(role=Role.USER, content="What's in this image?", image_path=str(image_path))
        assert msg.image_base64 is not None
        assert len(msg.image_base64) > 0
        
        formatted = msg.to_openai_format()
        assert isinstance(formatted["content"], list)
        assert formatted["content"][0]["type"] == "text"
        assert formatted["content"][1]["type"] == "image_url"
        
        print("✓ Image encoding and formatting works")
    except FileNotFoundError:
        print("⚠ Skipped (dog.png not found)")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("AI TOOL CALLING FRAMEWORK - TEST SUITE")
    print("="*80)
    
    tests = [
        test_role_enum,
        test_message_creation,
        test_tool_parameter_schema,
        test_tool_validation,
        test_tool_execution,
        test_tool_schema_generation,
        test_agent_tool_call_parsing,
        test_agent_system_prompt,
        test_message_with_image
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as error:
            print(f"\n✗ Test failed: {test_func.__name__}")
            print(f"  Error: {str(error)}")
            failed += 1
        except Exception as error:
            print(f"\n✗ Test error: {test_func.__name__}")
            print(f"  Error: {str(error)}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

