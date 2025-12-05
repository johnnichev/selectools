"""
AI Tool Calling Framework - Core Implementation

This module provides a complete framework for building AI agents with tool calling capabilities.

Key Components:
- Role: Enum defining message roles in conversation
- Message: Container for conversation messages with support for text, images, and tool results
- ToolParameter: Schema definition for tool parameters
- Tool: Encapsulates tool metadata and execution logic
- Agent: Main orchestrator that manages conversation flow and tool execution

Design Philosophy:
- Explicit over implicit: All types are clearly defined
- Extensible: Easy to add new tools and capabilities
- Robust: Comprehensive error handling and validation
- Transparent: Clear logging of agent reasoning process
"""

import os
from typing import List, Callable, Dict, Optional, Union
from enum import Enum
import json
import re
import base64
from pathlib import Path


class Role(Enum):
    """
    Defines the role of a message sender in the conversation.
    
    Attributes:
        USER: Message from the end user
        ASSISTANT: Message from the AI assistant
        SYSTEM: System-level instructions or context
        TOOL: Result from a tool execution
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message:
    """
    Represents a single message in the conversation.
    
    Supports text content, image attachments, and tool execution results.
    Images are automatically encoded to base64 for API compatibility.
    
    Attributes:
        role: The role of the message sender
        content: Text content of the message
        image_path: Optional path to an image file
        image_base64: Base64-encoded image data
        tool_name: Name of tool if this is a tool result
        tool_result: Result data from tool execution
    
    Example:
        >>> msg = Message(role=Role.USER, content="Hello", image_path="photo.jpg")
        >>> msg.role
        <Role.USER: 'user'>
    """
    
    def __init__(
        self,
        role: Role,
        content: str,
        image_path: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_result: Optional[str] = None
    ):
        """
        Initialize a message.
        
        Args:
            role: The role of the message sender
            content: Text content of the message
            image_path: Optional path to an image file to attach
            tool_name: Optional name of tool if this is a tool result
            tool_result: Optional result data from tool execution
        """
        self.role = role
        self.content = content
        self.image_path = image_path
        self.image_base64: Optional[str] = None
        self.tool_name = tool_name
        self.tool_result = tool_result
        
        if image_path:
            self._encode_image(image_path)
    
    def _encode_image(self, image_path: str) -> None:
        """
        Encode an image file to base64 format for API transmission.
        
        Args:
            image_path: Path to the image file
        
        Raises:
            FileNotFoundError: If the image file doesn't exist
            IOError: If the image file cannot be read
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(path, "rb") as image_file:
            self.image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    def to_openai_format(self) -> Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]:
        """
        Convert message to OpenAI API format.
        
        Returns:
            Dictionary formatted for OpenAI Chat Completions API
        
        Example:
            >>> msg = Message(role=Role.USER, content="Hello")
            >>> msg.to_openai_format()
            {'role': 'user', 'content': 'Hello'}
        """
        role_value = self.role.value if self.role != Role.TOOL else Role.ASSISTANT.value

        if self.image_base64:
            return {
                "role": role_value,
                "content": [
                    {"type": "text", "text": self.content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.image_base64}"
                        }
                    }
                ]
            }
        else:
            return {
                "role": role_value,
                "content": self.content
            }


def _load_env_file() -> None:
    """
    Load environment variables from a local .env file if present.

    This enables offline configuration of secrets (e.g., OPENAI_API_KEY) without
    adding external dependencies. Values already present in os.environ are not
    overridden.
    """
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]

    for env_path in candidate_paths:
        if not env_path.exists() or not env_path.is_file():
            continue

        try:
            for line in env_path.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception:
            # If loading fails, continue without overriding environment.
            continue


class ToolParameter:
    """
    Defines a parameter schema for a tool.
    
    Used for validation and documentation of tool parameters.
    
    Attributes:
        name: Parameter name
        param_type: Python type of the parameter (e.g., str, int, float)
        description: Human-readable description of the parameter
        required: Whether this parameter is required
    
    Example:
        >>> param = ToolParameter(
        ...     name="query",
        ...     param_type=str,
        ...     description="Search query",
        ...     required=True
        ... )
    """
    
    def __init__(
        self,
        name: str,
        param_type: type,
        description: str,
        required: bool = True
    ):
        """
        Initialize a tool parameter definition.
        
        Args:
            name: Parameter name
            param_type: Python type of the parameter
            description: Human-readable description
            required: Whether this parameter is required
        """
        self.name = name
        self.param_type = param_type
        self.description = description
        self.required = required
    
    def to_schema(self) -> Dict[str, Union[str, bool]]:
        """
        Convert parameter to JSON schema format.
        
        Returns:
            Dictionary representing the parameter schema
        """
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        
        return {
            "type": type_map.get(self.param_type, "string"),
            "description": self.description,
            "required": self.required
        }


class Tool:
    """
    Encapsulates a tool that the agent can use.
    
    A tool consists of metadata (name, description, parameters) and an execution function.
    The agent uses the metadata to decide when to call the tool, and the function to execute it.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        parameters: List of parameter definitions
        function: Callable that executes the tool logic
    
    Example:
        >>> def search(query: str) -> str:
        ...     return f"Results for: {query}"
        >>> 
        >>> tool = Tool(
        ...     name="search",
        ...     description="Search the web",
        ...     parameters=[
        ...         ToolParameter(name="query", param_type=str, description="Search query")
        ...     ],
        ...     function=search
        ... )
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        function: Callable[..., str]
    ):
        """
        Initialize a tool.
        
        Args:
            name: Unique identifier for the tool
            description: Human-readable description of what the tool does
            parameters: List of parameter definitions
            function: Callable that executes the tool logic
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
    
    def to_schema(self) -> Dict[str, Union[str, Dict[str, Union[str, bool]]]]:
        """
        Convert tool to JSON schema format for LLM consumption.
        
        Returns:
            Dictionary representing the tool schema
        """
        params_schema = {}
        for param in self.parameters:
            params_schema[param.name] = param.to_schema()
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params_schema
        }
    
    def validate_parameters(self, params: Dict[str, Union[str, int, float, bool]]) -> tuple[bool, Optional[str]]:
        """
        Validate that provided parameters match the tool's schema.
        
        Args:
            params: Dictionary of parameter names to values
        
        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is None
            If invalid, error_message describes the validation failure
        
        Example:
            >>> tool = Tool(name="test", description="", parameters=[
            ...     ToolParameter(name="query", param_type=str, required=True)
            ... ], function=lambda x: x)
            >>> tool.validate_parameters({"query": "hello"})
            (True, None)
            >>> tool.validate_parameters({})
            (False, 'Missing required parameter: query')
        """
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"
            
            if param.name in params:
                value = params[param.name]
                if not isinstance(value, param.param_type):
                    return False, f"Parameter '{param.name}' must be of type {param.param_type.__name__}, got {type(value).__name__}"
        
        return True, None
    
    def execute(self, params: Dict[str, Union[str, int, float, bool]]) -> str:
        """
        Execute the tool with the provided parameters.
        
        Args:
            params: Dictionary of parameter names to values
        
        Returns:
            String result from the tool execution
        
        Raises:
            ValueError: If parameters are invalid
            Exception: Any exception raised by the tool function
        """
        is_valid, error_message = self.validate_parameters(params)
        if not is_valid:
            raise ValueError(f"Invalid parameters for tool '{self.name}': {error_message}")
        
        return self.function(**params)


class Agent:
    """
    AI Agent that can use tools to accomplish tasks.
    
    The agent maintains conversation history and orchestrates an agentic loop:
    1. Analyze user request and conversation context
    2. Determine if tools are needed
    3. Execute tools with proper parameters
    4. Integrate results and continue or provide final answer
    
    The agent uses prompt engineering to guide the LLM to output structured tool calls
    that can be parsed and executed.
    
    Attributes:
        tools: List of available tools
        messages: Conversation history
        max_iterations: Maximum number of tool calls to prevent infinite loops
        model: OpenAI model to use
        api_key: OpenAI API key
    
    Example:
        >>> agent = Agent(tools=[search_tool], api_key="sk-...")
        >>> result = agent.run(messages=[
        ...     Message(role=Role.USER, content="Search for Python tutorials")
        ... ])
    """
    
    def __init__(
        self,
        tools: List[Tool],
        max_iterations: int = 10,
        model: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        """
        Initialize an agent.
        
        Args:
            tools: List of tools the agent can use
            max_iterations: Maximum number of tool calls to prevent infinite loops
            model: OpenAI model to use (default: gpt-4o for vision support)
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
        """
        self.tools = tools
        self.messages: List[Message] = []
        self.max_iterations = max_iterations
        self.model = model
        self.api_key = api_key
        
        self._tools_by_name: Dict[str, Tool] = {tool.name: tool for tool in tools}

        if not api_key:
            _load_env_file()
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt that instructs the LLM on how to use tools.
        
        Returns:
            System prompt string with tool schemas embedded
        """
        tools_description = []
        for tool in self.tools:
            schema = tool.to_schema()
            tools_description.append(json.dumps(schema, indent=2))
        
        tools_text = "\n\n".join(tools_description)
        
        return f"""You are an AI assistant with access to tools. Always prefer calling a tool when one is available and relevant.

Tool call format (MUST be used when a tool can answer the request):

TOOL_CALL: {{"tool_name": "name_of_tool", "parameters": {{"param1": "value1", "param2": "value2"}}}}

After you receive the tool result, use it to formulate your final answer to the user.

Available tools:

{tools_text}

Important guidelines:
1. If a relevant tool exists, you MUST respond with a TOOL_CALL first. Do not answer directly.
2. Provide all required parameters for the tool.
3. Wait for the tool result before providing your final answer.
4. If you don't have enough information to call a tool, ask the user for clarification.
5. Once you have the tool result, provide a clear, helpful answer to the user.
6. Do not make up or hallucinate tool results - always wait for the actual execution.
7. If any user message includes an image and a detection tool exists, you MUST call that detection tool with the provided image and the target object described by the user."""
    
    def _parse_tool_call(self, response: str) -> Optional[tuple[str, Dict[str, Union[str, int, float, bool]]]]:
        """
        Parse a tool call from the LLM response with robust fallbacks.
        
        Looks for the pattern: TOOL_CALL: {"tool_name": "...", "parameters": {...}}
        - Handles code fences and multiline responses
        - Falls back to lenient JSON parsing if needed
        """
        marker = "TOOL_CALL"
        if marker not in response:
            return None

        # If the response contains code fences, prefer the fenced block with TOOL_CALL
        if "```" in response:
            fenced_blocks = re.findall(r"```.*?```", response, re.DOTALL)
            for block in fenced_blocks:
                if marker in block:
                    # Strip backticks and whitespace
                    response = block.strip("` \n")
                    break

        try:
            subset = response[response.index(marker):]
        except ValueError:
            return None

        json_match = re.search(r"\{.*\}", subset, re.DOTALL)
        if not json_match:
            return None

        json_text = json_match.group()
        json_text_escaped = json_text.replace("\n", "\\n")

        def _attempt_parse(text: str) -> Optional[Dict[str, Union[str, int, float, bool]]]:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                try:
                    return json.loads(text.replace("'", '"'))
                except Exception:
                    return None

        tool_call_data = _attempt_parse(json_text) or _attempt_parse(json_text_escaped)
        if not tool_call_data:
            return None

        tool_name = tool_call_data.get("tool_name") or tool_call_data.get("tool")
        parameters = tool_call_data.get("parameters") or tool_call_data.get("params") or {}

        if not tool_name:
            return None

        return tool_name, parameters
    
    def _call_openai(self, messages: List[Message]) -> str:
        """
        Make a call to the OpenAI API.
        
        Args:
            messages: List of messages in the conversation
        
        Returns:
            The assistant's response text
        
        Raises:
            Exception: If the API call fails
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        
        client = OpenAI(api_key=self.api_key)
        
        formatted_messages = [msg.to_openai_format() for msg in messages]
        
        response = client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def run(self, messages: List[Message]) -> Message:
        """
        Run the agent with the given messages.
        
        This is the main entry point for the agent. It implements the agentic loop:
        1. Add system prompt with tool definitions
        2. Get LLM response
        3. Check if LLM wants to call a tool
        4. If yes, execute tool and loop back to step 2
        5. If no, return the final response
        
        Args:
            messages: List of messages representing the conversation
        
        Returns:
            The agent's final response as a Message
        
        Raises:
            ValueError: If max iterations is reached or tool execution fails
        
        Example:
            >>> agent = Agent(tools=[tool], api_key="sk-...")
            >>> response = agent.run([Message(role=Role.USER, content="Hello")])
            >>> print(response.content)
        """
        system_prompt = self._build_system_prompt()
        self.messages = [Message(role=Role.SYSTEM, content=system_prompt)] + messages
        
        iteration_count = 0
        
        while iteration_count < self.max_iterations:
            iteration_count += 1
            
            print(f"\n{'='*60}")
            print(f"Agent Iteration {iteration_count}/{self.max_iterations}")
            print(f"{'='*60}")
            
            try:
                response_text = self._call_openai(self.messages)
                print(f"\nLLM Response:\n{response_text[:200]}{'...' if len(response_text) > 200 else ''}")
                
                tool_call = self._parse_tool_call(response_text)
                
                if tool_call is None:
                    print("\nNo tool call detected. Returning final response.")
                    return Message(role=Role.ASSISTANT, content=response_text)
                
                tool_name, parameters = tool_call
                print(f"\nTool Call Detected:")
                print(f"  Tool: {tool_name}")
                print(f"  Parameters: {json.dumps(parameters, indent=2)}")
                
                if tool_name not in self._tools_by_name:
                    error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self._tools_by_name.keys())}"
                    print(f"\n{error_msg}")
                    self.messages.append(Message(role=Role.ASSISTANT, content=response_text))
                    self.messages.append(Message(role=Role.TOOL, content=error_msg, tool_name=tool_name))
                    continue
                
                tool = self._tools_by_name[tool_name]
                
                try:
                    result = tool.execute(parameters)
                    print(f"\nTool Result:\n{result[:200]}{'...' if len(result) > 200 else ''}")
                    
                    self.messages.append(Message(role=Role.ASSISTANT, content=response_text))
                    self.messages.append(
                        Message(
                            role=Role.TOOL,
                            content=f"Tool '{tool_name}' returned: {result}",
                            tool_name=tool_name,
                            tool_result=result
                        )
                    )
                    
                except Exception as error:
                    error_msg = f"Error executing tool '{tool_name}': {str(error)}"
                    print(f"\n{error_msg}")
                    self.messages.append(Message(role=Role.ASSISTANT, content=response_text))
                    self.messages.append(Message(role=Role.TOOL, content=error_msg, tool_name=tool_name))
                
            except Exception as error:
                error_msg = f"Error in agent loop: {str(error)}"
                print(f"\n{error_msg}")
                return Message(role=Role.ASSISTANT, content=f"I encountered an error: {error_msg}")
        
        return Message(
            role=Role.ASSISTANT,
            content=f"Maximum iterations ({self.max_iterations}) reached. Please try rephrasing your request or breaking it into smaller steps."
        )
