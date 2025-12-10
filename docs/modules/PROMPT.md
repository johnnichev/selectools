# Prompt Module

**File:** `src/selectools/prompt.py`
**Classes:** `PromptBuilder`
**Constants:** `DEFAULT_SYSTEM_INSTRUCTIONS`

## Table of Contents

1. [Overview](#overview)
2. [System Prompt Structure](#system-prompt-structure)
3. [Tool Schema Formatting](#tool-schema-formatting)
4. [Customization](#customization)
5. [Implementation](#implementation)

---

## Overview

The **PromptBuilder** generates system prompts that:

1. Explain the tool-calling contract to the LLM
2. List available tools with their JSON schemas
3. Provide guidelines for proper tool usage

This system prompt is **critical** - it's how the LLM learns what tools exist and how to invoke them.

---

## System Prompt Structure

### Components

```
┌─────────────────────────────────────────────┐
│  BASE INSTRUCTIONS                          │
│  • Tool calling contract                    │
│  • JSON format specification                │
│  • Usage guidelines                         │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  TOOL SCHEMAS                               │
│  • Tool 1: name, description, parameters    │
│  • Tool 2: name, description, parameters    │
│  • ...                                      │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  USAGE INSTRUCTIONS                         │
│  • When to use tools                        │
│  • When to answer directly                  │
└─────────────────────────────────────────────┘
```

### Full Example

```
You are an assistant that can call tools when helpful.

Tool call contract:
- Emit TOOL_CALL with JSON: {"tool_name": "<name>", "parameters": {...}}
- Include every required parameter. Ask for missing details instead of guessing.
- Wait for tool results before giving a final answer.
- Do not invent tool outputs; only report what was returned.
- Keep tool payloads compact (<=8k chars) and emit one tool call at a time.

Available tools (JSON schema):

{
  "name": "search",
  "description": "Search the web for information",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query"
      }
    },
    "required": ["query"]
  }
}

{
  "name": "calculator",
  "description": "Perform mathematical calculations",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "Mathematical expression to evaluate"
      }
    },
    "required": ["expression"]
  }
}

If a relevant tool exists, respond with a TOOL_CALL first. When no tool is useful, answer directly.
```

---

## Tool Schema Formatting

### Schema Generation

```python
def build(self, tools: List[Tool]) -> str:
    # 1. Start with base instructions
    prompt = f"{self.base_instructions.strip()}\n\n"

    # 2. Add tool schemas
    tool_blocks = []
    for tool in tools:
        # Get JSON schema from tool
        schema = tool.schema()

        # Format as pretty JSON
        tool_json = json.dumps(schema, indent=2)
        tool_blocks.append(tool_json)

    tools_text = "\n\n".join(tool_blocks)
    prompt += f"Available tools (JSON schema):\n\n{tools_text}\n\n"

    # 3. Add usage instructions
    prompt += (
        "If a relevant tool exists, respond with a TOOL_CALL first. "
        "When no tool is useful, answer directly."
    )

    return prompt
```

### Tool Schema Format

Each tool is represented as:

```json
{
  "name": "tool_name",
  "description": "What this tool does",
  "parameters": {
    "type": "object",
    "properties": {
      "param1": {
        "type": "string",
        "description": "Description of param1"
      },
      "param2": {
        "type": "integer",
        "description": "Description of param2"
      }
    },
    "required": ["param1"]
  }
}
```

This follows the [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) schema format, which is widely understood by LLMs.

---

## Customization

### Custom Base Instructions

```python
custom_instructions = """
You are a helpful assistant with access to tools.

When using tools:
1. Always verify parameters before calling
2. Handle errors gracefully
3. Provide clear explanations

Format: TOOL_CALL followed by JSON with tool_name and parameters.
"""

prompt_builder = PromptBuilder(base_instructions=custom_instructions)
agent = Agent(tools=[...], provider=provider, prompt_builder=prompt_builder)
```

### Domain-Specific Instructions

```python
# For a code assistant
code_instructions = """
You are an expert code assistant with access to tools.

Use tools when you need to:
- Execute code
- Search documentation
- Analyze files

Always explain your reasoning before using a tool.

Tool format: TOOL_CALL {"tool_name": "<name>", "parameters": {...}}
"""

# For a customer support bot
support_instructions = """
You are a customer support assistant with access to:
- Knowledge base search
- Order lookup
- Ticket creation

Be empathetic and professional. Use tools to provide accurate information.

Tool format: TOOL_CALL {"tool_name": "<name>", "parameters": {...}}
"""
```

---

## Implementation

### Default Instructions

```python
DEFAULT_SYSTEM_INSTRUCTIONS = """You are an assistant that can call tools when helpful.

Tool call contract:
- Emit TOOL_CALL with JSON: {"tool_name": "<name>", "parameters": {...}}
- Include every required parameter. Ask for missing details instead of guessing.
- Wait for tool results before giving a final answer.
- Do not invent tool outputs; only report what was returned.
- Keep tool payloads compact (<=8k chars) and emit one tool call at a time.
"""
```

### PromptBuilder Class

```python
class PromptBuilder:
    """Render a system prompt that includes tool schemas."""

    def __init__(self, base_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS):
        self.base_instructions = base_instructions

    def build(self, tools: List[Tool]) -> str:
        tool_blocks = []
        for tool in tools:
            tool_blocks.append(json.dumps(tool.schema(), indent=2))

        tools_text = "\n\n".join(tool_blocks)

        return (
            f"{self.base_instructions.strip()}\n\n"
            f"Available tools (JSON schema):\n\n{tools_text}\n\n"
            "If a relevant tool exists, respond with a TOOL_CALL first. "
            "When no tool is useful, answer directly."
        )
```

---

## Best Practices

### 1. Be Explicit About Contract

```python
# ✅ Good - Clear contract
"""
Emit TOOL_CALL with JSON: {"tool_name": "<name>", "parameters": {...}}
"""

# ❌ Bad - Vague
"""
Call tools when needed.
"""
```

### 2. Include Usage Guidelines

```python
# ✅ Good
"""
- Include every required parameter
- Ask for missing details instead of guessing
- Wait for tool results before answering
"""

# ❌ Bad - No guidelines
"""
Use tools.
"""
```

### 3. Specify JSON Format

```python
# ✅ Good
"""
{"tool_name": "<name>", "parameters": {...}}
"""

# ❌ Bad - Ambiguous
"""
Call: name(param1, param2)
"""
```

### 4. Set Expectations

```python
# ✅ Good
"""
- Do not invent tool outputs
- Keep payloads compact (<=8k chars)
- Emit one tool call at a time
"""
```

---

## Testing

### Verify Prompt Generation

```python
def test_prompt_builder():
    from selectools import Tool, ToolParameter, PromptBuilder

    tool = Tool(
        name="test",
        description="Test tool",
        parameters=[
            ToolParameter(name="arg", param_type=str, description="Test arg", required=True)
        ],
        function=lambda arg: arg
    )

    builder = PromptBuilder()
    prompt = builder.build([tool])

    # Check components
    assert "TOOL_CALL" in prompt
    assert "test" in prompt
    assert "Test tool" in prompt
    assert "arg" in prompt
    assert '"type": "string"' in prompt
```

### Custom Instructions

```python
def test_custom_instructions():
    custom = "Custom instructions here."
    builder = PromptBuilder(base_instructions=custom)

    prompt = builder.build([])
    assert "Custom instructions here." in prompt
```

---

## Provider-Specific Variations

### OpenAI

Receives the system prompt in the `system` role:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "User message"}
]
```

### Anthropic

Claude receives the system prompt via `system` parameter:

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system=system_prompt,
    messages=[...]
)
```

### Gemini

System instructions via `system_instruction`:

```python
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=system_prompt
)
```

The `PromptBuilder` generates provider-agnostic prompts. Providers handle the specific API formatting.

---

## Limitations

### 1. No Per-Tool Customization

All tools share the same instruction format. You can't have different rules for different tools.

**Workaround:** Include tool-specific notes in the tool description.

### 2. No Dynamic Instructions

The system prompt is static for the agent's lifetime.

**Workaround:** Create a new agent with updated prompt builder for different contexts.

### 3. No Conditional Tools

Can't show/hide tools based on conversation state.

**Workaround:** Create specialized agents for different contexts.

---

## Further Reading

- [Agent Module](AGENT.md) - How agents use system prompts
- [Tools Module](TOOLS.md) - Tool schema generation
- [Parser Module](PARSER.md) - How responses are parsed

---

**Next Steps:** Understand provider implementations in the [Providers Module](PROVIDERS.md).
