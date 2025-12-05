# AI Tool Calling Framework - Implementation Documentation

## Overview

This project implements a complete AI tool calling framework from scratch, without using any prebuilt tool calling libraries. The implementation demonstrates how to build an agentic system that can intelligently use tools to accomplish tasks.

## Architecture

### Core Components

#### 1. Role (Enum)

Defines the four types of message roles in a conversation:

- **USER**: Messages from the end user
- **ASSISTANT**: Messages from the AI assistant
- **SYSTEM**: System-level instructions and context
- **TOOL**: Results from tool executions

#### 2. Message (Class)

A flexible message container that supports:

- Text content
- Image attachments (automatically encoded to base64)
- Tool execution results
- Conversion to OpenAI API format

**Key Design Decision**: Images are encoded to base64 at message creation time, making them ready for API transmission without additional processing.

#### 3. ToolParameter (Class)

Defines the schema for tool parameters:

- Parameter name, type, description
- Required/optional flag
- Automatic type mapping to JSON schema format

**Key Design Decision**: Using Python's native type system (str, int, float, bool) makes the API intuitive while still providing strong validation.

#### 4. Tool (Class)

Encapsulates a tool with:

- Metadata (name, description, parameters)
- Execution function
- Parameter validation logic
- Schema generation for LLM consumption

**Key Design Decision**: Tools are defined declaratively with metadata separate from implementation, making them easy to document and validate.

#### 5. Agent (Class)

The main orchestrator that implements the agentic loop:

1. Build system prompt with tool definitions
2. Call LLM with conversation history
3. Parse response for tool calls
4. Execute tools and integrate results
5. Repeat until final answer or max iterations

**Key Design Decision**: The agent maintains full conversation history, enabling multi-turn interactions and context-aware tool usage.

## Design Decisions

### 1. Tool Call Format

**Decision**: Use structured JSON within natural language responses

```
TOOL_CALL: {"tool_name": "...", "parameters": {...}}
```

**Rationale**:

- Easy to parse with regex and JSON.loads
- Works with any LLM that can follow instructions
- No dependency on provider-specific tool calling APIs
- Clear and debuggable format

### 2. Prompt Engineering Strategy

**Decision**: Inject tool schemas directly into system prompt with explicit instructions

**Rationale**:

- LLM sees all available tools and their parameters upfront
- Clear format reduces parsing errors
- System prompt is reusable across conversations
- Easy to extend with new tools

**Key Elements**:

- JSON schema format for tools
- Explicit TOOL_CALL format specification
- Guidelines for when to use tools vs. direct answers
- Instructions to ask for clarification when parameters are missing

### 3. Parameter Validation

**Decision**: Validate parameters before execution with detailed error messages

**Rationale**:

- Prevents runtime errors in tool functions
- Provides clear feedback to LLM for retry
- Type checking catches common mistakes
- Required/optional distinction enforced

**Implementation**:

- Check for missing required parameters
- Validate parameter types match schema
- Return descriptive error messages for LLM to understand

### 4. Error Handling Strategy

**Decision**: Graceful degradation with error feedback to LLM

**Approach**:

- Tool execution errors are caught and returned as tool messages
- LLM receives error context and can retry or ask for clarification
- Max iterations prevent infinite loops
- All errors are logged for debugging

**Rationale**:

- Errors become part of the conversation, not fatal failures
- LLM can adapt its approach based on error feedback
- Users see transparent error messages
- System remains stable even with bad inputs

### 5. Conversation State Management

**Decision**: Maintain full message history including tool calls and results

**Rationale**:

- LLM has complete context for decision making
- Multi-turn conversations work naturally
- Tool results inform subsequent tool calls
- Easy to debug by reviewing conversation history

**Trade-off**: Higher token usage vs. better context - we chose better context for more reliable tool usage

### 6. Vision Integration

**Decision**: Use base64 encoding for images with OpenAI's vision API

**Rationale**:

- Works with file paths without requiring URLs
- Automatic encoding in Message class
- Supports GPT-4o's vision capabilities
- No external image hosting required

## Bounding Box Detection Implementation

### Tool Design

The `detect_bounding_box` tool demonstrates a complex, real-world use case:

**Parameters**:

- `target_object`: String description of what to find
- `image_path`: Path to the image file

**Process**:

1. Validate image file exists
2. Encode image to base64
3. Call OpenAI Vision API with structured prompt
4. Parse normalized coordinates (0.0-1.0)
5. Validate coordinates are in valid range
6. Load image with Pillow
7. Convert normalized to pixel coordinates
8. Draw bounding box
9. Add label with background
10. Save output image
11. Return structured JSON result

### Prompt Engineering for Vision

**Key Technique**: Request structured JSON output from vision model

```
Return ONLY a JSON object with the bounding box coordinates in normalized format (0.0 to 1.0):
{
    "found": true/false,
    "x_min": 0.0-1.0,
    ...
}
```

**Rationale**:

- Normalized coordinates work for any image size
- JSON format is easy to parse and validate
- Explicit format reduces hallucination
- Confidence and description provide context

### Image Processing

**Decisions**:

- Box thickness scales with image size (0.5% of min dimension)
- Label positioned above box, or inside if too close to edge
- Red color for high visibility
- Font size scales with image height (3%)
- Fallback fonts for cross-platform compatibility

## Code Quality Standards

### Type Hints

- Every function has complete type hints
- No use of `Any` type
- Union types for optional parameters
- Return types explicitly declared

### Documentation

- Comprehensive docstrings on every class and function
- Docstrings include: purpose, parameters, returns, raises, examples
- Inline comments for complex logic
- Module-level documentation

### Error Messages

- Clear, actionable error messages
- Include context (what failed, why, how to fix)
- Errors distinguish between user mistakes and system failures

### Logging

- Strategic print statements show agent reasoning
- Tool execution is logged with inputs and outputs
- Iteration count prevents silent failures
- Results are clearly formatted

## Testing Approach

### Demo Modes

**1. Interactive Chat** (default):

```bash
python scripts/chat.py --interactive
```

- Multi-turn chat; ask for detections (e.g., `assets/dog.png`, `assets/environment.png`)
- Saves outputs beside source images (e.g., `dog_with_bbox.png`, `environment_with_bbox.png`)
- Shows full agent reasoning and tool calls

**2. Simple One-Shot Detection**:

```bash
python scripts/chat.py
```

- Single-turn detection of `assets/dog.png` -> `assets/dog_with_bbox.png`
- Minimal interaction; good for sanity check

### Validation

The implementation handles:

- Multiple objects in scene (specify which one)
- Cluttered scenes (vision model filters noise)
- Partial occlusions (detects visible portion)
- Varying lighting (vision model robust to lighting)
- Diverse backgrounds (object detection works in context)
- Missing files (clear error messages)
- Invalid coordinates (validation and retry)
- API errors (graceful degradation)

## Extensibility

### Adding New Tools

To add a new tool:

```python
def my_new_tool(param1: str, param2: int) -> str:
    """Implement tool logic here"""
    return f"Result: {param1} x {param2}"

my_tool = Tool(
    name="my_tool",
    description="What this tool does",
    parameters=[
        ToolParameter(name="param1", param_type=str, description="...", required=True),
        ToolParameter(name="param2", param_type=int, description="...", required=True),
    ],
    function=my_tool_impl
)

agent = Agent(tools=[my_tool, other_tool, ...])
```

### Supporting New LLM Providers

To add support for other providers (Claude, Gemini):

1. Add a provider parameter to Agent
2. Implement provider-specific message formatting
3. Create provider-specific API call method
4. Update `_call_openai` to dispatch to correct provider

The rest of the framework (tool calling, parsing, validation) remains unchanged.

## Performance Considerations

### Token Usage

- System prompt includes all tool schemas (~200-500 tokens)
- Full conversation history maintained (grows with turns)
- Images encoded as base64 (large token cost)

**Optimization**: For production, consider:

- Summarizing old conversation turns
- Removing tool results after they're used
- Compressing images before encoding

### API Calls

- One API call per agent iteration
- Vision API calls are expensive (bounding box detection)
- No caching implemented (every run is fresh)

**Optimization**: For production, consider:

- Caching vision results for same image+object
- Batch processing multiple detections
- Streaming responses for faster feedback

### Error Recovery

- Max iterations prevents infinite loops (default: 10)
- Tool errors don't crash the agent
- Malformed tool calls trigger retry with context

## Limitations and Future Improvements

### Current Limitations

1. Only supports OpenAI API (not Claude, Gemini)
2. No streaming responses (wait for complete response)
3. No tool call parallelization (sequential only)
4. No conversation summarization (token usage grows)
5. No persistent state (each run is independent)

### Potential Improvements

1. **Multi-provider support**: Abstract LLM interface
2. **Streaming**: Show agent thinking in real-time
3. **Parallel tools**: Execute independent tools simultaneously
4. **Memory**: Persist conversation state across runs
5. **Tool composition**: Tools that call other tools
6. **Confidence scoring**: Agent reports certainty
7. **Explanation**: Agent explains why it chose a tool
8. **Fallback strategies**: Multiple approaches if first fails
