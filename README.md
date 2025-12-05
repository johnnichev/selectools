# AI Tool Calling from Scratch

## Overview

Imagine you just built the world's first multi-modal chat bot, and now you want to add tool calling to it. The goal of this coding challenge is to create a simple library for defining and running AI agents that can use tool calling. This should take around 2-3 hours to complete.
Your implementation can roughly follow the interface defined in `agent.py`.

## Main Task: Bounding Box Detection

Create an agent that can detect and return bounding boxes around / return full new versions of specific items in images using API endpoints from common providers (Gemini, OpenAI, Claude). The agent should:

- Accept a photo and text as input
- Use an API library (Gemini, OpenAI, or Claude) to recognize items in the image.
- Return bounding box coordinates for the detected item
- Be general enough to handle complex images: multiple objects, cluttered scenes, partial occlusions, varying lighting conditions, and diverse backgrounds
  Example use case: Provide a photo of a scene containing a dog, cat, and tree, and get a bounding box around the dog. (This is a very simple use case...)

```python
agent = Agent(
    tools=[Tool(name="detect_bounding_box", ...)]
)
agent.run(messages=[Message(role=Role.USER, content="Make a bounding box around the dog in this image", image="path/to/dog.jpg")])
```

## Rules

- The only non-standard libraries you can use are strictly for LLM/Vision API calls and Pillow (for image drawing/editing). You can use any LLM or Vision API you want (e.g., Gemini, OpenAI, Claude), but you cannot use any libraries' prebuilt tool calling capabilities (because that is exactly what you're making from scratch).
- Everything else must be implemented through your own code. This includes any prompt engineering, agentic logic, state management, response parsing, etc.
- The `Agent` skeleton in `agent.py` is just a starting point. Feel free to change/add any methods/classes, create helper classes, etc. The implementation of `Tool`, `Message`, and `Role` is up to you.

## Design Considerations

- What should the agent do if it doesn't have all the information it needs to answer the query?
- What should the agent do if it doesn't have the necessary parameters it needs to call a tool?
- How should the agent decide which tool to call?
- How should the agent decide when to stop calling tools?
- How should the agent handle errors from tool calls?
- How should the agent handle the response from tool calls?

## Requirements

- Implement the `Agent` class in `agent.py`.
- Implement the `Tool` class in `agent.py`.
- Implement the bounding box detection use case in `chat.py` file to demonstrate how to use your `Agent` and `Tool` classes with a photo of a dog.
- Output a photo of the original photo that has the bounding box overlayed.
- Anything beyond these requirements is totally up to you.

## Submission

- Submit a zip file containing the `agent.py` and `chat.py` files.
- Show an example of an agent in action.
- Give a brief explanation of your design decisions and how you implemented the agent.

---

## Implementation

### Quick Start

1. **Create a virtual environment (recommended):**

```bash
python3 -m venv .venv
```

2. **Activate the virtual environment:**

```bash
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set your OpenAI API key:**

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Alternatively, create a local `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

5. **Run the bounding box detection demo:**

```bash
python scripts/chat.py
```

This will detect the dog in `dog.png` and output `dog_with_bbox.png` with the bounding box drawn.

### Usage Examples

**Basic Tool Usage:**

```python
from agent import Agent, Tool, Message, Role, ToolParameter

# Define a tool
def search(query: str) -> str:
    return f"Results for: {query}"

search_tool = Tool(
    name="search",
    description="Search the web",
    parameters=[
        ToolParameter(name="query", param_type=str, description="Search query", required=True)
    ],
    function=search
)

# Create agent and run
agent = Agent(tools=[search_tool])
response = agent.run(messages=[
    Message(role=Role.USER, content="Search for Python tutorials")
])
print(response.content)
```

**Interactive Chat Mode (default):**

```bash
python scripts/chat.py --interactive
```

### Files

- **`agent.py`**: Core framework implementation (Role, Message, Tool, Agent classes)
- **`chat.py`**: Bounding box detection demo with image processing
- **`IMPLEMENTATION.md`**: Detailed documentation of design decisions and architecture
- **`requirements.txt`**: Python dependencies

### Key Features

- Extensible: add tools by defining parameters and a function
- Typed: full hints, no `any`, clean validation before execution
- Documented: docstrings throughout for functions and classes
- Resilient: clear error messages and guardrails in the agent loop
- Vision: OpenAI GPT-4 Vision support for bounding boxes
- Practical: logging of tool calls/results for transparency

### Design Highlights

- Custom tool calling (no prebuilt libraries)
- Structured TOOL_CALL JSON the agent emits and we parse
- Parameter validation before running any tool
- Agent loop with max-iteration stop to avoid hangs
- Full conversation history preserved for context
- Error handling and retries to keep responses stable

See `IMPLEMENTATION.md` for detailed design decisions and architecture documentation.
