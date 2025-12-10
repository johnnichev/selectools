# Parser Module

**File:** `src/selectools/parser.py`
**Classes:** `ToolCallParser`, `ParseResult`

## Table of Contents

1. [Overview](#overview)
2. [TOOL_CALL Contract](#tool_call-contract)
3. [Parsing Strategy](#parsing-strategy)
4. [JSON Extraction](#json-extraction)
5. [Error Recovery](#error-recovery)
6. [Implementation Details](#implementation-details)

---

## Overview

The **ToolCallParser** robustly extracts `TOOL_CALL` directives from LLM responses. Unlike strict JSON parsers, it's designed to handle the messy reality of LLM outputs:

- Fenced code blocks (`json ... `)
- Inline JSON mixed with explanatory text
- Malformed JSON with common errors
- Multiple variations of field names
- Newline-heavy formatting

### Design Philosophy

**Lenient by Design**: LLMs don't always produce perfect JSON. The parser uses multiple strategies to extract tool calls even from imperfect responses.

**No False Negatives**: It's better to parse a slightly malformed tool call than to reject a valid one.

---

## TOOL_CALL Contract

### Specification

The agent instructs the LLM to emit tool calls in this format:

```
TOOL_CALL
{
  "tool_name": "<name>",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

### Field Variations

The parser accepts multiple field names for flexibility:

**Tool Name:**

- `tool_name` (preferred)
- `tool`
- `name`

**Parameters:**

- `parameters` (preferred)
- `params`

### Examples of Valid Formats

#### Standard Format

```
TOOL_CALL
{
  "tool_name": "search",
  "parameters": {"query": "Python tutorials"}
}
```

#### With Code Fence

````markdown
TOOL_CALL

```json
{
  "tool_name": "search",
  "parameters": { "query": "Python tutorials" }
}
```
````

```

#### Mixed with Text

```

I'll search for that information.

TOOL_CALL
{"tool_name": "search", "parameters": {"query": "Python tutorials"}}

Let me find that for you.

```

#### Alternate Field Names

```

TOOL_CALL
{
"tool": "search",
"params": {"query": "Python tutorials"}
}

```

---

## Parsing Strategy

### Multi-Stage Extraction

```

LLM Response Text
│
▼
┌────────────────────────────────┐
│ Stage 1: Find Candidate Blocks│
├────────────────────────────────┤
│ • Search for TOOL_CALL marker │
│ • Extract fenced code blocks │
│ • Find balanced JSON objects │
└────────────┬───────────────────┘
│
▼
┌────────────────────────────────┐
│ Stage 2: Clean & Normalize │
├────────────────────────────────┤
│ • Strip markers and fences │
│ • Remove extra whitespace │
│ • Normalize quotes │
└────────────┬───────────────────┘
│
▼
┌────────────────────────────────┐
│ Stage 3: JSON Parsing │
├────────────────────────────────┤
│ • Try direct parsing │
│ • Try with quote normalization │
│ • Try with newline handling │
└────────────┬───────────────────┘
│
▼
┌────────────────────────────────┐
│ Stage 4: Field Extraction │
├────────────────────────────────┤
│ • Extract tool_name/tool/name │
│ • Extract parameters/params │
│ • Validate presence │
└────────────┬───────────────────┘
│
▼
ParseResult

````

### Implementation Flow

```python
def parse(self, text: str) -> ParseResult:
    # 1. Extract candidate blocks
    candidates = self._extract_candidate_blocks(text)

    # 2. Try to parse each candidate
    for candidate in candidates:
        # Size limit check
        if self.max_payload_chars and len(candidate) > self.max_payload_chars:
            continue

        # 3. Try JSON parsing
        tool_data = self._load_json(candidate)
        if not tool_data:
            continue

        # 4. Extract fields (flexible field names)
        tool_name = (
            tool_data.get("tool_name")
            or tool_data.get("tool")
            or tool_data.get("name")
        )
        parameters = (
            tool_data.get("parameters")
            or tool_data.get("params")
            or {}
        )

        # 5. Validate and return
        if tool_name:
            return ParseResult(
                tool_call=ToolCall(tool_name=tool_name, parameters=parameters),
                raw_text=text
            )

    # No tool call found
    return ParseResult(tool_call=None, raw_text=text)
````

---

## JSON Extraction

### Candidate Block Extraction

````python
def _extract_candidate_blocks(self, text: str) -> List[str]:
    blocks = []

    # 1. Extract blocks near TOOL_CALL marker
    marker_positions = [m.start() for m in re.finditer(self.marker, text)]
    for pos in marker_positions:
        subset = text[pos:]
        blocks.extend(self._find_balanced_json(subset))

    # 2. Extract from fenced code blocks
    fenced_blocks = re.findall(r"```.*?```", text, re.DOTALL)
    for block in fenced_blocks:
        if self.marker in block or "tool_name" in block or "parameters" in block:
            cleaned = block.strip("` \n")
            blocks.extend(self._find_balanced_json(cleaned))

    # 3. Fallback: search entire text
    if not blocks:
        blocks.extend(self._find_balanced_json(text))

    # 4. Deduplicate
    return self._deduplicate(blocks)
````

### Balanced Bracket Parsing

```python
def _find_balanced_json(self, text: str) -> List[str]:
    """Find all balanced JSON objects using bracket matching."""
    candidates = []

    # Find all '{' positions
    starts = [m.start() for m in re.finditer(r"\{", text)]

    for start in starts:
        depth = 0

        # Track bracket depth
        for idx in range(start, len(text)):
            char = text[idx]

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                # Found matching closing bracket
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break

    return candidates
```

This handles nested objects correctly:

```json
{
  "tool_name": "process",
  "parameters": {
    "config": {
      "nested": "value"
    }
  }
}
```

### Lenient JSON Parsing

```python
def _load_json(self, candidate: str) -> Optional[Dict[str, Any]]:
    # 1. Remove TOOL_CALL marker
    normalized = candidate
    if self.marker in normalized:
        normalized = normalized.split(self.marker, maxsplit=1)[-1]

    # 2. Strip markers
    normalized = normalized.strip("` \n:")

    # 3. Try multiple strategies
    attempts = [
        normalized,                          # Direct
        normalized.replace("'", '"'),        # Single quotes to double
        normalized.replace("\n", "\\n"),     # Escape newlines
    ]

    for attempt in attempts:
        try:
            result = json.loads(attempt)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    return None
```

---

## Error Recovery

### Common LLM Errors

#### 1. Single Quotes Instead of Double

```json
{
  "tool_name": "search",
  "parameters": { "query": "test" }
}
```

**Recovery:** `normalized.replace("'", '"')`

#### 2. Unescaped Newlines in Strings

```json
{
  "tool_name": "write_file",
  "parameters": {
    "content": "Line 1
Line 2"
  }
}
```

**Recovery:** `normalized.replace("\n", "\\n")`

#### 3. Mixed with Explanatory Text

```
I'll search for that information using the search tool.

TOOL_CALL
{"tool_name": "search", "parameters": {"query": "Python"}}

This will help me find relevant results.
```

**Recovery:** Marker-based extraction + balanced bracket parsing

#### 4. Multiple JSON Objects

```
{"tool_name": "first"}
{"tool_name": "second"}
```

**Recovery:** Parse first valid object only (agents execute one tool at a time)

### Max Payload Size

Large responses are rejected to prevent processing issues:

```python
if self.max_payload_chars and len(candidate) > self.max_payload_chars:
    continue  # Skip this candidate
```

Default: 8000 characters

---

## Implementation Details

### ParseResult Dataclass

```python
@dataclass
class ParseResult:
    tool_call: Optional[ToolCall]  # None if no tool call found
    raw_text: str                  # Original LLM response
```

### ToolCall Type

```python
@dataclass
class ToolCall:
    tool_name: str
    parameters: Dict[str, Any]
```

### Parser Configuration

```python
parser = ToolCallParser(
    marker="TOOL_CALL",           # Keyword to search for
    max_payload_chars=8000        # Max JSON size
)
```

### Example Usage

```python
parser = ToolCallParser()

# Valid tool call
result = parser.parse("""
TOOL_CALL
{"tool_name": "search", "parameters": {"query": "Python"}}
""")

assert result.tool_call is not None
assert result.tool_call.tool_name == "search"
assert result.tool_call.parameters == {"query": "Python"}

# No tool call
result = parser.parse("Just a regular response with no tool call.")
assert result.tool_call is None
```

---

## Edge Cases

### Empty Parameters

```json
{
  "tool_name": "get_time",
  "parameters": {}
}
```

**Handling:** Valid. Tool may have no parameters or all optional.

### Null Parameters

```json
{
  "tool_name": "get_time",
  "parameters": null
}
```

**Handling:** Treated as empty dict `{}`.

### Missing Parameters Field

```json
{
  "tool_name": "get_time"
}
```

**Handling:** Parameters default to `{}`.

### Invalid Tool Name

```json
{
  "tool_name": "",
  "parameters": {}
}
```

**Handling:** `tool_name` is falsy, so `parse_result.tool_call` will be `None`.

### Deeply Nested JSON

```json
{
  "tool_name": "complex",
  "parameters": {
    "level1": {
      "level2": {
        "level3": {
          "value": "deep"
        }
      }
    }
  }
}
```

**Handling:** Balanced bracket parser handles any depth. Tool validation happens later.

---

## Performance Considerations

### Regex vs Manual Parsing

The parser uses regex for:

- Finding markers (`re.finditer`)
- Extracting code blocks (`re.findall`)
- Finding bracket positions (`re.finditer`)

But uses manual iteration for:

- Balanced bracket matching (depth tracking)
- JSON parsing (stdlib `json.loads`)

### Deduplication

```python
def _deduplicate(self, blocks: List[str]) -> List[str]:
    deduped = []
    seen = set()

    for block in blocks:
        if block in seen:
            continue
        deduped.append(block)
        seen.add(block)

    return deduped
```

Prevents processing the same JSON multiple times.

### Early Exit

Parser returns immediately after finding first valid tool call:

```python
for candidate in candidates:
    tool_data = self._load_json(candidate)
    if tool_data and tool_data.get("tool_name"):
        return ParseResult(...)  # Found it!

return ParseResult(tool_call=None, ...)  # Not found
```

---

## Testing

### Unit Tests

````python
def test_parse_standard_format():
    parser = ToolCallParser()
    result = parser.parse('TOOL_CALL\n{"tool_name": "search", "parameters": {"q": "test"}}')

    assert result.tool_call is not None
    assert result.tool_call.tool_name == "search"
    assert result.tool_call.parameters == {"q": "test"}

def test_parse_with_code_fence():
    parser = ToolCallParser()
    text = '''
    ```json
    {"tool_name": "search", "parameters": {"q": "test"}}
    ```
    '''
    result = parser.parse(text)

    assert result.tool_call is not None

def test_no_tool_call():
    parser = ToolCallParser()
    result = parser.parse("Just a normal response.")

    assert result.tool_call is None

def test_malformed_json():
    parser = ToolCallParser()
    # Single quotes instead of double
    result = parser.parse("TOOL_CALL\n{'tool_name': 'search', 'parameters': {}}")

    assert result.tool_call is not None  # Should still parse

def test_alternate_field_names():
    parser = ToolCallParser()
    result = parser.parse('{"tool": "search", "params": {}}')

    assert result.tool_call is not None
    assert result.tool_call.tool_name == "search"
````

### Integration with Agent

```python
def test_parser_with_agent():
    from selectools import Agent, Tool, ToolParameter
    from selectools.providers.stubs import LocalProvider

    tool = Tool(
        name="test",
        description="Test tool",
        parameters=[],
        function=lambda: "result"
    )

    agent = Agent(
        tools=[tool],
        provider=LocalProvider(),
        parser=ToolCallParser()  # Custom parser
    )
```

---

## Best Practices

### 1. Use Standard Field Names in Prompts

In your system prompt, prefer:

- `tool_name` over `tool` or `name`
- `parameters` over `params`

This reduces parsing ambiguity.

### 2. Keep Marker Simple

`TOOL_CALL` is clear and unlikely to appear in regular text.

Avoid:

- Very common words (`CALL`, `TOOL`)
- Special characters that need escaping
- Very long markers

### 3. Set Reasonable Size Limits

```python
parser = ToolCallParser(max_payload_chars=8000)
```

Prevents processing enormous invalid JSONs.

### 4. Monitor Parse Failures

```python
result = parser.parse(response)
if result.tool_call is None:
    logger.warning(f"Failed to parse tool call from: {response[:100]}")
```

### 5. Test with Real LLM Outputs

Collect examples of actual LLM responses (both valid and invalid) and add them to your test suite.

---

## Debugging

### Verbose Parsing

```python
def parse_verbose(text: str):
    parser = ToolCallParser()

    print(f"Input text length: {len(text)}")
    print(f"Contains marker: {parser.marker in text}")

    candidates = parser._extract_candidate_blocks(text)
    print(f"Found {len(candidates)} candidate blocks")

    for i, candidate in enumerate(candidates):
        print(f"\nCandidate {i}:")
        print(candidate[:200])

        tool_data = parser._load_json(candidate)
        print(f"Parsed: {tool_data}")

    result = parser.parse(text)
    print(f"\nFinal result: {result}")

    return result
```

---

## Limitations

### 1. One Tool Call Per Response

The parser returns the first valid tool call found. If the LLM outputs multiple tool calls, only the first is used.

**Workaround:** Agents execute one tool at a time by design.

### 2. No Syntax Error Messages

If parsing fails, the parser silently returns `None`. It doesn't report why.

**Rationale:** The agent continues without tool execution. The LLM response is returned as-is.

### 3. Language-Specific JSON

Only handles JSON, not other data formats (YAML, TOML, etc.).

**Rationale:** JSON is universal and well-supported by all LLMs.

---

## Future Enhancements

Potential improvements (not currently implemented):

1. **Streaming Parsing**: Parse tool calls from streaming responses
2. **Multiple Tool Calls**: Support parallel tool execution
3. **Structured Output Mode**: Use provider-specific structured output APIs
4. **Error Reporting**: Detailed parsing failure reasons
5. **Alternative Formats**: XML, YAML support

---

## Further Reading

- [Agent Module](AGENT.md) - How the agent uses parsed tool calls
- [Tools Module](TOOLS.md) - Tool definition and validation
- [Prompt Module](PROMPT.md) - How the TOOL_CALL contract is specified

---

**Next Steps:** Learn how prompts are built in the [Prompt Module](PROMPT.md).
