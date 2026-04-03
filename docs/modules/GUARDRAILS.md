---
description: "Input/output validation: PII redaction, toxicity, topic blocking, format checks"
tags:
  - security
  - guardrails
---

# Guardrails Engine

**Import:** `from selectools.guardrails import GuardrailsPipeline`

**Stability:** stable

```python title="guardrails_quickstart.py"
from selectools import Agent, AgentConfig, Message, Role, tool
from selectools.providers.stubs import LocalProvider
from selectools.guardrails import GuardrailsPipeline, PIIGuardrail, TopicGuardrail

@tool(description="Look up a customer by email")
def lookup_customer(email: str) -> str:
    return f"Customer found: Jane Doe ({email})"

# PII redaction + topic blocking
guardrails = GuardrailsPipeline(
    input=[
        PIIGuardrail(action="rewrite"),
        TopicGuardrail(deny=["politics", "religion"]),
    ],
    output=[],
)

provider = LocalProvider()
agent = Agent(
    tools=[lookup_customer],
    provider=provider,
    config=AgentConfig(guardrails=guardrails, max_iterations=1),
)

# PII is automatically redacted before reaching the LLM
result = agent.run([Message(role=Role.USER, content="Look up user@example.com")])
print(result.content)
```

!!! tip "See Also"
    - [Security](SECURITY.md) - Tool output screening and prompt injection detection
    - [Audit](AUDIT.md) - JSONL audit logging with privacy levels

---

**Added in:** v0.15.0

Guardrails validate content **before** (input) and **after** (output) every LLM call. They catch unsafe inputs, redact PII, enforce output formats, and block toxic content — all without changing your application code.

---

## Quick Start

```python
from selectools import Agent, AgentConfig, OpenAIProvider, tool
from selectools.guardrails import GuardrailsPipeline, TopicGuardrail, PIIGuardrail

@tool(description="Look up a customer by email")
def lookup_customer(email: str) -> str:
    return f"Customer found: John Doe ({email})"

guardrails = GuardrailsPipeline(
    input=[
        TopicGuardrail(deny=["politics", "religion"]),
        PIIGuardrail(action="rewrite"),   # redact PII in user messages
    ],
    output=[],  # no output guardrails for now
)

agent = Agent(
    tools=[lookup_customer],
    provider=OpenAIProvider(),
    config=AgentConfig(guardrails=guardrails),
)

# This works fine:
result = agent.ask("Look up customer john@example.com")
# Input is rewritten: "Look up customer [EMAIL:********]"

# This raises GuardrailError:
result = agent.ask("What do you think about politics?")
# GuardrailError: Guardrail 'topic' blocked: Denied topics detected: politics
```

---

## How It Works

```
User Message → Input Guardrails → LLM Call → Output Guardrails → Response
                    ↓                              ↓
              block / rewrite / warn          block / rewrite / warn
```

1. **Input guardrails** run on every user message before it reaches the LLM
2. **Output guardrails** run on the LLM response before it's returned to you
3. Guardrails execute **in order** — if one rewrites content, the next sees the rewritten version
4. If a guardrail **blocks**, processing stops immediately with a `GuardrailError`

---

## Failure Actions

Every guardrail has an `action` that controls what happens when content fails the check:

| Action | Behaviour | Use Case |
|---|---|---|
| `block` (default) | Raises `GuardrailError` | Hard safety boundaries |
| `rewrite` | Returns sanitised content | PII redaction, length truncation |
| `warn` | Logs a warning, continues | Monitoring without blocking |

```python
from selectools.guardrails import GuardrailAction, TopicGuardrail

# Block (default) — raises exception
TopicGuardrail(deny=["politics"], action=GuardrailAction.BLOCK)

# Warn — logs and continues
TopicGuardrail(deny=["politics"], action=GuardrailAction.WARN)
```

---

## Built-in Guardrails

### TopicGuardrail

Block content mentioning denied topics using keyword matching with word boundaries.

```python
from selectools.guardrails import TopicGuardrail

# Basic usage
g = TopicGuardrail(deny=["politics", "religion", "gambling"])

# Case-sensitive matching
g = TopicGuardrail(deny=["API_KEY"], case_sensitive=True)

# Warn instead of block
g = TopicGuardrail(deny=["competitors"], action="warn")
```

### PIIGuardrail

Detect and redact personally identifiable information using regex patterns.

**Built-in PII types:** `email`, `phone_us`, `ssn`, `credit_card`, `ipv4`

```python
from selectools.guardrails import PIIGuardrail, GuardrailAction

# Redact all PII (default action is rewrite)
g = PIIGuardrail()
result = g.check("Email me at user@example.com, SSN 123-45-6789")
# result.content = "Email me at [EMAIL:********], SSN [SSN:********]"

# Detect specific types only
g = PIIGuardrail(detect=["email", "credit_card"])

# Block instead of redact
g = PIIGuardrail(action=GuardrailAction.BLOCK)

# Add custom patterns
g = PIIGuardrail(custom_patterns={
    "employee_id": r"EMP-\d{6}",
    "internal_ip": r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}",
})

# Just detect without a guardrail pipeline
matches = g.detect("Contact user@example.com")
for m in matches:
    print(f"  {m.pii_type}: '{m.value}' at {m.start}-{m.end}")
```

### ToxicityGuardrail

Score content against a keyword blocklist. Configurable threshold controls sensitivity.

```python
from selectools.guardrails import ToxicityGuardrail

# Block on any toxic word (threshold=0.0)
g = ToxicityGuardrail(threshold=0.0)

# Only block when many toxic words appear
g = ToxicityGuardrail(threshold=0.3)

# Custom blocklist
g = ToxicityGuardrail(blocklist={"spam", "scam", "phishing"})

# Check score without blocking
score = g.score("Some text to check")
matched = g.matched_words("Some text to check")
```

### FormatGuardrail

Validate output format — JSON structure, required keys, length bounds.

```python
from selectools.guardrails import FormatGuardrail

# Require valid JSON
g = FormatGuardrail(require_json=True)

# Require specific keys in JSON
g = FormatGuardrail(require_json=True, required_keys=["intent", "confidence"])

# Length bounds (characters)
g = FormatGuardrail(min_length=10, max_length=5000)
```

### LengthGuardrail

Enforce content length in characters or words. Supports truncation on `rewrite`.

```python
from selectools.guardrails import LengthGuardrail, GuardrailAction

# Hard limit
g = LengthGuardrail(max_chars=10000)

# Truncate to fit (rewrite mode)
g = LengthGuardrail(max_words=500, action=GuardrailAction.REWRITE)

# Minimum length (useful for output guardrails)
g = LengthGuardrail(min_words=10)
```

---

## Pipeline Examples

### Input: PII Redaction + Topic Blocking

```python
pipeline = GuardrailsPipeline(
    input=[
        PIIGuardrail(action="rewrite"),          # Step 1: redact PII
        TopicGuardrail(deny=["internal_only"]),   # Step 2: block restricted topics
    ],
)
```

### Output: JSON Validation + Length Cap

```python
pipeline = GuardrailsPipeline(
    output=[
        FormatGuardrail(require_json=True, required_keys=["answer"]),
        LengthGuardrail(max_chars=2000, action="rewrite"),
    ],
)
```

### Both Input and Output

```python
pipeline = GuardrailsPipeline(
    input=[
        PIIGuardrail(action="rewrite"),
        TopicGuardrail(deny=["violence", "illegal"]),
    ],
    output=[
        ToxicityGuardrail(threshold=0.0),
        LengthGuardrail(max_chars=5000, action="rewrite"),
    ],
)

agent = Agent(
    tools=[...],
    provider=provider,
    config=AgentConfig(guardrails=pipeline),
)
```

---

## Custom Guardrails

Subclass `Guardrail` and override `check()`:

```python
from selectools.guardrails import Guardrail, GuardrailAction, GuardrailResult
import re

class NoProfanityGuardrail(Guardrail):
    name = "no_profanity"
    action = GuardrailAction.BLOCK

    def __init__(self, words: list[str]) -> None:
        self._patterns = [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]

    def check(self, content: str) -> GuardrailResult:
        for pattern in self._patterns:
            if pattern.search(content):
                return GuardrailResult(
                    passed=False,
                    content=content,
                    reason=f"Profanity detected: {pattern.pattern}",
                    guardrail_name=self.name,
                )
        return GuardrailResult(passed=True, content=content, guardrail_name=self.name)

# Use it
pipeline = GuardrailsPipeline(
    input=[NoProfanityGuardrail(words=["badword1", "badword2"])],
)
```

---

## Error Handling

When a guardrail with `action=block` fails, it raises `GuardrailError`:

```python
from selectools.guardrails import GuardrailError

try:
    result = agent.ask("Tell me about politics")
except GuardrailError as e:
    print(f"Blocked by: {e.guardrail_name}")
    print(f"Reason: {e.reason}")
```

---

## Trace Integration

Guardrail activations appear in the execution trace:

```python
result = agent.ask("Some input")
for step in result.trace:
    if step.type == "guardrail":
        print(f"Guardrail fired: {step.summary}")
```

---

## API Reference

| Class | Description |
|---|---|
| `GuardrailsPipeline(input=[], output=[])` | Ordered pipeline of input and output guardrails |
| `Guardrail` | Base class — subclass and override `check()` |
| `GuardrailResult(passed, content, reason)` | Result of a single check |
| `GuardrailError(guardrail_name, reason)` | Raised when `action=block` fails |
| `GuardrailAction.BLOCK` | Raise exception on failure |
| `GuardrailAction.REWRITE` | Return sanitised content |
| `GuardrailAction.WARN` | Log warning and continue |
| `TopicGuardrail(deny=[...])` | Keyword-based topic blocking |
| `PIIGuardrail(detect=[...], action=...)` | PII detection and redaction |
| `ToxicityGuardrail(threshold=0.0)` | Keyword-based toxicity scoring |
| `FormatGuardrail(require_json=True)` | JSON/length format validation |
| `LengthGuardrail(max_chars=..., max_words=...)` | Content length enforcement |

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 20 | [`20_customer_support_bot.py`](https://github.com/johnnichev/selectools/blob/main/examples/20_customer_support_bot.py) | Customer support bot with guardrails, PII redaction, and memory |
| 29 | [`29_guardrails.py`](https://github.com/johnnichev/selectools/blob/main/examples/29_guardrails.py) | Complete guardrails demo with topic blocking, PII, and toxicity |
