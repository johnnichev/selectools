# Models Module

**File:** `src/selectools/models.py`
**Classes:** `ModelInfo`
**Constants:** `ALL_MODELS`, `MODELS_BY_ID`, `OpenAI`, `Anthropic`, `Gemini`, `Ollama`, `Cohere`

## Table of Contents

1. [Overview](#overview)
2. [Model Registry System](#model-registry-system)
3. [Model Classes](#model-classes)
4. [Usage Patterns](#usage-patterns)
5. [Model Metadata](#model-metadata)
6. [Implementation](#implementation)

---

## Overview

The **Models** module provides a **single source of truth** for all supported LLM and embedding models. It includes:

- 130+ models across 5 providers
- Pricing per 1M tokens
- Context windows
- Max output tokens
- Type-safe constants with IDE autocomplete

### Why a Model Registry?

**Before:**

```python
# ❌ Error-prone
provider = OpenAIProvider(default_model="gpt-4o-mini")  # Typo?
# ❌ No pricing info
# ❌ No autocomplete
```

**After:**

```python
# ✅ Type-safe with autocomplete
from selectools.models import OpenAI

provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)

# ✅ Access metadata
model = OpenAI.GPT_4O_MINI
print(f"Cost: ${model.prompt_cost}/${model.completion_cost} per 1M tokens")
print(f"Context: {model.context_window:,} tokens")
```

---

## Model Registry System

### ModelInfo Dataclass

```python
@dataclass(frozen=True)
class ModelInfo:
    id: str                 # Model identifier (e.g., "gpt-4o")
    provider: str           # "openai", "anthropic", "gemini", "ollama"
    type: ModelType         # "chat", "embedding", "image", "audio"
    prompt_cost: float      # USD per 1M input tokens
    completion_cost: float  # USD per 1M output tokens
    max_tokens: int         # Maximum output tokens
    context_window: int     # Maximum context length
```

### Registry Structure

```python
# Typed model constants
OpenAI.GPT_4O              # ModelInfo instance
Anthropic.SONNET_3_5       # ModelInfo instance
Gemini.FLASH_2_0           # ModelInfo instance

# Complete list
ALL_MODELS                 # List[ModelInfo] - all 130+ models

# Quick lookup
MODELS_BY_ID               # Dict[str, ModelInfo] - O(1) lookup
```

---

## Model Classes

### OpenAI Models (65 models)

```python
from selectools.models import OpenAI

# GPT-5 Series (Latest)
OpenAI.GPT_5_1              # $1.25 / $10.00 per 1M tokens
OpenAI.GPT_5_MINI           # $0.25 / $2.00 per 1M tokens
OpenAI.GPT_5_NANO           # $0.05 / $0.40 per 1M tokens

# GPT-4o Series
OpenAI.GPT_4O               # $2.50 / $10.00 per 1M tokens
OpenAI.GPT_4O_MINI          # $0.15 / $0.60 per 1M tokens ⭐ Best value

# o-series (Reasoning)
OpenAI.O1                   # $15.00 / $60.00 per 1M tokens
OpenAI.O3_MINI              # $1.10 / $4.40 per 1M tokens

# GPT-4 Turbo
OpenAI.GPT_4_TURBO          # $10.00 / $30.00 per 1M tokens

# GPT-3.5 Turbo
OpenAI.GPT_3_5_TURBO        # $0.50 / $1.50 per 1M tokens

# Embeddings
OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL  # $0.02 per 1M tokens ⭐
OpenAI.Embeddings.TEXT_EMBEDDING_3_LARGE  # $0.13 per 1M tokens
OpenAI.Embeddings.ADA_002                 # $0.10 per 1M tokens
```

### Anthropic Models (18 models)

```python
from selectools.models import Anthropic

# Claude 4.5 Series
Anthropic.OPUS_4_5          # $5.00 / $25.00 per 1M tokens
Anthropic.SONNET_4_5        # $3.00 / $15.00 per 1M tokens
Anthropic.HAIKU_4_5         # $1.00 / $5.00 per 1M tokens

# Claude 3.5 Series
Anthropic.SONNET_3_5_20241022  # $3.00 / $15.00 per 1M tokens ⭐
Anthropic.HAIKU_3_5_20241022   # $0.80 / $4.00 per 1M tokens

# Embeddings (Voyage AI)
Anthropic.Embeddings.VOYAGE_3       # $0.06 per 1M tokens
Anthropic.Embeddings.VOYAGE_3_LITE  # $0.02 per 1M tokens
```

### Gemini Models (26 models)

```python
from selectools.models import Gemini

# Gemini 2.5 Series
Gemini.PRO_2_5              # $1.25 / $10.00 per 1M tokens
Gemini.FLASH_2_5            # $0.30 / $2.50 per 1M tokens
Gemini.FLASH_LITE_2_5       # $0.10 / $0.40 per 1M tokens

# Gemini 2.0 Series
Gemini.FLASH_2_0            # $0.10 / $0.40 per 1M tokens ⭐ Great value

# Gemini 1.5 Series
Gemini.PRO_1_5              # $1.25 / $5.00 per 1M tokens
Gemini.FLASH_1_5            # $0.075 / $0.30 per 1M tokens

# Embeddings
Gemini.Embeddings.EMBEDDING_004  # FREE ⭐⭐⭐
Gemini.Embeddings.EMBEDDING_001  # FREE
```

### Ollama Models (13 models)

```python
from selectools.models import Ollama

# All FREE (local execution)
Ollama.LLAMA_3_2            # Local, FREE
Ollama.LLAMA_3_1            # Local, FREE
Ollama.MISTRAL              # Local, FREE
Ollama.CODELLAMA            # Local, FREE ⭐ For coding
Ollama.PHI                  # Local, FREE
```

### Cohere Models (3 models)

```python
from selectools.models import Cohere

# Embeddings only
Cohere.Embeddings.EMBED_V3                # $0.10 per 1M tokens
Cohere.Embeddings.EMBED_MULTILINGUAL_V3   # $0.10 per 1M tokens ⭐ 100+ languages
Cohere.Embeddings.EMBED_V3_LIGHT          # $0.10 per 1M tokens
```

---

## Usage Patterns

### With Providers

```python
from selectools import OpenAIProvider, AnthropicProvider, GeminiProvider
from selectools.models import OpenAI, Anthropic, Gemini

# OpenAI
provider = OpenAIProvider(default_model=OpenAI.GPT_4O_MINI.id)

# Anthropic
provider = AnthropicProvider(default_model=Anthropic.SONNET_3_5_20241022.id)

# Gemini
provider = GeminiProvider(default_model=Gemini.FLASH_2_0.id)
```

### With Agent Config

```python
from selectools import Agent, AgentConfig
from selectools.models import OpenAI

config = AgentConfig(
    model=OpenAI.GPT_4O_MINI.id,
    temperature=0.0,
    max_tokens=OpenAI.GPT_4O_MINI.max_tokens
)

agent = Agent(tools=[...], provider=provider, config=config)
```

### Accessing Model Metadata

```python
from selectools.models import OpenAI

model = OpenAI.GPT_4O_MINI

print(f"Model ID: {model.id}")
print(f"Provider: {model.provider}")
print(f"Type: {model.type}")
print(f"Prompt cost: ${model.prompt_cost} per 1M tokens")
print(f"Completion cost: ${model.completion_cost} per 1M tokens")
print(f"Max output: {model.max_tokens:,} tokens")
print(f"Context window: {model.context_window:,} tokens")

# Output:
# Model ID: gpt-4o-mini
# Provider: openai
# Type: chat
# Prompt cost: $0.15 per 1M tokens
# Completion cost: $0.60 per 1M tokens
# Max output: 16,384 tokens
# Context window: 128,000 tokens
```

### Calculating Costs

```python
from selectools.pricing import calculate_cost
from selectools.models import OpenAI

cost = calculate_cost(
    model=OpenAI.GPT_4O_MINI.id,
    prompt_tokens=1000,
    completion_tokens=500
)

# Or manually:
model = OpenAI.GPT_4O_MINI
cost = (1000 / 1_000_000 * model.prompt_cost) + (500 / 1_000_000 * model.completion_cost)
```

### Quick Lookup

```python
from selectools.models import MODELS_BY_ID

# O(1) lookup
model = MODELS_BY_ID["gpt-4o-mini"]
print(f"Cost: ${model.prompt_cost}/${model.completion_cost}")

# Check if model exists
if "gpt-99" in MODELS_BY_ID:
    print("Model supported")
else:
    print("Model not in registry")
```

### List All Models

```python
from selectools.models import ALL_MODELS

# All 130+ models
print(f"Total models: {len(ALL_MODELS)}")

# Filter by provider
openai_models = [m for m in ALL_MODELS if m.provider == "openai"]
print(f"OpenAI models: {len(openai_models)}")

# Filter by type
embedding_models = [m for m in ALL_MODELS if m.type == "embedding"]
print(f"Embedding models: {len(embedding_models)}")

# Sort by cost
cheapest = sorted(ALL_MODELS, key=lambda m: m.prompt_cost)[:5]
for model in cheapest:
    print(f"{model.id}: ${model.prompt_cost}")
```

---

## Model Metadata

### Complete Example

```python
from selectools.models import OpenAI

model = OpenAI.GPT_4O

# Core identification
model.id                # "gpt-4o"
model.provider          # "openai"
model.type              # "chat"

# Pricing (USD per 1M tokens)
model.prompt_cost       # 2.50
model.completion_cost   # 10.00

# Capabilities
model.max_tokens        # 16384 (max output)
model.context_window    # 128000 (max input+output)

# Example calculation
input_tokens = 50000
output_tokens = 5000

input_cost = input_tokens / 1_000_000 * model.prompt_cost
output_cost = output_tokens / 1_000_000 * model.completion_cost
total_cost = input_cost + output_cost

print(f"Total cost: ${total_cost:.6f}")  # $0.175000
```

### ModelType Enum

```python
from selectools.models import ModelType

ModelType = Literal["chat", "embedding", "image", "audio", "multimodal"]

# Chat models
OpenAI.GPT_4O.type == "chat"

# Embedding models
OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.type == "embedding"

# Audio models
OpenAI.GPT_REALTIME.type == "audio"

# Multimodal models
OpenAI.GPT_4_1106_VISION_PREVIEW.type == "multimodal"
```

---

## Implementation

### Model Definition

```python
# In models.py

class OpenAI:
    GPT_4O_MINI = ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        type="chat",
        prompt_cost=0.15,
        completion_cost=0.60,
        max_tokens=16384,
        context_window=128000,
    )

    class Embeddings:
        TEXT_EMBEDDING_3_SMALL = ModelInfo(
            id="text-embedding-3-small",
            provider="openai",
            type="embedding",
            prompt_cost=0.02,
            completion_cost=0.0,  # Embeddings don't have completion cost
            max_tokens=8191,
            context_window=8191,
        )
```

### Registry Generation

```python
def _collect_all_models() -> List[ModelInfo]:
    """Collect all model definitions from provider classes."""
    models = []

    for provider_class in [OpenAI, Anthropic, Gemini, Ollama, Cohere]:
        for attr_name in dir(provider_class):
            if attr_name.startswith("_"):
                continue

            attr = getattr(provider_class, attr_name)

            if isinstance(attr, ModelInfo):
                models.append(attr)
            elif isinstance(attr, type) and attr_name == "Embeddings":
                # Nested Embeddings class
                for embed_attr_name in dir(attr):
                    if embed_attr_name.startswith("_"):
                        continue
                    embed_attr = getattr(attr, embed_attr_name)
                    if isinstance(embed_attr, ModelInfo):
                        models.append(embed_attr)

    return models

ALL_MODELS = _collect_all_models()
MODELS_BY_ID = {model.id: model for model in ALL_MODELS}
```

---

## Best Practices

### 1. Use Model Constants

```python
# ✅ Good - Type-safe, autocomplete
from selectools.models import OpenAI
model = OpenAI.GPT_4O_MINI.id

# ❌ Bad - String literals (typo-prone)
model = "gpt-4o-mini"
```

### 2. Check Model Costs Before Using

```python
from selectools.models import OpenAI

model = OpenAI.O1  # Expensive reasoning model
print(f"Warning: This model costs ${model.prompt_cost}/${model.completion_cost} per 1M tokens")

if model.prompt_cost > 10.0:
    print("Consider using a cheaper alternative")
```

### 3. Choose Appropriate Model for Task

```python
from selectools.models import OpenAI

# Simple tasks
config = AgentConfig(model=OpenAI.GPT_4O_MINI.id)  # $0.15/$0.60

# Complex reasoning
config = AgentConfig(model=OpenAI.O1.id)  # $15.00/$60.00

# Embeddings
from selectools.embeddings import OpenAIEmbeddingProvider
embedder = OpenAIEmbeddingProvider(model=OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.id)
```

### 4. Validate Model IDs

```python
from selectools.models import MODELS_BY_ID

user_model = "gpt-4o-super"  # User input

if user_model not in MODELS_BY_ID:
    raise ValueError(f"Unknown model: {user_model}")

model_info = MODELS_BY_ID[user_model]
```

---

## Cost Optimization

### Model Comparison

```python
from selectools.models import OpenAI, Anthropic, Gemini

# Budget options
print("Budget chat models:")
print(f"  OpenAI GPT-4o-mini: ${OpenAI.GPT_4O_MINI.prompt_cost}/${OpenAI.GPT_4O_MINI.completion_cost}")
print(f"  Gemini Flash 2.0: ${Gemini.FLASH_2_0.prompt_cost}/${Gemini.FLASH_2_0.completion_cost}")
print(f"  Anthropic Haiku 3.5: ${Anthropic.HAIKU_3_5_20241022.prompt_cost}/${Anthropic.HAIKU_3_5_20241022.completion_cost}")

# Output:
# Budget chat models:
#   OpenAI GPT-4o-mini: $0.15/$0.60
#   Gemini Flash 2.0: $0.10/$0.40
#   Anthropic Haiku 3.5: $0.80/$4.00
```

### Embedding Costs

```python
from selectools.models import OpenAI, Anthropic, Gemini, Cohere

print("Embedding costs:")
print(f"  Gemini: ${Gemini.Embeddings.EMBEDDING_004.prompt_cost} (FREE)")
print(f"  OpenAI small: ${OpenAI.Embeddings.TEXT_EMBEDDING_3_SMALL.prompt_cost}")
print(f"  Voyage lite: ${Anthropic.Embeddings.VOYAGE_3_LITE.prompt_cost}")
print(f"  Cohere: ${Cohere.Embeddings.EMBED_V3.prompt_cost}")

# Output:
# Embedding costs:
#   Gemini: $0.0 (FREE)
#   OpenAI small: $0.02
#   Voyage lite: $0.02
#   Cohere: $0.1
```

---

## Testing

```python
def test_model_registry():
    from selectools.models import OpenAI, ALL_MODELS, MODELS_BY_ID

    # Test model constant
    model = OpenAI.GPT_4O_MINI
    assert model.id == "gpt-4o-mini"
    assert model.provider == "openai"
    assert model.type == "chat"
    assert model.prompt_cost > 0
    assert model.context_window > 0

    # Test registry
    assert len(ALL_MODELS) >= 130
    assert "gpt-4o-mini" in MODELS_BY_ID

    # Test lookup
    looked_up = MODELS_BY_ID["gpt-4o-mini"]
    assert looked_up.id == model.id

def test_pricing_calculation():
    from selectools.models import OpenAI
    from selectools.pricing import calculate_cost

    model = OpenAI.GPT_4O_MINI
    cost = calculate_cost(model.id, prompt_tokens=1000, completion_tokens=500)

    # Manual calculation
    expected = (1000 / 1_000_000 * model.prompt_cost) + (500 / 1_000_000 * model.completion_cost)

    assert abs(cost - expected) < 0.000001
```

---

## Updating Models

When new models are released:

1. **Add to appropriate class:**

```python
class OpenAI:
    NEW_MODEL = ModelInfo(
        id="gpt-99",
        provider="openai",
        type="chat",
        prompt_cost=1.0,
        completion_cost=5.0,
        max_tokens=32768,
        context_window=256000,
    )
```

2. **Update registry** (automatic via `_collect_all_models()`)

3. **Update documentation**

4. **Add tests**

---

## Further Reading

- [Providers Module](PROVIDERS.md) - Using models with providers
- [Usage Module](USAGE.md) - Cost tracking
- [Pricing Module](../ARCHITECTURE.md#support-systems) - Cost calculation

---

**Congratulations!** You've completed the selectools implementation documentation. Return to [ARCHITECTURE.md](../ARCHITECTURE.md) for the system overview.
