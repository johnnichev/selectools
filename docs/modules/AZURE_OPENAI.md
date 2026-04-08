---
description: "Azure OpenAI Service provider — use selectools agents with Azure-deployed GPT-4 / GPT-4o models"
tags:
  - providers
  - azure
  - openai
---

# Azure OpenAI Provider

**Import:** `from selectools import AzureOpenAIProvider`
**Stability:** beta
**Added in:** v0.21.0

`AzureOpenAIProvider` lets selectools talk to OpenAI models deployed on Azure
OpenAI Service. It extends `OpenAIProvider` and uses the OpenAI SDK's built-in
`AzureOpenAI` client, so you get every feature of the regular OpenAI provider
(streaming, tool calling, structured output, multimodal) without having to
maintain a separate code path.

```python title="azure_openai_quick.py"
from selectools import Agent, AzureOpenAIProvider, tool

@tool()
def get_time() -> str:
    """Return the current time."""
    from datetime import datetime
    return datetime.utcnow().isoformat()

provider = AzureOpenAIProvider(
    azure_endpoint="https://my-resource.openai.azure.com",
    api_key="<your-azure-key>",
    azure_deployment="gpt-4o",  # your Azure deployment name
)

agent = Agent(tools=[get_time], provider=provider)
print(agent.run("What time is it?").content)
```

!!! tip "See Also"
    - [Providers](PROVIDERS.md) - All available LLM providers
    - [Fallback Provider](PROVIDERS.md#fallback) - Use Azure as a fallback for the public OpenAI API

---

## Install

No new dependencies. Azure support uses the same `openai>=1.30.0` package that
ships as a core selectools dependency.

```bash
pip install selectools  # Azure already supported
```

---

## Constructor

```python
AzureOpenAIProvider(
    azure_endpoint: str | None = None,
    api_key: str | None = None,
    api_version: str = "2024-10-21",
    azure_deployment: str | None = None,
    azure_ad_token: str | None = None,
)
```

| Parameter | Description |
|---|---|
| `azure_endpoint` | Azure resource endpoint (`https://<name>.openai.azure.com`). Falls back to `AZURE_OPENAI_ENDPOINT` env var. |
| `api_key` | Azure API key. Falls back to `AZURE_OPENAI_API_KEY`. Optional when `azure_ad_token` is set. |
| `api_version` | Azure OpenAI API version string. Defaults to a recent stable release. |
| `azure_deployment` | The deployment name to use as the default model (Azure uses deployment names, not OpenAI model IDs). Falls back to `AZURE_OPENAI_DEPLOYMENT`. |
| `azure_ad_token` | An Azure Active Directory token for AAD-based auth. When set, `api_key` is not required. |

---

## Environment Variables

`AzureOpenAIProvider()` with no arguments works if you set the standard Azure
env vars:

```bash
export AZURE_OPENAI_ENDPOINT="https://my-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
```

```python
provider = AzureOpenAIProvider()  # Reads everything from env
```

---

## Azure Deployments vs Model IDs

In the public OpenAI API you pass model IDs like `"gpt-4o"`. In Azure OpenAI you
pass **deployment names** that you create in the Azure Portal. selectools maps
the `azure_deployment` parameter to the `model` argument internally, so the rest
of your agent code is unchanged:

```python
# Same Agent code, swappable providers
agent = Agent(provider=OpenAIProvider(model="gpt-4o"))           # Public OpenAI
agent = Agent(provider=AzureOpenAIProvider(azure_deployment="gpt-4o"))  # Azure
```

---

## AAD Token Auth

For enterprise deployments using Azure Active Directory:

```python
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default").token

provider = AzureOpenAIProvider(
    azure_endpoint="https://my-resource.openai.azure.com",
    azure_deployment="gpt-4o",
    azure_ad_token=token,
)
```

---

## Inheritance

`AzureOpenAIProvider` extends `OpenAIProvider`, so it inherits everything:

- `complete()` / `acomplete()`
- `stream()` / `astream()`
- Tool calling, structured output, multimodal messages
- Token usage and cost tracking via `selectools.pricing`

Only `__init__` is overridden — to use the `AzureOpenAI` client class instead of
the regular `OpenAI` one.

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 86 | [`86_azure_openai.py`](https://github.com/johnnichev/selectools/blob/main/examples/86_azure_openai.py) | Azure OpenAI agent with deployment-name routing |
