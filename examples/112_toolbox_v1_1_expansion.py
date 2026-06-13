#!/usr/bin/env python3
"""
Toolbox v1.1 Expansion -- Discord, S3, browser, and image generation tools.

Fully offline: every external service is mocked. The real tools need their
optional deps:
    pip install selectools[toolbox]                      # Discord (requests)
    pip install selectools[aws]                          # S3 (boto3)
    pip install selectools[browser] && playwright install chromium
    # image generation uses the core openai dependency

Run: python examples/112_toolbox_v1_1_expansion.py
"""

import sys
import types
from unittest.mock import MagicMock

from selectools.toolbox import get_tools_by_category

print("=== Toolbox v1.1 Expansion Example ===\n")

# 1. Discord (mocked -- demonstrates the call shape without a token or network)
print("--- Discord (mocked) ---")
fake_requests = types.ModuleType("requests")
fake_exceptions = types.ModuleType("requests.exceptions")
fake_exceptions.RequestException = type("RequestException", (Exception,), {})
fake_requests.exceptions = fake_exceptions
response = MagicMock()
response.status_code = 200
response.json.return_value = {"id": "999888777666"}
fake_requests.post = MagicMock(return_value=response)
fake_requests.get = MagicMock(return_value=response)
sys.modules["requests"] = fake_requests
sys.modules["requests.exceptions"] = fake_exceptions

from selectools.toolbox.discord_tools import discord_send_message  # noqa: E402

print(discord_send_message.function("123456789012345678", "v1.1 is baking", token="bot-demo"))

# 2. S3 (mocked boto3 client)
print("\n--- S3 (mocked) ---")
s3_client = MagicMock()
s3_client.list_objects_v2.return_value = {
    "Contents": [
        {"Key": "reports/2026-06.csv", "Size": 1024},
        {"Key": "reports/2026-05.csv", "Size": 2048},
    ]
}
fake_boto3 = types.ModuleType("boto3")
fake_boto3.client = MagicMock(return_value=s3_client)
fake_botocore = types.ModuleType("botocore")
fake_botocore_exc = types.ModuleType("botocore.exceptions")
fake_botocore_exc.BotoCoreError = type("BotoCoreError", (Exception,), {})
fake_botocore_exc.ClientError = type("ClientError", (Exception,), {})
fake_botocore.exceptions = fake_botocore_exc
sys.modules["boto3"] = fake_boto3
sys.modules["botocore"] = fake_botocore
sys.modules["botocore.exceptions"] = fake_botocore_exc

from selectools.toolbox.s3_tools import s3_list_objects  # noqa: E402

print(s3_list_objects.function("analytics-bucket", prefix="reports/"))

# 3. Category registry -- the new categories are first-class
print("\n--- New categories ---")
for category in ("discord", "s3", "browser", "image"):
    tools = get_tools_by_category(category)
    print(f"{category}: {', '.join(t.name for t in tools)}")

print(
    """
--- Agent Pattern ---
from selectools import Agent, AgentConfig
from selectools.providers import OpenAIProvider
from selectools.toolbox import get_tools_by_category

agent = Agent(
    tools=get_tools_by_category("browser") + get_tools_by_category("image"),
    provider=OpenAIProvider(),
    config=AgentConfig(max_iterations=5),
)
agent.ask("Screenshot https://example.com, then generate a stylized version")
"""
)
