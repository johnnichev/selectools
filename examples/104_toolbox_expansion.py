#!/usr/bin/env python3
"""
Toolbox Expansion -- calculator, email, PDF, Slack, Notion, and Linear tools.

Fully offline: the calculator runs locally and the Slack call is mocked.
The API-backed tools (email, PDF, Slack, Notion, Linear) need their optional
deps: pip install selectools[toolbox]

Run: python examples/104_toolbox_expansion.py
"""

import sys
import types
from unittest.mock import MagicMock

from selectools.toolbox import get_tools_by_category
from selectools.toolbox.calculator_tools import evaluate_expression, unit_convert

print("=== Toolbox Expansion Example ===\n")

# 1. Calculator -- safe AST-based evaluation, no eval()/exec()
print("--- Calculator ---")
print(evaluate_expression.function("2 + 3 * 4"))
print(evaluate_expression.function("sqrt(16) + 2 ** 10"))
print(evaluate_expression.function("sin(pi / 2)"))

# Hostile input is rejected, never executed:
print(evaluate_expression.function("__import__('os').system('id')"))

print()
print(unit_convert.function(10, "km", "mi"))
print(unit_convert.function(100, "c", "f"))
print(unit_convert.function(1, "gib", "mb"))

# 2. Slack (mocked -- demonstrates the call shape without a token or network)
print("\n--- Slack (mocked) ---")
fake_client = MagicMock()
fake_client.chat_postMessage.return_value = {"ok": True, "ts": "1718000000.000100"}
fake_sdk = types.ModuleType("slack_sdk")
fake_errors = types.ModuleType("slack_sdk.errors")
fake_errors.SlackApiError = type("SlackApiError", (Exception,), {})
fake_sdk.WebClient = MagicMock(return_value=fake_client)
sys.modules["slack_sdk"] = fake_sdk
sys.modules["slack_sdk.errors"] = fake_errors

from selectools.toolbox.slack_tools import slack_send_message  # noqa: E402

print(slack_send_message.function("#deploys", "Release v0.23.0 is live", token="xoxb-demo"))

# 3. Category registry -- the new categories are first-class
print("\n--- New categories ---")
for category in ("calculator", "email", "pdf", "slack", "notion", "linear"):
    tools = get_tools_by_category(category)
    print(f"{category}: {', '.join(t.name for t in tools)}")

print(
    """
--- Agent Pattern ---
from selectools import Agent, AgentConfig
from selectools.providers import OpenAIProvider
from selectools.toolbox import get_tools_by_category

agent = Agent(
    tools=get_tools_by_category("calculator") + get_tools_by_category("slack"),
    provider=OpenAIProvider(),
    config=AgentConfig(max_iterations=5),
)
agent.ask("Convert 5 miles to km, then post the answer to #general")

Credentials come from env vars -- never hardcode them:
  SMTP_HOST / SMTP_USERNAME / SMTP_PASSWORD   (email)
  IMAP_HOST / IMAP_USERNAME / IMAP_PASSWORD   (inbox)
  SLACK_BOT_TOKEN                             (slack)
  NOTION_API_KEY                              (notion)
  LINEAR_API_KEY                              (linear)
"""
)
