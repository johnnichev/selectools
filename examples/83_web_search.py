#!/usr/bin/env python3
"""
Web Search Tools -- search the web and scrape URLs.

No API key needed. Uses DuckDuckGo for search (no rate limits for moderate use).

Run: python examples/83_web_search.py
"""

from selectools.toolbox.search_tools import scrape_url, web_search

print("=== Web Search Tools Example ===\n")

# Note: These tools make real HTTP requests.
# Uncomment to test with live web access:

# 1. Web search (DuckDuckGo)
# result = web_search.function("Python AI agent frameworks 2026")
# print(f"Search results:\n{result[:500]}")

# 2. Scrape a URL
# result = scrape_url.function("https://example.com")
# print(f"Scraped:\n{result[:300]}")

# Show the API pattern
print("web_search tool:")
print(f"  Name: {web_search.name}")
print(f"  Description: {web_search.description}")
print(f"  Parameters: query (str), num_results (int, default=5)")

print(f"\nscrape_url tool:")
print(f"  Name: {scrape_url.name}")
print(f"  Description: {scrape_url.description}")
print(f"  Parameters: url (str), selector (str, optional CSS selector)")

print(
    """
--- Agent Pattern ---
from selectools import Agent
from selectools.providers import OpenAIProvider
from selectools.toolbox.search_tools import web_search, scrape_url

agent = Agent(
    tools=[web_search, scrape_url],
    provider=OpenAIProvider(),
)
result = agent.run("Search for the latest Python release and summarize")
"""
)

print("Done!")
