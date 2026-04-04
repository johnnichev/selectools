#!/usr/bin/env python3
"""
GitHub Tools -- search repos, read files, list issues from agents.

No API key needed (optional GITHUB_TOKEN increases rate limit from 60 to 5000/hr).
Read-only operations only.

Run: python examples/84_github_tools.py
"""

from selectools.toolbox.github_tools import github_get_file, github_list_issues, github_search_repos

print("=== GitHub Tools Example ===\n")

# Note: These tools make real API calls to GitHub.
# Uncomment to test:

# 1. Search repositories
# result = github_search_repos.function("python ai agent framework", max_results=3)
# print(f"Repos:\n{result}\n")

# 2. Get a file
# result = github_get_file.function("johnnichev/selectools", "README.md")
# print(f"File content:\n{result[:200]}...\n")

# 3. List issues
# result = github_list_issues.function("johnnichev/selectools", state="open", max_results=5)
# print(f"Issues:\n{result}\n")

# Show tool metadata
for tool in [github_search_repos, github_get_file, github_list_issues]:
    print(f"{tool.name}:")
    print(f"  {tool.description}")
    print()

print(
    """
--- Agent Pattern ---
from selectools import Agent
from selectools.providers import OpenAIProvider
from selectools.toolbox.github_tools import github_search_repos, github_get_file

agent = Agent(
    tools=[github_search_repos, github_get_file],
    provider=OpenAIProvider(),
)
result = agent.run("Find the top Python AI frameworks and read their README")
"""
)

print("Set GITHUB_TOKEN env var for higher rate limits (5000/hr vs 60/hr)")
print("Done!")
