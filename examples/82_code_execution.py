#!/usr/bin/env python3
"""
Code Execution Tools -- run Python and shell commands from agents.

No API key needed. Demonstrates the execute_python and execute_shell tools.

WARNING: These tools execute code on your local machine. Do not use with
untrusted input without sandboxing.

Run: python examples/82_code_execution.py
"""

from selectools.toolbox.code_tools import execute_python, execute_shell

print("=== Code Execution Tools Example ===\n")

# 1. Execute Python code
print("--- Python Execution ---")
result = execute_python.function("import math; print(f'Pi = {math.pi:.6f}')")
print(f"Result: {result}")

# 2. Multi-line Python
result = execute_python.function(
    """
data = [1, 2, 3, 4, 5]
total = sum(data)
avg = total / len(data)
print(f"Sum: {total}, Average: {avg}")
"""
)
print(f"Multi-line: {result}")

# 3. Shell commands
print("--- Shell Execution ---")
result = execute_shell.function("echo 'Hello from shell' && date")
print(f"Shell: {result}")

# 4. With timeout
result = execute_python.function("print('fast!')", timeout=5)
print(f"With timeout: {result}")

# 5. Error handling
result = execute_python.function("1/0")
print(f"Error output: {result[:100]}")

# 6. Agent integration pattern
print(
    """
--- Agent Pattern ---
from selectools import Agent
from selectools.providers import OpenAIProvider
from selectools.toolbox.code_tools import execute_python

agent = Agent(
    tools=[execute_python],
    provider=OpenAIProvider(),
)
result = agent.run("Calculate the first 10 Fibonacci numbers")
# Agent writes and executes Python code to solve the task
"""
)

print("Done!")
