---
name: handoff
description: Generate a project handoff document for transferring context to a new Claude instance
argument-hint: [focus-area]
---

# Project Handoff Generator

Generate a concise handoff document for the selectools project. If a focus area is specified ("$ARGUMENTS"), narrow the scope to that area. Otherwise, generate a full project handoff.

## Live Project State

- Version: !`grep -m1 __version__ src/selectools/__init__.py`
- Tests: !`pytest tests/ --collect-only -q 2>/dev/null | tail -1`
- Examples: !`ls examples/*.py | wc -l | tr -d ' '`
- Recent commits: !`git log --oneline -10`
- Current branch: !`git branch --show-current`
- Working tree: !`git status --short`

## Handoff Structure

Generate a markdown document with these sections:

### 1. Project Identity
- Name, version (from above), Python 3.9+, src-layout, pytest, MkDocs Material
- PyPI: https://pypi.org/project/selectools/
- Docs: https://johnnichev.github.io/selectools
- Repo: https://github.com/johnnichev/selectools

### 2. Current State
- Summarize what the recent commits show about current work
- Note any uncommitted changes (from working tree above)
- Note any WIP branches

### 3. Architecture Overview
Read `CLAUDE.md` for the codebase structure. Summarize the key module groups:
- Agent core (`agent/core.py`, `agent/config.py`)
- Providers (`providers/`)
- Tools (`tools/`, `toolbox/`)
- RAG (`rag/`)
- Memory (`memory.py`, `sessions.py`, `entity_memory.py`, `knowledge_graph.py`, `knowledge.py`)
- Enterprise (`guardrails/`, `audit.py`, `security.py`, `coherence.py`)

### 4. What's Next
Read `ROADMAP.md` and `MULTI_AGENT_PLAN.md` (if it exists). Summarize:
- What's completed (recent releases)
- What's in progress
- What's planned next

### 5. Known Pitfalls
Summarize the "Common Pitfalls" section from `CLAUDE.md` — these are past bugs that are easy to reintroduce.

### 6. Key Files to Read First
List the files a new Claude instance should read to get productive:
- `CLAUDE.md`, `src/selectools/__init__.py`, `src/selectools/agent/config.py`, `ROADMAP.md`, `CHANGELOG.md` (latest entry)

## Output Format
Write the handoff as clean markdown. Keep it under 300 lines. Focus on what a new instance needs to be productive immediately.
