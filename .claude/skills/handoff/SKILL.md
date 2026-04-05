---
name: handoff
description: Auto-fill HANDOFF.md with current session state from git, then suggest /clear for a fresh start
---

# Session Handoff

Auto-fill `HANDOFF.md` with the current session state so the next session can resume.

## Steps

1. Run `git status --short` to find modified/untracked files
2. Run `git log --oneline -5` to get recent commits
3. Run `git branch --show-current` to get current branch
4. Run `python3 -m pytest tests/ -x -q --tb=no 2>&1 | tail -1` to check test status

5. Write `HANDOFF.md` with:

```markdown
# Session Handoff

## What I Was Doing
[Summarize from conversation context - what was the user working on?]

## Current State
- Branch: [from git branch]
- Last commit: [from git log]
- Tests passing: [from pytest output]
- Files in progress: [from git status]

## What's Left
1. [Infer from conversation - what hasn't been done yet?]

## Key Decisions Made
- [List decisions from this session that aren't in the code]

## Watch Out For
- [Any gotchas or context that would be lost]
```

6. Tell the user: "HANDOFF.md updated. Run `/clear` to start fresh. The next session will read HANDOFF.md automatically."
