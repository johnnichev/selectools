# Doc Audit

Deploy parallel QA sub-agents to find documentation inconsistencies, stale counts, missing references, and improvement opportunities. Then fix everything found.

## Resolve Scope

- `all` → deploy 4 agents (one per category below)
- `counts` → 1 agent: stale numbers across all docs
- `content` → 1 agent: missing features, examples, outdated descriptions
- `links` → 1 agent: broken links, missing nav entries, dead references
- `quality` → 1 agent: unclear writing, missing code examples, incomplete sections

## Agent Template

Each agent reads ALL documentation files and cross-references against the actual codebase.

### Agent 1: Count Consistency
Verify these numbers match reality across ALL docs (README.md, CLAUDE.md, docs/index.md, CHANGELOG.md, CONTRIBUTING.md, landing/index.html, docs/ARCHITECTURE.md, docs/QUICKSTART.md):
- Test count (run `pytest --co -q` to get actual)
- Example count (run `ls examples/*.py | wc -l`)
- Model count (grep MODELS_BY_ID or count ModelInfo)
- StepType count (count enum members)
- Observer event count (count methods on AgentObserver/AsyncAgentObserver)
- Tool count (count toolbox tools)
- Evaluator count
- Provider count

### Agent 2: Content Completeness
For EVERY feature in the codebase, verify it's mentioned in:
- README.md "What's Included" / "Why Selectools" / feature table
- README.md examples section (all 61 examples should be referenced or at least the count)
- docs/index.md feature table
- docs/QUICKSTART.md (if user-facing)
- CLAUDE.md codebase structure (every source file)

Check specifically:
- Are Pipeline, @step, parallel(), branch() in README feature table?
- Are all 61 examples listed or counted?
- Are v0.18.0 features (orchestration + pipelines) prominently featured?
- Is the Apache-2.0 license mentioned everywhere it should be?
- Are all providers mentioned (OpenAI, Anthropic, Gemini, Ollama, Fallback)?

### Agent 3: Link & Navigation Audit
- Run `mkdocs build` and report ALL warnings
- Check every `[link](path)` in README.md resolves
- Check mkdocs.yml nav covers all docs/modules/*.md files
- Check CHANGELOG.md is synced with docs/CHANGELOG.md
- Check CONTRIBUTING.md is synced with docs/CONTRIBUTING.md
- Verify GitHub URLs point to correct repo

### Agent 4: Quality & Clarity
- Is the README.md "Quick Start" section current and working?
- Are code examples in README.md syntactically correct?
- Is the "What's New" section ordered correctly (newest first)?
- Are there outdated references to old versions or removed features?
- Is the install section current (pip extras)?
- Are there TODO/FIXME/placeholder texts left in docs?

## Output Format

Each agent reports:

```
## [Category] Doc Audit Report

### MUST FIX (blocks release)
| # | File | Issue | Fix |

### SHOULD FIX (improves quality)
| # | File | Issue | Fix |

### NICE TO HAVE
| # | File | Issue | Fix |
```

## After All Agents Complete

1. Compile master report
2. Deduplicate
3. Fix ALL "MUST FIX" and "SHOULD FIX" items
4. Fix "NICE TO HAVE" if quick
5. Rebuild mkdocs and verify clean
6. Present summary to user
