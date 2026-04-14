# Agentic Memory (Memory-as-Tool)

**Stack:** Python 3.9+, src-layout, pytest, stdlib only (no new deps)
**Date:** 2026-04-13
**Status:** Draft

## Problem

Selectools memory is currently **passive** — `EntityMemory`, `KnowledgeMemory`, and `ConversationMemory` are wired into `_prepare_run()` to inject context automatically. The agent never decides *what* to remember or *when* to recall. Every competitor with "agentic memory" (Agno, mem0, LangChain) lets the agent store and retrieve via explicit tool calls, giving it agency over its own knowledge.

A partial implementation exists: `toolbox/memory_tools.py` has a `make_remember_tool()` factory for `KnowledgeMemory.remember()`. But there is no corresponding `recall` tool, no `EntityMemory` tool integration, no semantic search, and the existing factory isn't documented or exported from the top-level API.

## Solution

Expand `toolbox/memory_tools.py` into a complete agentic memory toolkit with factory functions that create `Tool` instances backed by existing memory classes:

| Tool | Backed by | Purpose |
|------|-----------|---------|
| `remember` | `KnowledgeMemory.remember()` | Store a fact with category and importance |
| `recall` | `KnowledgeStore.search()` | Semantic search over stored knowledge |
| `note_entity` | `EntityMemory.update()` | Explicitly register/update an entity |
| `recall_entities` | `EntityMemory.entities` | Retrieve known entities, optionally filtered |

Plus a convenience factory `make_memory_tools()` that returns all applicable tools for a given memory configuration.

The agent decides when to call these tools — no automatic injection needed when using agentic memory mode.

## Acceptance Criteria

- [ ] `make_remember_tool(knowledge: KnowledgeMemory) -> Tool` — existing, cleaned up: adds `importance` param (float, optional, default 0.5), adds `ttl_days` param (int, optional)
- [ ] `make_recall_tool(knowledge: KnowledgeMemory) -> Tool` — new: `recall(query: str, category: str = "", max_results: int = 5) -> str` returns formatted knowledge entries matching the query
- [ ] `make_note_entity_tool(entity_memory: EntityMemory) -> Tool` — new: `note_entity(name: str, entity_type: str, attributes: str = "") -> str` creates/updates an Entity
- [ ] `make_recall_entities_tool(entity_memory: EntityMemory) -> Tool` — new: `recall_entities(filter_type: str = "") -> str` returns formatted entity list
- [ ] `make_memory_tools(knowledge: Optional[KnowledgeMemory] = None, entity_memory: Optional[EntityMemory] = None) -> List[Tool]` — convenience factory, returns tools for whichever memory backends are provided
- [ ] `recall` tool uses `KnowledgeStore.search()` if the store supports it, falls back to `KnowledgeStore.list()` with client-side substring match
- [ ] All tools return `str` (required by tool executor contract)
- [ ] All tools work with both `FileKnowledgeStore` and `SQLiteKnowledgeStore`
- [ ] `attributes` param on `note_entity` accepts comma-separated `key=value` pairs (LLM-friendly format)
- [ ] Existing `make_remember_tool` behavior is preserved — no breaking changes to current users
- [ ] All factories exported from `selectools.toolbox` and top-level `selectools` package
- [ ] Stability marker: `@beta` on all new public functions; existing `make_remember_tool` stays as-is
- [ ] ≥95% test coverage on new/modified code
- [ ] Cookbook recipe: `docs/cookbook/agentic-memory.md` demonstrating remember + recall + entity tools in an agent
- [ ] One example in `examples/` showing a multi-turn conversation where the agent decides to remember and later recall

## Non-Goals

- Replacing passive memory injection — agentic memory is an **alternative mode**, not a replacement. Users can still use `AgentConfig.entity_memory` for auto-injection.
- Vector-store-backed semantic recall — `KnowledgeStore.search()` is keyword/metadata-based today. True vector search requires a `VectorStore` integration which is a separate feature.
- Cross-agent memory sharing — tools operate on the memory instance passed in. Sharing is the user's responsibility (pass same instance to multiple agents).
- Memory consolidation/summarization as a tool — the existing `ConversationMemory` summarize-on-trim handles this passively.
- Tool-level access control — any agent with the tool can read/write. No per-key permissions.

## Technical Approach

### Modified file: `src/selectools/toolbox/memory_tools.py`

Current state: ~65 lines with `make_remember_tool()` only.

**Changes:**

1. **Enhance `make_remember_tool`**: add `importance` and `ttl_days` parameters to the inner `_remember()` function. Parse `importance` from string (LLMs send strings). Backward compatible — new params are optional.

2. **Add `make_recall_tool`**:
```python
def make_recall_tool(knowledge: KnowledgeMemory) -> Tool:
    def _recall(query: str, category: str = "", max_results: int = 5) -> str:
        entries = knowledge._store.search(query, category=category or None, limit=max_results)
        if not entries:
            return "No matching memories found."
        return "\n---\n".join(
            f"[{e.category}] {e.content} (importance: {e.importance}, created: {e.created_at:%Y-%m-%d})"
            for e in entries
        )
```

3. **Add `make_note_entity_tool`**:
```python
def make_note_entity_tool(entity_memory: EntityMemory) -> Tool:
    def _note_entity(name: str, entity_type: str, attributes: str = "") -> str:
        attrs = {}
        if attributes:
            for pair in attributes.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    attrs[k.strip()] = v.strip()
        entity = Entity(name=name, entity_type=entity_type, attributes=attrs, ...)
        entity_memory.update([entity])
        return f"Entity '{name}' ({entity_type}) stored with {len(attrs)} attributes."
```

4. **Add `make_recall_entities_tool`**: format `entity_memory.entities` as a readable list, optionally filtered by `entity_type`.

5. **Add `make_memory_tools`**: introspect which backends are provided, return applicable tool list.

### KnowledgeStore.search() support

Check if `KnowledgeStore` protocol and implementations already have `search()`:

- `FileKnowledgeStore` — has `list()` with `category` filter. Add `search(query, category, limit)` that filters `list()` results by substring match on `content`.
- `SQLiteKnowledgeStore` — has `list()`. Add `search()` with `LIKE '%query%'` SQL (mark `# nosec B608` — query is user-knowledge content, not untrusted SQL).

### Other modified files

| File | Change |
|------|--------|
| `toolbox/__init__.py` | Export `make_recall_tool`, `make_note_entity_tool`, `make_recall_entities_tool`, `make_memory_tools` |
| `__init__.py` | Add new factories to top-level exports with `@beta` |
| `knowledge.py` | Add `search()` to `KnowledgeStore` protocol + both implementations |

## Dependencies

- `KnowledgeMemory` and `EntityMemory` — already exist and stable
- `Entity` dataclass — already in `entity_memory.py`
- `Tool` and `ToolParameter` — already in `tools/base.py`
- `make_remember_tool` — already in `toolbox/memory_tools.py`
- `KnowledgeStore` protocol — already in `knowledge.py`

## Risks

| Risk | Mitigation |
|------|-----------|
| LLMs send args as strings (e.g., `importance: "0.8"`) | Parse with `float()` / `int()` with try/except, same pattern as existing `persistent` param |
| `search()` on FileKnowledgeStore is O(N) substring match | Acceptable for <1000 entries. Document that SQLiteKnowledgeStore is recommended for large stores |
| `entity_memory.update()` requires LLM-extracted entities in current API | `note_entity` constructs an `Entity` directly, bypassing LLM extraction — this is intentional for agentic use |
| `attributes` string parsing is fragile | Simple `key=value` comma-split is LLM-friendly. Document the format in tool description. Edge cases (commas in values) are acceptable for v1 |
