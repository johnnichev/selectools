#!/usr/bin/env python3
"""
Unified Memory — Tiered memory with auto-promotion.

UnifiedMemory orchestrates the four existing memory systems into one
lifecycle: short-term (ConversationMemory rolling window), long-term
(KnowledgeMemory, auto-promoted by importance), entity (EntityMemory,
optional), and episodic (date-keyed history with retention pruning).

No API key needed. Runs entirely offline — importance scoring is rule-based
by default, and compaction falls back to truncation without a summarizer.

Prerequisites: pip install selectools
Run: python examples/106_unified_memory.py
"""

from selectools import ImportanceRule, UnifiedMemory, score_importance


def main() -> None:
    # Zero-arg default: in-memory tiers, rule-based scoring, no LLM required.
    memory = UnifiedMemory(
        importance_threshold=0.7,
        short_term_limit=6,  # tiny window so promotion triggers quickly
        long_term_limit=1000,
        episodic_retention_days=30,
        auto_promote=True,
    )

    # --- Rule-based importance scoring (the promotion gate) ---

    print("=== Importance scores (default rule table) ===\n")
    for text in (
        "My name is John Niche",  # identity rule -> 0.9
        "I prefer dark roast coffee",  # preference rule -> 0.75
        "I live in Brusque, Brazil",  # location rule -> 0.6
        "The weather is mild today",  # no rule -> base 0.3
    ):
        print(f"  {score_importance(text):.2f}  {text}")

    # --- Add turns: STM fills, old turns age out and get auto-promoted ---

    print("\n=== Adding conversation turns ===\n")
    turns = [
        ("My name is John Niche", "Nice to meet you, John!"),
        ("I prefer dark roast coffee", "Noted — dark roast it is."),
        ("I live in Brusque, Brazil", "Beautiful part of Santa Catarina."),
        ("What's a good breakfast?", "Eggs and fruit pair well with coffee."),
        ("Any book suggestions?", "Try a Brandon Sanderson novel."),
    ]
    for user, assistant in turns:
        memory.add_turn(user, assistant)
    print(memory)

    # Items above the 0.7 threshold were promoted as they aged out of the
    # 6-message window. Location (0.6) stays short-term/episodic only.
    print("\n=== Long-term entries (auto-promoted) ===")
    for entry in memory.long_term.store.query(limit=10):
        print(f"  [{entry.category}] importance={entry.importance:.2f}  {entry.content}")

    # --- Explicit consolidation scores whatever is still in STM ---

    promoted = memory.consolidate()
    print(f"\nconsolidate() promoted {promoted} more item(s) — dedup prevents repeats.")

    # --- Federated recall across tiers ---

    print("\n=== recall('what coffee does the user like?') ===")
    for result in memory.recall("what coffee does the user like?"):
        print(f"  {result.score:.2f} [{result.source}] {result.content}")

    # --- Assembled context with compaction ---

    print("\n=== assemble_context(max_tokens=4000) ===\n")
    print(memory.assemble_context(max_tokens=4000))

    # When the context exceeds 70% of the budget, older STM is compacted.
    # Without a summarizer= callable, a truncation marker is used:
    print("\n=== assemble_context(max_tokens=120) — compaction kicks in ===\n")
    print(memory.assemble_context(max_tokens=120))

    # --- Custom rule table ---

    rules = [ImportanceRule(name="project", pattern=r"selectools", score=0.95)]
    print("\n=== Custom rule table ===")
    print(f"  'selectools ships today' -> {score_importance('selectools ships today', rules):.2f}")


if __name__ == "__main__":
    main()
