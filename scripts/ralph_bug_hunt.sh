#!/usr/bin/env bash
# Ralph loop: hunt + fix per module until 3 consecutive clean passes.
#
# Usage:
#   bash scripts/ralph_bug_hunt.sh              # all 7 modules
#   bash scripts/ralph_bug_hunt.sh rag          # single module
#   bash scripts/ralph_bug_hunt.sh agent providers  # multiple modules
#
# Modules: agent | providers | tools | rag | memory | evals | security
#
# Exit codes:
#   0 — all modules reached 3 consecutive clean passes
#   1 — one or more modules hit the iteration cap without converging

set -euo pipefail

ALL_MODULES=("agent" "providers" "tools" "rag" "memory" "evals" "security")
MAX_ITER=10
REQUIRED_CLEAN=3
LOG_DIR="logs/ralph_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Resolve modules from args (default: all)
if [ "$#" -eq 0 ]; then
    MODULES=("${ALL_MODULES[@]}")
else
    MODULES=("$@")
fi

# Validate module names
for m in "${MODULES[@]}"; do
    valid=false
    for a in "${ALL_MODULES[@]}"; do
        [ "$m" = "$a" ] && valid=true && break
    done
    if [ "$valid" = false ]; then
        echo "ERROR: unknown module '$m'. Valid: ${ALL_MODULES[*]}" >&2
        exit 1
    fi
done

overall_pass=true

for module in "${MODULES[@]}"; do
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  Ralph loop: $module"
    echo "╚══════════════════════════════════════════════╝"

    clean_passes=0
    iter=0

    while [ "$clean_passes" -lt "$REQUIRED_CLEAN" ]; do
        iter=$((iter + 1))

        if [ "$iter" -gt "$MAX_ITER" ]; then
            echo ""
            echo "⚠  $module: hit max iterations ($MAX_ITER) without $REQUIRED_CLEAN clean passes"
            echo "   Manual review required. Logs: $LOG_DIR/${module}_iter*.log"
            overall_pass=false
            break
        fi

        log="$LOG_DIR/${module}_iter${iter}.log"
        echo ""
        echo "  [$module] iter $iter / max $MAX_ITER  (clean: $clean_passes/$REQUIRED_CLEAN)"
        echo "  Log: $log"

        # Run ralph-bug-hunt via Claude Code CLI
        # --print: non-interactive, output to stdout
        # --permission-mode acceptEdits: auto-accept file edits (no prompts)
        claude --print --permission-mode acceptEdits \
            "/ralph-bug-hunt $module" 2>&1 | tee "$log"

        last_line=$(tail -1 "$log")

        if echo "$last_line" | grep -q "RALPH_RESULT: CLEAN"; then
            clean_passes=$((clean_passes + 1))
            echo ""
            echo "  ✓ Clean pass $clean_passes/$REQUIRED_CLEAN for $module"
        elif echo "$last_line" | grep -q "RALPH_RESULT: FOUND"; then
            clean_passes=0
            found_summary=$(echo "$last_line" | sed 's/RALPH_RESULT: //')
            echo ""
            echo "  ✗ $found_summary — resetting clean counter for $module"
        else
            # Unexpected output — treat as non-clean to be safe
            clean_passes=0
            echo ""
            echo "  ✗ Unexpected sentinel output: '$last_line'"
            echo "    Treating as non-clean and retrying."
        fi
    done

    if [ "$clean_passes" -ge "$REQUIRED_CLEAN" ]; then
        echo ""
        echo "  ✅ $module: $REQUIRED_CLEAN consecutive clean passes achieved"
    fi
done

echo ""
echo "════════════════════════════════════════════"
if [ "$overall_pass" = true ]; then
    echo "✅ RALPH LOOP COMPLETE — all modules clean."
    echo "   Safe to proceed with release tagging."
    exit 0
else
    echo "❌ RALPH LOOP INCOMPLETE — one or more modules need manual review."
    echo "   DO NOT TAG THE RELEASE until the issues above are resolved."
    echo "   Logs saved to: $LOG_DIR/"
    exit 1
fi
