#!/bin/bash
# ENGRAM_STATUSLINE — do not remove this marker (used by installer for idempotency)
# Engram statusline for Claude Code
# If user had a prior statusline, it was moved to ~/.claude/statusline-user.sh
# and this script calls it first, then appends Engram memory lines.
input=$(cat)

# --- User's original statusline (preserved during install) ---
USER_SL="$HOME/.claude/statusline-user.sh"
if [ -x "$USER_SL" ]; then
  echo "$input" | "$USER_SL"
else
  # Default model + context bar
  MODEL=$(echo "$input" | jq -r '.model.display_name // "unknown"')
  PCT=$(echo "$input" | jq -r '.context_window.used_percentage // 0' | cut -d. -f1)
  BWIDTH=20
  FILLED=$((PCT * BWIDTH / 100))
  EMPTY=$((BWIDTH - FILLED))
  BAR=""
  [ "$FILLED" -gt 0 ] && printf -v FILL "%${FILLED}s" && BAR="${FILL// /▓}"
  [ "$EMPTY" -gt 0 ] && printf -v PAD "%${EMPTY}s" && BAR="${BAR}${PAD// /░}"
  echo "$MODEL [$BAR] ${PCT}%"
fi

# --- Engram: memory stats + session recall ---
SESSION_ID=$(echo "$input" | jq -r '.session_id // ""')
DIM=$'\033[2m'
RST=$'\033[0m'

# Memory counts (cached 30s, per-project to avoid cross-session clobbering)
DB="$HOME/.claude/memory.sqlite"
CWD=$(echo "$input" | jq -r '.workspace.project_dir // ""')
PROJECT=$(basename "$CWD" 2>/dev/null)
STATS_CACHE="/tmp/engram-stats-${PROJECT:-unknown}.cache"

mem_label=""
if [ -f "$DB" ]; then
  use_cache=0
  if [ -f "$STATS_CACHE" ]; then
    cache_age=$(( $(date +%s) - $(stat -f %m "$STATS_CACHE" 2>/dev/null || stat -c %Y "$STATS_CACHE" 2>/dev/null || echo 0) ))
    [ "$cache_age" -lt 30 ] && use_cache=1
  fi

  if [ "$use_cache" -eq 1 ]; then
    mem_label=$(cat "$STATS_CACHE")
  else
    TOTAL=$(sqlite3 "$DB" "SELECT COUNT(*) FROM Memory;" 2>/dev/null || echo "?")
    if [ -n "$PROJECT" ] && [ "$PROJECT" != "" ]; then
      PROJ_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM Memory WHERE project='$PROJECT';" 2>/dev/null || echo "0")
      mem_label="🧠 ${TOTAL} (${PROJ_COUNT} ${PROJECT})"
    else
      mem_label="🧠 ${TOTAL}"
    fi
    echo "$mem_label" > "$STATS_CACHE"
  fi
fi

echo "${DIM}${mem_label}${RST}"

# Session recall log (header + 5 direct + 3 connected + saves = ~12 lines max)
RECALL_LOG="$HOME/.claude/memory-logs/recall-${SESSION_ID}.log"
if [ -n "$SESSION_ID" ] && [ -f "$RECALL_LOG" ]; then
  head -1 "$RECALL_LOG"
  grep "•" "$RECALL_LOG" | head -5
  grep "├" "$RECALL_LOG" | head -3
  SAVES=$(grep "+" "$RECALL_LOG" | grep -v "recalled" | tail -3)
  [ -n "$SAVES" ] && echo "$SAVES"
fi
