#!/bin/bash
# Install the Engram statusline into Claude Code.
# Preserves the user's existing statusline (moved to ~/.claude/statusline-user.sh).
# Safe to re-run — detects if already installed and updates in place.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUSLINE_SRC="$SCRIPT_DIR/engram-statusline.sh"
STATUSLINE_DST="$HOME/.claude/statusline.sh"
SETTINGS_FILE="$HOME/.claude/settings.json"

# Ensure source exists
if [ ! -f "$STATUSLINE_SRC" ]; then
    echo "Error: engram-statusline.sh not found at $STATUSLINE_SRC"
    exit 1
fi

mkdir -p "$HOME/.claude/memory-logs"

if [ -f "$STATUSLINE_DST" ] && grep -q "ENGRAM_STATUSLINE" "$STATUSLINE_DST"; then
    # Already installed — update in place
    cp "$STATUSLINE_SRC" "$STATUSLINE_DST"
    chmod +x "$STATUSLINE_DST"
    echo "Updated Engram statusline"
else
    # Preserve user's existing statusline
    if [ -f "$STATUSLINE_DST" ]; then
        mv "$STATUSLINE_DST" "$HOME/.claude/statusline-user.sh"
        chmod +x "$HOME/.claude/statusline-user.sh"
        echo "Preserved existing statusline as ~/.claude/statusline-user.sh"
    fi
    cp "$STATUSLINE_SRC" "$STATUSLINE_DST"
    chmod +x "$STATUSLINE_DST"
    echo "Installed Engram statusline"
fi

# Register in settings.json
if [ -f "$SETTINGS_FILE" ] && command -v jq &>/dev/null; then
    jq '.statusLine = {"type": "command", "command": "'"$STATUSLINE_DST"'"}' "$SETTINGS_FILE" > "$SETTINGS_FILE.tmp"
    mv "$SETTINGS_FILE.tmp" "$SETTINGS_FILE"
    echo "Registered statusline in settings.json"
elif [ ! -f "$SETTINGS_FILE" ]; then
    echo '{"statusLine":{"type":"command","command":"'"$STATUSLINE_DST"'"}}' > "$SETTINGS_FILE"
    echo "Created settings.json with statusline"
else
    echo "Warning: jq not found. Add this to $SETTINGS_FILE manually:"
    echo '  "statusLine": {"type": "command", "command": "'"$STATUSLINE_DST"'"}'
fi

echo "Done. Restart Claude Code to see the new statusline."
