#!/bin/bash
set -e

INSTALL_DIR="$HOME/.claude/bin"

echo "Uninstalling memory server..."

# Deregister from Claude Code
env -u CLAUDECODE claude mcp remove memory 2>/dev/null || true

# Remove binary and bundles
rm -f "$INSTALL_DIR/memory"
rm -rf "$INSTALL_DIR/ClaudeMemory_ClaudeMemoryLib.bundle"
rm -rf "$INSTALL_DIR/swift-transformers_Hub.bundle"
rm -rf "$INSTALL_DIR/SwiftLM_SwiftLM.bundle"

# Remove memory instructions from CLAUDE.md
CLAUDE_MD="$HOME/.claude/CLAUDE.md"
if [ -f "$CLAUDE_MD" ]; then
    # Remove the memory block (from "# Memory" to the end of the memory MCP section)
    sed -i '' '/^# Memory$/,/^.*All persistent knowledge goes through the memory MCP server\.$/d' "$CLAUDE_MD"
    # Clean up any trailing blank lines
    sed -i '' -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$CLAUDE_MD"
    # Remove file if empty
    [ -s "$CLAUDE_MD" ] || rm -f "$CLAUDE_MD"
    echo "Removed memory instructions from $CLAUDE_MD"
fi

echo "Done! Server removed."
echo "Note: Database preserved at ~/.claude/memory.sqlite"
echo "To also delete memories: rm ~/.claude/memory.sqlite*"
