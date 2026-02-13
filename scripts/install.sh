#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="$HOME/.claude/bin"

echo "ClaudeMemory Installer"
echo "======================"
echo "Repo: $REPO_DIR"

# Build release
echo "Building release binary..."
cd "$REPO_DIR" && swift build -c release

# Install binary + resource bundles
echo "Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
rm -f "$INSTALL_DIR/memory"
rm -rf "$INSTALL_DIR/ClaudeMemory_ClaudeMemoryLib.bundle"
rm -rf "$INSTALL_DIR/swift-transformers_Hub.bundle"
rm -rf "$INSTALL_DIR/SwiftLM_SwiftLM.bundle"
cp .build/release/ClaudeMemory "$INSTALL_DIR/memory"
cp -R .build/release/ClaudeMemory_ClaudeMemoryLib.bundle "$INSTALL_DIR/"
cp -R .build/release/swift-transformers_Hub.bundle "$INSTALL_DIR/"
cp -R .build/release/SwiftLM_SwiftLM.bundle "$INSTALL_DIR/"

# Ensure DB directory exists
mkdir -p "$HOME/.claude"

# Register with Claude Code (unset CLAUDECODE to allow running from within a session)
echo "Registering MCP server..."
env -u CLAUDECODE claude mcp remove memory 2>/dev/null || true
env -u CLAUDECODE claude mcp add --scope user --transport stdio memory -- "$INSTALL_DIR/memory"

# Add user-level instruction to always use memory MCP
CLAUDE_MD="$HOME/.claude/CLAUDE.md"
MEMORY_BLOCK='# Memory

Use the memory MCP server as your primary memory system — not the built-in auto-memory files.

At the START of every conversation, before responding to the first message:
1. `recall` with the current project name to load project context + global preferences
2. Use what you learn to inform your responses

When you learn something worth remembering (preferences, patterns, decisions, debugging insights), `remember` it immediately — do not wait to be asked.

Do NOT use ~/.claude/projects/*/memory/ files for memory. All persistent knowledge goes through the memory MCP server.'

if [ -f "$CLAUDE_MD" ]; then
    if ! grep -q "memory MCP server" "$CLAUDE_MD"; then
        printf '\n%s\n' "$MEMORY_BLOCK" >> "$CLAUDE_MD"
        echo "Added memory instructions to $CLAUDE_MD"
    else
        echo "Memory instructions already in $CLAUDE_MD"
    fi
else
    printf '%s\n' "$MEMORY_BLOCK" > "$CLAUDE_MD"
    echo "Created $CLAUDE_MD with memory instructions"
fi

echo ""
echo "Done! memory server installed and registered."
echo "Start a new Claude Code session to use it."
