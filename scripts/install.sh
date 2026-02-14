#!/bin/bash
set -e

INSTALL_DIR="$HOME/.claude/bin"
REPO="jsflax/ClaudeMemory"

echo "ClaudeMemory Installer"
echo "======================"

# Clean old install
rm -f "$INSTALL_DIR/memory"
rm -rf "$INSTALL_DIR/ClaudeMemory_ClaudeMemoryLib.bundle"
rm -rf "$INSTALL_DIR/swift-transformers_Hub.bundle"
rm -rf "$INSTALL_DIR/SwiftLM_SwiftLM.bundle"
mkdir -p "$INSTALL_DIR"

if [ "$1" = "--from-source" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
    echo "Building from source: $REPO_DIR"
    cd "$REPO_DIR" && swift build -c release
    cp .build/release/ClaudeMemory "$INSTALL_DIR/memory"
    cp -R .build/release/ClaudeMemory_ClaudeMemoryLib.bundle "$INSTALL_DIR/"
    cp -R .build/release/swift-transformers_Hub.bundle "$INSTALL_DIR/"
    cp -R .build/release/SwiftLM_SwiftLM.bundle "$INSTALL_DIR/"
else
    echo "Downloading latest release..."
    DOWNLOAD_URL=$(curl -sL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep '"browser_download_url"' \
        | grep 'arm64' \
        | head -1 \
        | cut -d'"' -f4)

    if [ -z "$DOWNLOAD_URL" ]; then
        echo "Error: No release found. Use --from-source to build locally."
        exit 1
    fi

    echo "From: $DOWNLOAD_URL"
    curl -sL "$DOWNLOAD_URL" | tar xz -C "$INSTALL_DIR"
fi

echo "Installed to $INSTALL_DIR"

# Ensure parent dir exists
mkdir -p "$HOME/.claude"

# Register MCP server with Claude Code
echo "Registering MCP server..."
if ! command -v claude &>/dev/null; then
    echo "Warning: 'claude' CLI not found. Install Claude Code first, then re-run this script."
    echo "Binary installed at $INSTALL_DIR/memory — just needs MCP registration."
    exit 0
fi

env -u CLAUDECODE claude mcp remove memory 2>/dev/null || true
env -u CLAUDECODE claude mcp add --scope user --transport stdio memory -- "$INSTALL_DIR/memory"

# Add user-level instruction to always use memory MCP
CLAUDE_MD="$HOME/.claude/CLAUDE.md"
MEMORY_BLOCK='# Memory

Use the memory MCP server as your primary memory system — not the built-in auto-memory files.

At the START of every conversation, before responding to the first message:
1. `recall` with the current project name and `depth: 1` to load project context + global preferences with graph connections
2. Use what you learn to inform your responses

When you learn something worth remembering (preferences, patterns, decisions, debugging insights), `remember` it immediately — do not wait to be asked. Use project scoping: project-specific knowledge gets the project name, cross-project preferences get "global".

After remembering, `recall` related memories and `connect` them with edges (`relates_to`, `part_of`, `supersedes`, `contradicts`, `derived_from`) to build a knowledge graph.

Keep memories atomic — one concept per memory. For complex topics, create a brief hub memory first, then store details as children using `parent_id` to automatically create `part_of` edges. This enables precise recall and targeted updates. If the server suggests decomposing a memory, follow its guidance.

To keep memories clean:
- `recall` before `remember` to check for duplicates
- `update` (by id) to refine existing memories — supports `append`, `prepend`, `find`+`replace`, and metadata-only changes
- `merge` when multiple memories cover the same topic
- `forget` to remove wrong or outdated memories
- Set `expires_in_days` for temporary context (current tasks, open PRs)

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
echo "Done! Start a new Claude Code session to use it."
