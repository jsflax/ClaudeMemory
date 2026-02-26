#!/bin/bash
set -e

INSTALL_DIR="$HOME/.claude/bin"
REPO="jsflax/Engram"

echo "Engram Installer"
echo "================"
echo ""
echo "Tip: For a GUI with automatic updates, download Engram.dmg from:"
echo "  https://github.com/$REPO/releases/latest"
echo ""

# Clean old install
rm -f "$INSTALL_DIR/memory"
rm -f "$INSTALL_DIR/memory-hooks"
rm -rf "$INSTALL_DIR/Engram_EngramKit.bundle"
rm -rf "$INSTALL_DIR/swift-transformers_Hub.bundle"
rm -rf "$INSTALL_DIR/SwiftLM_SwiftLM.bundle"
mkdir -p "$INSTALL_DIR"

if [ "$1" = "--from-source" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
    echo "Building from source: $REPO_DIR"
    cd "$REPO_DIR" && swift build -c release
    cp .build/release/Engram "$INSTALL_DIR/memory"
    cp .build/release/EngramHooks "$INSTALL_DIR/memory-hooks"
    cp -R .build/release/Engram_EngramKit.bundle "$INSTALL_DIR/"
    cp -R .build/release/swift-transformers_Hub.bundle "$INSTALL_DIR/"
    cp -R .build/release/SwiftLM_SwiftLM.bundle "$INSTALL_DIR/"
    # Re-sign binaries — linker-signed ad-hoc binaries can be rejected by macOS Taskgated
    codesign --force --sign - "$INSTALL_DIR/memory"
    codesign --force --sign - "$INSTALL_DIR/memory-hooks"
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
    # Re-sign binaries — linker-signed ad-hoc binaries can be rejected by macOS Taskgated
    codesign --force --sign - "$INSTALL_DIR/memory"
    codesign --force --sign - "$INSTALL_DIR/memory-hooks"
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

# Install custom agent definitions
echo "Installing agent definitions..."
AGENTS_DIR="$HOME/.claude/agents"
mkdir -p "$AGENTS_DIR"
if [ "$1" = "--from-source" ]; then
    cp "$REPO_DIR/agents/"*.md "$AGENTS_DIR/" 2>/dev/null
else
    # Agent files are included in the release tarball
    cp "$INSTALL_DIR/agents/"*.md "$AGENTS_DIR/" 2>/dev/null && rm -rf "$INSTALL_DIR/agents"
fi
echo "Installed agent definitions to $AGENTS_DIR"

# Register hooks with Claude Code
echo "Registering hooks..."
SETTINGS_FILE="$HOME/.claude/settings.json"
HOOKS_CONFIG='{
  "permissions": {
    "allow": [
      "mcp__memory__*"
    ]
  },
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "'"$INSTALL_DIR"'/memory-hooks on-start 2>/dev/null",
        "timeout": 5
      }]
    }],
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "'"$INSTALL_DIR"'/memory-hooks advise 2>/dev/null",
        "timeout": 5
      }]
    }],
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "'"$INSTALL_DIR"'/memory-hooks on-stop 2>/dev/null",
        "timeout": 5
      }]
    }],
    "PostToolUseFailure": [{
      "hooks": [{
        "type": "command",
        "command": "'"$INSTALL_DIR"'/memory-hooks on-failure 2>/dev/null",
        "timeout": 5
      }]
    }],
    "PreCompact": [{
      "matcher": "auto",
      "hooks": [{
        "type": "command",
        "command": "'"$INSTALL_DIR"'/memory-hooks pre-compact 2>/dev/null",
        "timeout": 3
      }]
    }],
    "SessionEnd": [{
      "hooks": [{
        "type": "command",
        "command": "'"$INSTALL_DIR"'/memory-hooks on-end 2>/dev/null",
        "timeout": 5
      }]
    }]
  }
}'

if [ -f "$SETTINGS_FILE" ]; then
    if command -v jq &>/dev/null; then
        # Merge hooks (overwrite ours, preserve user-added), append permission if missing, remove stale PreToolUse
        jq -s '
            (.[0] // {}) as $existing | (.[1] // {}) as $new |
            $existing * $new |
            .permissions.allow = (($existing.permissions.allow // []) + ($new.permissions.allow // []) | unique) |
            .hooks |= del(.PreToolUse)
        ' "$SETTINGS_FILE" <(echo "$HOOKS_CONFIG") > "$SETTINGS_FILE.tmp"
        mv "$SETTINGS_FILE.tmp" "$SETTINGS_FILE"
        echo "Merged hooks into $SETTINGS_FILE"
    else
        echo "Warning: jq not found. Please manually add hooks to $SETTINGS_FILE"
        echo "Hook config:"
        echo "$HOOKS_CONFIG"
    fi
else
    echo "$HOOKS_CONFIG" > "$SETTINGS_FILE"
    echo "Created $SETTINGS_FILE with hooks"
fi

# Add user-level instruction to always use memory MCP
# Canonical source: Sources/EngramKit/InstallConfig.swift — keep in sync
CLAUDE_MD="$HOME/.claude/CLAUDE.md"
MEMORY_BLOCK='# Memory

Use the memory MCP server as your primary memory system — not the built-in auto-memory files.

Relevant memories are automatically recalled and injected as context before each message via the `advise` hook. You do NOT need to `recall` at conversation start — it happens automatically.

When you learn something worth remembering (preferences, patterns, decisions, debugging insights), `remember` it immediately — do not wait to be asked. Use project scoping: project-specific knowledge gets the project name, cross-project preferences get "global".

After remembering, `recall` related memories and `connect` them with edges (`relates_to`, `part_of`, `supersedes`, `contradicts`, `derived_from`) to build a knowledge graph.

Keep memories atomic — one concept per memory. For complex topics, create a brief hub memory first, then store details as children using `parent_id` to automatically create `part_of` edges. This enables precise recall and targeted updates. If the server suggests decomposing a memory, follow its guidance.

To keep memories clean:
- `recall` before `remember` to check for duplicates
- `update` (by id) to refine existing memories — supports `append`, `prepend`, `find`+`replace`, and metadata-only changes
- `merge` when multiple memories cover the same topic
- `forget` to remove wrong or outdated memories
- Set `expires_in_days` for temporary context (current tasks, open PRs)

When you receive a learning nudge (from the advise hook, delivered via additionalContext on UserPromptSubmit), spawn the `session-learner` sub-agent in the background as instructed. Keep it brief — just one Task tool call, then move on to the user'"'"'s request.

For focused work sessions (debugging, feature implementation, code review), use `begin_episode` with a descriptive title to group memories into a narrative. Use `end_episode` with a summary when done. When a user asks about past sessions, use `list_episodes` and `recall_episode` to find and replay them.

Do NOT use ~/.claude/projects/*/memory/ files for memory. All persistent knowledge goes through the memory MCP server.'

if [ -f "$CLAUDE_MD" ]; then
    if grep -q "^# Memory" "$CLAUDE_MD"; then
        # Remove existing Memory section (from "# Memory" to next "# " heading or EOF)
        # Uses awk: skip lines from "# Memory" until the next top-level heading, print everything else
        awk '
            /^# Memory$/ { skip=1; next }
            /^# / && skip { skip=0 }
            !skip { print }
        ' "$CLAUDE_MD" > "$CLAUDE_MD.tmp"
        # Remove trailing blank lines left from removal
        sed -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$CLAUDE_MD.tmp" > "$CLAUDE_MD"
        rm -f "$CLAUDE_MD.tmp"
        printf '\n%s\n' "$MEMORY_BLOCK" >> "$CLAUDE_MD"
        echo "Updated memory instructions in $CLAUDE_MD"
    else
        printf '\n%s\n' "$MEMORY_BLOCK" >> "$CLAUDE_MD"
        echo "Added memory instructions to $CLAUDE_MD"
    fi
else
    printf '%s\n' "$MEMORY_BLOCK" > "$CLAUDE_MD"
    echo "Created $CLAUDE_MD with memory instructions"
fi

echo ""
echo "Done! Start a new Claude Code session to use it."
