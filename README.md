# ClaudeMemory

A local MCP server that gives Claude Code persistent, semantic memory across sessions.

## Why not MEMORY.md?

Claude Code has built-in memory via `MEMORY.md` files. Here's why ClaudeMemory is better:

| | MEMORY.md | ClaudeMemory |
|---|---|---|
| **Retrieval** | Dumps entire file into system prompt | Semantic vector search — only relevant memories surfaced |
| **Capacity** | Truncated at 200 lines | Unlimited — stores thousands, retrieves the best matches |
| **Search** | Position-based (top of file = seen first) | Similarity-based (most relevant = seen first) |
| **Cross-project** | Per-project files, no sharing | Project scoping + global scope — preferences follow you everywhere |
| **Maintenance** | Append-only text that gets stale | `update`, `merge`, `forget`, auto-expiring memories |
| **Privacy** | Plain text files | Local SQLite + on-device embeddings (MiniLM-L6). Nothing leaves your machine |

**The one-liner:** MEMORY.md is 200 lines of flat text that gets stale. ClaudeMemory is a vector database that scales, searches semantically, and self-maintains.

## Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with semantic embedding. Supports project scoping, topics, and expiration |
| `recall` | Search memories by semantic similarity. Returns project-specific + global results |
| `forget` | Delete by ID, topic, project, or all |
| `update` | Find a memory by similarity and replace its content |
| `merge` | Consolidate multiple memories into one |
| `stats` | Database overview with per-project and per-topic breakdowns |
| `list_topics` | List all topics with counts |

## Install

```bash
curl -sL https://raw.githubusercontent.com/jsflax/ClaudeMemory/main/scripts/install.sh | bash
```

This downloads the pre-built binary, registers the MCP server, and configures Claude Code — takes a few seconds.

To build from source instead:

```bash
git clone https://github.com/jsflax/ClaudeMemory.git
cd ClaudeMemory
./scripts/install.sh --from-source
```

Start a new Claude Code session — memory tools are immediately available.

## Uninstall

```bash
curl -sL https://raw.githubusercontent.com/jsflax/ClaudeMemory/main/scripts/uninstall.sh | bash
```

Removes the binary and MCP registration. Database is preserved at `~/.claude/memory.sqlite` (delete manually if desired).

## How it works

- **Embedding model**: [paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) runs locally via CoreML. 384-dimensional vectors.
- **Storage**: SQLite via [Lattice](https://github.com/jflasher/Lattice) ORM with [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector search.
- **Transport**: stdio MCP — server starts with each Claude Code session, loads embedding model once, stays alive for the session duration.
- **Scoping**: Memories have `project` and `topic` fields. Recall with a project returns project-specific + global memories. Global memories (project: `"global"`) are always included.
- **Expiration**: Set `expires_in_days` for temporal context ("currently working on X"). Expired memories are filtered from recall automatically.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `CLAUDE_MEMORY_DB` | `~/.claude/memory.sqlite` | Database path |
| `CLAUDE_MEMORY_MODEL` | Bundled MiniLM-L6 | Custom embedding model path |

## Requirements

- macOS (CoreML for embeddings)
- Swift 6.0+
- Claude Code CLI
