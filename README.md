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
| **Maintenance** | Append-only text that gets stale | `update`, `merge`, `forget`, auto-expiring memories, conflict detection |
| **Structure** | Flat text, no relationships | Knowledge graph — connect memories with typed edges, traverse on recall |
| **Privacy** | Plain text files | Local SQLite + on-device embeddings (MiniLM-L6). Nothing leaves your machine |

**The one-liner:** MEMORY.md is 200 lines of flat text that gets stale. ClaudeMemory is a vector database that scales, searches semantically, and self-maintains.

## Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with semantic embedding. Conflict detection blocks near-duplicates (`force: true` to override) |
| `recall` | Semantic search with soft project boosting and optional graph traversal (`depth: 1-3`) |
| `forget` | Delete by ID, topic, or project. Cascades edge cleanup |
| `update` | Edit by ID or similarity — full replace, `append`, `prepend`, `find`+`replace`, or metadata-only (`topic`, `source`, `expires_in_days`) |
| `merge` | Consolidate multiple memories into one. Cleans up source edges |
| `connect` | Create a directed edge between memories (`relates_to`, `contradicts`, `supersedes`, `derived_from`, `part_of`) |
| `disconnect` | Remove edges by edge ID or by (from, to) pair |
| `graph` | View a memory's neighborhood — shows connections up to a given depth |
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
- **Scoping**: Memories have `project` and `topic` fields. Project is a soft ranking signal — same-project and global memories rank higher, but cross-project results still surface if semantically relevant.
- **Knowledge graph**: Connect memories with typed directed edges. Recall with `depth > 0` follows edges via BFS to surface connected knowledge. Edges auto-cleanup on forget/merge.
- **Conflict detection**: `remember` checks for near-duplicates (cosine distance < 0.12 same-project, < 0.05 cross-scope). Blocks storage with a warning; use `force: true` to override.
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
