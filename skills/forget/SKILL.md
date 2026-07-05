---
name: forget
description: Remove a memory from Engram by ID or by searching for it. Use when the user wants to delete outdated or incorrect memories.
argument-hint: [memory ID or search query]
disable-model-invocation: true
allowed-tools: mcp__memory__recall, mcp__memory__forget, mcp__memory__graph
---

Remove a memory from the Engram memory MCP server.

## Steps

1. Parse `$ARGUMENTS` — it may be a numeric memory ID (e.g., "42") or a search query.
2. If it's a numeric ID: call `graph(id: <id>, depth: 1)` to show the memory and its connections, then confirm with the user before deleting.
3. If it's a search query: `recall(query: "$ARGUMENTS", limit: 5)` to find candidates. Present results and ask the user which one(s) to forget.
4. After confirmation, call `forget(id: <id>)` for each memory to remove.
5. Report what was deleted.

## Safety

- Always show the memory content and ask for confirmation before deleting.
- Warn if the memory has graph connections (part_of, relates_to, etc.) that will be orphaned.
- Suggest `update` as an alternative if the memory just needs correction rather than removal.
