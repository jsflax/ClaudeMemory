---
name: remember
description: Store a new memory in Engram. Use when the user explicitly wants to save something for future sessions.
argument-hint: [what to remember]
allowed-tools: mcp__memory__recall, mcp__memory__remember, mcp__memory__connect, mcp__memory__update, mcp__memory__graph
---

Store a new memory using the Engram memory MCP server.

## Steps

1. Parse `$ARGUMENTS` as the content to remember.
2. Infer the `project` from context — use the current project name for project-specific knowledge, "global" for cross-project preferences/patterns.
3. **Check for duplicates first**: `recall(query: "$ARGUMENTS", project: "<inferred>", limit: 3)` to see if this already exists. If a near-duplicate is found, offer to `update` the existing memory instead.
4. Choose an appropriate `topic` based on content (e.g., "preferences", "architecture", "debugging", "patterns", "conventions", "workflow").
5. Call `remember(content: "...", project: "<inferred>", topic: "<chosen>")`. Set `importance` (1-5) based on how broadly useful this is.
6. After storing, `recall` related memories and `connect` them with appropriate edges (`relates_to`, `part_of`, `supersedes`, `contradicts`, `derived_from`).
7. Confirm what was stored and any connections made.

## Guidelines

- Keep memories atomic — one concept per memory.
- For complex topics, create a hub memory first, then store details as children using `parent_id`.
- Set `expires_in_days` for temporary context (current tasks, open PRs, in-progress work).
