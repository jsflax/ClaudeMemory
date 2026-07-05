---
name: recall
description: Search Engram memory for relevant context. Use when the user wants to look up what's been stored about a topic, project, or past session.
argument-hint: [query]
allowed-tools: mcp__memory__recall, mcp__memory__list_topics, mcp__memory__list_episodes, mcp__memory__recall_episode, mcp__memory__graph, mcp__memory__stats, mcp__memory__timeline
---

Recall memories matching the user's query using the Engram memory MCP server.

## Steps

1. Parse `$ARGUMENTS` as the recall query. If empty, ask the user what to search for.
2. Infer the `project` from the current working directory name (e.g., `~/Projects/Engram` → project: "Engram"). If the query is clearly cross-project or about preferences, use project: "global".
3. Call `recall(query: "$ARGUMENTS", project: "<inferred>", limit: 10, depth: 1)` to get results with graph neighbors.
4. Present the results clearly — show memory IDs, content summaries, topics, and any graph connections.
5. If results are sparse, try `list_topics(project: "<inferred>")` to suggest alternative queries, or broaden with `recall(query: "$ARGUMENTS", limit: 10)` without project filter.
