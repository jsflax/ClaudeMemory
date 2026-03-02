---
name: memory-status
description: Show Engram memory statistics — memory counts, topic distribution, recent activity, and active tasks.
allowed-tools: mcp__memory__stats, mcp__memory__list_topics, mcp__memory__timeline, mcp__memory__list_tasks, mcp__memory__list_episodes
---

Show a dashboard of the user's Engram memory system.

## Steps

1. Run these in parallel:
   - `stats()` — memory counts by project and topic
   - `list_topics()` — all topics with counts
   - `timeline(period: "week", limit: 4)` — recent memory activity
   - `list_tasks()` — any active/paused tasks
   - `list_episodes()` — recent episodes

2. Present a concise dashboard:
   - **Overview**: total memories, projects, topics
   - **By project**: top projects by memory count
   - **Recent activity**: what was stored/updated this week
   - **Active tasks**: any in-progress or paused tasks
   - **Open episodes**: any active episodes

3. If any project has a topic with >15 memories, flag it as a candidate for consolidation.
