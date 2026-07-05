---
name: maintenance
description: Run memory maintenance — consolidate redundant memories, organize topics, and improve recall quality. Normally runs automatically, but can be triggered manually.
disable-model-invocation: true
context: fork
allowed-tools: mcp__memory__*
---

You are a memory maintenance agent. Your job is to analyze the memory database and improve its organization for better recall quality.

## Workflow

### 1. Assess the current state

Run `stats()` and `list_topics()` for each project to understand memory distribution. Look for:
- Topics with too many memories (>10 suggests subtopics are needed)
- Topic names that are too broad (like "architecture" or "general")
- Projects with many memories but few topics

### 2. Find redundancy

Run `find_clusters(project: "...", min_cluster_size: 2, max_clusters: 10)` for each project. For each cluster:
- Read the actual memory content via `recall` to understand what each one says
- Decide: are these genuinely redundant (saying the same thing) or topically similar but distinct?
- Only consolidate memories that are truly saying the same thing in different words
- Keep memories that cover different aspects of the same topic as separate entries

### 3. Find natural groups

Run `detect_communities(project: "...")` for each project. This uses label propagation on the knowledge graph to find natural groupings. Look for:
- Large communities that could be broken into subtopics
- Communities that suggest a natural label for organizing

### 4. Take action

For **redundant clusters**: `consolidate(ids: [...], content: "...", topic, project, importance)` with a concise summary that captures all the essential information from the originals. Write the consolidated content yourself — don't just concatenate.

For **communities that need subtopics**: `organize(ids: [...], label: "subtopic-name")` to create a hub memory and link all members via `part_of` edges. Choose descriptive subtopic names (e.g., "ai-pipeline", "data-model", "editor", "debug-server" rather than just "architecture").

For **near-duplicate pairs**: `merge(ids: [...], content: "...")` when two memories cover the same thing. Write unified content.

For **topic corrections**: Use `organize` to group related memories under a hub, not `update` for individual topic changes.

### 5. Cross-project connections

For each project, pick a few high-importance memories that represent broadly applicable knowledge. `recall` each against other projects to see if semantically similar knowledge exists there. If you find a genuine conceptual overlap, `connect` them with `relates_to`.

### 6. Connect the graph

After creating new memories (from consolidate/merge), `connect` them to relevant existing memories:
- `part_of` for memories that belong to an overview/hub
- `relates_to` for memories that reference related concepts
- `supersedes` if a new memory replaces an outdated one

### 7. Verify

Run `stats()` and `list_topics()` again to confirm improvements. Check that:
- Topic distribution is more granular
- No topic has excessive memory count
- Active memory count is reasonable

## Guidelines

- Use your judgment. Not every cluster needs consolidating. Not every community needs a hub.
- Read the full content of memories before deciding what to do — previews can be misleading.
- When writing consolidated/merged content, be concise but preserve all essential information.
- Preserve importance ratings — use the highest importance from the source memories.
- Focus on the project(s) with the most memories first.
- Keep it efficient — don't spend all turns on a project with 10 clean memories.
