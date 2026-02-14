# Feature Roadmap

Evolving ClaudeMemory from a memory store into a fully-fledged brain for Claude persistence.

## Tier 1: High-Impact, Builds on Current Architecture

### 1. Fine-Grained Memory Editing

`update` currently only works by similarity match and replaces the entire content. Claude needs precise, surgical control over memories.

- **Edit by ID**: `update` accepts an `id` parameter for direct targeting (no similarity guessing)
- **Partial content edits**: `append`, `prepend`, and `find_replace` modes alongside full replacement
- **Field-level updates**: change `topic`, `project`, `source`, or `expires_in_days` without touching content
- **Batch operations**: update multiple memories at once (e.g., re-topic a set of memories, move memories between projects)
- **Re-embed on content change**: automatically recompute embedding when content is modified, skip re-embedding for metadata-only changes
- Existing similarity-based update preserved as a convenience, but ID-based is the primary path

### 2. Memory Relationships / Knowledge Graph

Memories currently exist in isolation. A brain connects ideas.

- Add an `edges` table with `(source_id, target_id, relation_type)` columns
- Relation types: `relates_to`, `contradicts`, `supersedes`, `derived_from`, `part_of`
- New `connect` tool to create edges between memories
- Enhance `recall` to optionally traverse edges (depth-limited graph walk)
- Query patterns: "recall everything connected to the auth refactor" follows edges, not just similarity
- Visualization via `graph` tool: show a memory's neighborhood

### 3. Reinforcement / Importance Scoring

`lastAccessedAt` exists but isn't used for ranking. Memories that keep proving useful should float to the top.

- Add `accessCount` and `importance` fields to Memory model
- Scoring formula: `score = similarity * (recency_weight + frequency_weight + explicit_importance)`
- Recency: exponential decay from `lastAccessedAt`
- Frequency: logarithmic scale on `accessCount`
- Explicit importance: optional 1-5 rating set on `remember` or `update`
- Blend scoring into recall ranking alongside cosine similarity
- Memories that haven't been recalled in months naturally decay

### 4. Automatic Conflict Detection

`remember` currently always creates new entries, even if they contradict existing knowledge.

- On `remember`, check for semantic near-duplicates (cosine > 0.85 threshold)
- If near-duplicate found, return a warning with the existing memory and ask to `update` or keep both
- Detect contradictions: high similarity but opposing sentiment/content
- New `contradicts` edge type automatically created when both are kept
- Configurable: `auto_dedup: true` to silently update, or `prompt` mode (default) to surface conflicts

### 5. Hybrid Search (Semantic + Full-Text + Metadata)

Pure vector search misses exact keyword matches. Pure keyword search misses meaning.

- Add FTS5 full-text index on `content` field
- Recall scoring: weighted combination of cosine similarity + FTS5 rank + metadata match
- Exact identifiers (function names, file paths, error codes) matched precisely via FTS5
- Semantic meaning matched via existing vector search
- Metadata filters (project, topic, date range) applied as pre-filters before scoring
- Configurable weights per query or globally

## Tier 2: Makes It Feel Like Cognition

### 6. Episodic Memory / Session Chains

Group related memories from a single work session into coherent episodes.

- New `Episode` model: `(id, title, summary, project, startedAt, endedAt)`
- `Memory` gets optional `episodeId` foreign key
- New `begin_episode` / `end_episode` tools to bracket a work session
- Auto-episode: if no explicit episode, group memories by time proximity (e.g., within 30 min)
- `recall_episode` tool: retrieve a full episode as a linked narrative
- Episode summaries auto-generated from member memories
- Enables "recall the debugging session where we found the race condition"

### 7. Automatic Summarization / Consolidation

As memories accumulate, consolidate clusters into higher-level abstractions.

- Periodic maintenance: cluster memories by semantic similarity
- Clusters above a size threshold (e.g., 5+ memories on same topic) trigger consolidation
- Generate a summary memory that captures the gist of the cluster
- Original memories marked with `summarized_by` edge to the summary
- Originals optionally archived (kept but excluded from default recall)
- New `consolidate` tool to manually trigger for a topic/project
- Mimics human memory consolidation: details fade, patterns persist

### 8. Temporal Queries

Time as a first-class query dimension, not just metadata.

- New query syntax for time ranges: `recall --since "last week"` / `--before "2024-01-01"`
- Timeline view: `timeline` tool shows memories chronologically for a project
- Change tracking: "how has the API design evolved?" returns memories sorted by time with diffs
- Recency-weighted recall mode: prioritize recent memories when explicitly requested
- Calendar-style grouping: memories by day/week/month

### 9. Task Continuity / Work-in-Progress State

Checkpoint ongoing tasks so Claude can pick up where it left off.

- New `Task` model: `(id, title, status, plan, progress, project, checkpointedAt)`
- `checkpoint` tool: save current task state (what's done, what's left, current approach)
- `resume` tool: load the most recent checkpoint for a project/task
- Auto-checkpoint: before session ends, prompt to save WIP state
- Task memories linked to relevant episodic and semantic memories
- Enables genuine continuity: "I was halfway through migrating the database layer"

## Tier 3: Advanced

### 10. Confidence Levels and Provenance Tracking

Not all memories are equally trustworthy.

- Add `confidence` field (0.0-1.0) to Memory model
- Confidence increases when a memory is confirmed across multiple sessions
- Confidence decreases when conflicting evidence appears
- Provenance chain: track which conversations/files/events created or reinforced a memory
- Display confidence in recall results: `[id:5 confidence:high]`
- Low-confidence memories can be flagged for verification

### 11. Proactive Surfacing / Triggers

Shift from passive recall to active cognitive assistance.

- Register trigger patterns: `trigger("force push", memory_id)`
- When Claude's context matches a trigger pattern, automatically surface the memory
- Safety triggers: "user prefers to never force-push to main" fires before `git push --force`
- Convention triggers: "this project uses tabs" fires when writing new files
- Implemented as a `triggers` table with pattern + memory_id + priority
- New `add_trigger` / `remove_trigger` tools

### 12. Multi-Type Memory

Typed memories with schema-appropriate handling.

- Memory subtypes: `fact`, `code_pattern`, `error_resolution`, `decision`, `preference`, `procedure`
- Each type has structured fields:
  - `code_pattern`: snippet, language, context, when_to_use
  - `error_resolution`: symptom, root_cause, fix, prevention
  - `decision`: options_considered, chosen, rationale, outcome
  - `procedure`: steps, preconditions, postconditions
- Type-aware retrieval: searching for errors prioritizes `error_resolution` memories
- Type-specific display formatting in recall results

### 13. Memory Validation / Truth Maintenance

Stored memories can become stale. Verify against current reality.

- Verifiable memory types: file paths, function signatures, dependency versions, config values
- `validate` tool: checks verifiable memories against current codebase state
- Auto-validation: on project recall, spot-check a sample of verifiable memories
- Stale memories flagged with `[stale]` in recall or auto-expired
- Validation log: track what was checked and when
- Integrates with git: detect when referenced files have changed significantly

## Implementation Priority

1. **Fine-Grained Editing** — low complexity, immediately useful, unblocks precise memory management
2. **Relationships / Knowledge Graph** — most transformative, builds on existing SQLite
3. **Conflict Detection** — prevents bad data from accumulating
4. **Hybrid Search** — immediate recall quality improvement via FTS5
5. **Reinforcement Scoring** — makes recall smarter over time
6. **Episodic Memory** — enables session continuity
7. **Task Continuity** — enables genuine pick-up-where-you-left-off
8. **Temporal Queries** — natural extension of existing timestamp fields
9. **Summarization** — important for long-term scaling
10. **Multi-Type Memory** — richer data model
11. **Confidence Tracking** — quality improvement
12. **Proactive Triggers** — paradigm shift from pull to push
13. **Validation** — long-term maintenance
