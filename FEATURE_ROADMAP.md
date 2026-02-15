# Feature Roadmap

Evolving ClaudeMemory from a memory store into a fully-fledged brain for Claude persistence.

## Completed

### Fine-Grained Memory Editing
- Edit by ID with `update` — direct targeting, no similarity guessing
- Partial content edits: `append`, `prepend`, `find`+`replace` modes
- Field-level updates: change `topic`, `project`, `source`, `importance`, `expires_in_days` without touching content
- Re-embed on content change, skip for metadata-only changes

### Knowledge Graph
- `edges` table with `(source_id, target_id, relation)` — six relation types: `relates_to`, `contradicts`, `supersedes`, `derived_from`, `part_of`, `summarized_by`
- `connect` / `disconnect` tools for creating and removing edges
- `graph` tool to view a memory's neighborhood at configurable depth
- `recall` with `depth: 1-3` follows edges via BFS to surface connected knowledge
- Hierarchical memories via `parent_id` (auto-creates `part_of` edges)
- Edges auto-cleanup on forget/merge

### Reinforcement / Importance Scoring
- `accessCount` and `importance` fields on Memory model
- Scoring: cosine similarity blended with frequency (log-scaled, 15% boost), importance (1-5, 20% boost), recency (exponential decay, 10% boost)
- Recall ranking uses reinforcement signals to fine-tune ordering among close matches

### Conflict Detection
- `remember` checks for near-duplicates using both cosine distance (< 0.12 same-project, < 0.05 cross-scope) AND Jaccard term overlap (40%+ shared terms)
- Blocks storage with a warning showing the existing memory; `force: true` to override

### Hybrid Search (Semantic + Full-Text)
- FTS5 full-text index on `content` field
- Recall combines FTS5 (any matching term) with vector cosine similarity
- Falls back to FTS5-only if embedding model is unavailable

### Episodic Memory
- `Episode` model with title, summary, project, timestamps
- `begin_episode` / `end_episode` / `recall_episode` / `list_episodes` tools
- Auto-episodes: first `remember` creates one; >30 minute gap or project switch starts a new one

### Task Continuity
- `Checkpoint` model with plan, progress, context, status
- `checkpoint` / `resume` / `list_tasks` tools
- Tasks persist across conversations with active/paused/completed status

### Temporal Queries
- `timeline` tool with chronological view grouped by day/week/month
- `since` / `before` temporal filters on recall
- Calendar-style grouping with project and topic filters

### Clustering & Consolidation
- `find_clusters` uses greedy cosine-distance + Jaccard term overlap clustering
- `consolidate` creates a summary memory, deprioritizes originals (importance -> 0), links with `summarized_by` edges

### Memory Visualizer
- Interactive force-directed graph (SwiftUI Canvas, 60fps)
- Project-colored nodes with inter-project repulsion for visual separation
- FTS5-backed search with prefix matching, highlight mode
- Edge type filtering with color-coded relations
- Time slider with play button to animate graph growth
- Semantic cluster hulls, minimap, detail panel, keyboard shortcuts
- PNG export

## Not Yet Implemented

### Confidence Levels and Provenance Tracking
- `confidence` field (0.0-1.0) that increases with confirmation across sessions, decreases with conflicting evidence
- Provenance chain tracking which conversations/files/events created or reinforced a memory
- Low-confidence memories flagged for verification

### Proactive Surfacing / Triggers
- Register trigger patterns that auto-surface memories when context matches
- Safety triggers ("never force-push to main"), convention triggers ("this project uses tabs")
- `add_trigger` / `remove_trigger` tools

### Multi-Type Memory
- Typed memories: `fact`, `code_pattern`, `error_resolution`, `decision`, `preference`, `procedure`
- Structured fields per type (e.g., error_resolution has symptom, root_cause, fix, prevention)
- Type-aware retrieval and display formatting

### Memory Validation / Truth Maintenance
- Verify stored memories against current codebase state (file paths, function signatures, dependency versions)
- Auto-validation on project recall, spot-checking verifiable memories
- Stale memories flagged or auto-expired
- Git integration for detecting significant file changes
