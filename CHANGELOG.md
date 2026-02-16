# Changelog

All notable changes to ClaudeMemory are documented in this file.

## [Unreleased]

## [0.7.1] - 2026-02-15

### Added
- MemoryVisualizer: activity log panel — persistent scrollable list of recent memories with animated slide-in, cyan glow on new entries, and click-to-navigate

### Changed
- MemoryVisualizer: replace transient toast notifications with the persistent activity log
- MemoryVisualizer: raise normal label zoom threshold from 1.4x to 1.8x for cleaner overview
- README: use `swift run -c release` for MemoryVisualizer launch commands

### Improved
- MemoryVisualizer: extract search bar into isolated view to prevent expensive recomputation on each keystroke
- MemoryVisualizer: move cluster computation to reactive state for snappier search typing
- Hooks: simplified hook handlers, removed unused Analyze/TranscriptParser modules

## [0.7.0] - 2026-02-15

### Added
- **`organize` tool** — batch re-topic memories with automatic hub creation and `part_of` edge linking
- **Cross-project hub linking** — `remember` auto-links memories to other projects' hubs when content mentions them by name (word-boundary, case-insensitive)
- **Hooks binary** (`memory-hooks`) — `advise` hook injects recalled memories as context before each message; `analyze` hook nudges session learning after substantive interactions
- **Custom agents** — `session-learner` and `memory-maintenance` agent definitions for background work
- Install script improvements: better update detection, hooks binary installation, agents directory

### Changed
- Episodes refactored to hub memories (topic: "episode") linked via `part_of` edges — no separate Episode model
- Label propagation uses synchronous updates for deterministic community detection across bridge edges

### Improved
- MemoryVisualizer: smaller node label fonts, material background on stats overlay, bold hub labels, golden-angle colors, trackpad gestures, better clustering layout

### Fixed
- Replace force-unwraps with safe optional handling across MCP tool handlers
- Improve MCP error reporting with descriptive messages instead of crashes
- Label propagation bridge-edge test now passes (synchronous updates prevent label cascading)

## [0.6.0] - 2026-02-15

### Added
- MemoryVisualizer: search, filtering, clustering visualization, and performance fixes
- Pre-compile CoreML embedding model at build time for fast MCP startup
- Lazy-load embeddings on first use instead of at initialization

## [0.5.2] - 2026-02-14

### Changed
- Update Lattice dependency to versioned 0.1.0

## [0.5.1] - 2026-02-14

### Fixed
- Install script now updates existing memory instructions instead of skipping them

## [0.5.0] - 2026-02-14

### Added
- Jaccard term-overlap similarity to conflict detection (reduces false positives from topically similar but distinct memories)
- MemoryVisualizer macOS app for exploring the knowledge graph
- Demo GIF in README

## [0.4.4] - 2026-02-14

### Added
- `parent_id` parameter on `remember` for hierarchical memories (auto-creates `part_of` edges)
- Atomic memory nudges in system instructions (one concept per memory guidance)
- Smarter graph display with hub/detail hierarchy

## [0.4.3] - 2026-02-14

### Fixed
- Add consolidation guardrails to prevent over-consolidation of distinct memories

## [0.4.2] - 2026-02-14

### Fixed
- Parse `ids` parameter correctly when MCP client sends comma-separated string instead of array (affects `merge` and `consolidate`)

## [0.4.1] - 2026-02-14

### Added
- `summarized_by` relation type for the `connect` tool
- README updated with all 20 tools, Tier 2 features, and hybrid search documentation

## [0.4.0] - 2026-02-14

### Added
- **Tier 2 features**: temporal queries (`timeline`), task continuity (`checkpoint`/`resume`/`list_tasks`), episodic memory (`begin_episode`/`end_episode`/`recall_episode`/`list_episodes`), and clustering (`find_clusters`/`consolidate`/`detect_communities`)

## [0.3.0] - 2026-02-14

### Added
- Knowledge graph with directed edges (`connect`, `disconnect`, `graph` tools)
- Relation types: `relates_to`, `contradicts`, `supersedes`, `derived_from`, `part_of`
- Fine-grained `update` tool with `append`, `prepend`, `find`+`replace`, and metadata-only modes
- Conflict detection on `remember` using embedding distance thresholds
- Reinforcement and importance scoring in recall ranking

## [0.1.0] - 2026-02-13

### Added
- Initial release — local MCP memory server for Claude Code
- Core memory tools: `remember`, `recall`, `forget`, `update`, `merge`, `list_topics`, `stats`
- Hybrid search combining semantic embeddings (CoreML) and full-text search (FTS5)
- SQLite-backed persistent storage with per-project scoping
- Pre-built universal binary releases for macOS
- One-liner install script (`scripts/install.sh`)
- CI/CD with GitHub Actions on macOS 26 / Swift 6.2
- Homebrew tap workflow

[Unreleased]: https://github.com/jsflax/ClaudeMemory/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/jsflax/ClaudeMemory/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jsflax/ClaudeMemory/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/jsflax/ClaudeMemory/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/jsflax/ClaudeMemory/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/jsflax/ClaudeMemory/compare/v0.4.4...v0.5.0
[0.4.4]: https://github.com/jsflax/ClaudeMemory/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/jsflax/ClaudeMemory/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/jsflax/ClaudeMemory/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/jsflax/ClaudeMemory/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/jsflax/ClaudeMemory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jsflax/ClaudeMemory/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/jsflax/ClaudeMemory/releases/tag/v0.1.0
